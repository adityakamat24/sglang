"""
Suffix Decoding worker for SGlang.

This module implements suffix decoding, a training-free, CPU-only speculative
decoding method that wraps the Arctic Inference library.

Based on: https://arxiv.org/abs/2411.04975
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.suffix_info import SuffixVerifyInput

logger = logging.getLogger(__name__)


USE_FULL_MASK = True


class SuffixWorker:
    """
    Suffix Decoding worker for SGlang.

    Wraps Arctic Inference library for CPU-only, training-free speculation.
    Similar to NGRAMWorker but uses Arctic Inference SuffixDecodingCache.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Lazy import to avoid error when suffix decoding is not used
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.max_tree_depth: int = server_args.speculative_suffix_max_tree_depth
        self.max_spec_factor: float = server_args.speculative_suffix_max_spec_factor
        self.min_token_prob: float = server_args.speculative_suffix_min_token_prob

        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self._init_preallocated_tensors()

        # Initialize Arctic Inference cache
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=server_args.speculative_suffix_max_tree_depth,
            max_cached_requests=server_args.speculative_suffix_max_cached_requests,
        )

        logger.info(
            f"Initialized SuffixWorker with max_tree_depth={self.max_tree_depth}, "
            f"max_cached_requests={server_args.speculative_suffix_max_cached_requests}, "
            f"max_spec_factor={self.max_spec_factor}, "
            f"min_token_prob={self.min_token_prob}"
        )

    def clear_cache_pool(self):
        """Clear the suffix cache."""
        # Arctic Inference doesn't have a reset method, but we can clear active requests
        for req_id in list(self.suffix_cache.active_requests):
            self.suffix_cache.stop_request(req_id)

    def _efficient_concat_last_n(self, seq1: List[int], seq2: List[int], n: int):
        """Efficiently concatenate last n elements from two sequences."""
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]

        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _init_preallocated_tensors(self):
        """Preallocate tensors for efficient batch processing."""
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrive_next_token_batch = []
        self.retrive_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrive_next_token_batch.append(self.retrive_next_token[:bs, :])
            self.retrive_next_sibling_batch.append(self.retrive_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare draft tokens using Arctic Inference SuffixDecodingCache.

        This is the key adaptation from NGRAMWorker - we use SuffixDecodingCache
        instead of NgramCache.
        """
        bs = batch.batch_size()

        # Prepare batch for suffix cache
        req_drafts_list = []
        mask_list = []

        for req in batch.reqs:
            req_id = str(req.rid)  # Convert to string for Arctic Inference

            # Get current tokens (prompt + output)
            current_tokens = req.origin_input_ids + req.output_ids
            num_tokens = len(current_tokens)

            # Handle new requests
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    # Reset the suffix cache for this request
                    self.suffix_cache.evict_cached_response(req_id)

                # Start a new request with prompt tokens
                self.suffix_cache.start_request(req_id, req.origin_input_ids)

            # Add output tokens to suffix cache (if any)
            if len(req.output_ids) > 0:
                # Add the most recent output token
                self.suffix_cache.add_active_response(req_id, [req.output_ids[-1]])

            # Extract pattern from the end of the input
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = current_tokens[start:num_tokens]

            # Generate draft tokens using suffix cache
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=self.draft_token_num,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            # Convert draft to numpy array and pad to draft_token_num
            draft_ids = draft.token_ids
            draft_parents = draft.parents

            # Pad or truncate to match draft_token_num
            if len(draft_ids) < self.draft_token_num:
                # Pad with zeros if not enough drafts
                pad_len = self.draft_token_num - len(draft_ids)
                draft_ids = draft_ids + [0] * pad_len
                # Pad parents with -1 (no parent)
                draft_parents = draft_parents + [-1] * pad_len
            elif len(draft_ids) > self.draft_token_num:
                # Truncate if too many drafts
                draft_ids = draft_ids[: self.draft_token_num]
                draft_parents = draft_parents[: self.draft_token_num]

            req_drafts_list.extend(draft_ids)

            # Create tree mask from draft.parents structure
            # The parents field encodes the tree: parents[i] is the parent index of token i
            req_mask = np.zeros(
                (self.draft_token_num, self.draft_token_num), dtype=bool
            )

            # Build attention mask from parent structure
            # Token i can attend to token j if j is an ancestor of i in the tree
            for i in range(self.draft_token_num):
                # Each token can attend to itself
                req_mask[i, i] = True

                # Follow parent chain to mark all ancestors
                parent_idx = draft_parents[i]
                while parent_idx >= 0 and parent_idx < self.draft_token_num:
                    req_mask[i, parent_idx] = True
                    parent_idx = draft_parents[parent_idx]

            mask_list.append(req_mask)

        # Convert to numpy arrays
        req_drafts = np.array(req_drafts_list, dtype=np.int64)
        mask = np.concatenate([m.flatten() for m in mask_list])

        # Stop requests that were not seen in the batch
        active_req_ids = {str(req.rid) for req in batch.reqs}
        for req_id in list(self.suffix_cache.active_requests - active_req_ids):
            self.suffix_cache.stop_request(req_id)

        total_draft_token_num = len(req_drafts)
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"

        return req_drafts, mask

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        """
        Prepare batch for speculative decoding.

        This method is almost identical to NGRAMWorker's implementation,
        just using SuffixVerifyInput instead of NgramVerifyInput.
        """
        if batch.forward_mode.is_extend():
            return

        bs = batch.batch_size()

        retrive_index = self.retrieve_indexes_batch[bs]
        retrive_next_token = self.retrive_next_token_batch[bs]
        retrive_next_sibling = self.retrive_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        req_drafts, mask = self._prepare_draft_tokens(batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrive_index,  # mutable
            retrive_next_token,  # mutable
            retrive_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        if USE_FULL_MASK:
            tree_mask = []
            mask = mask.reshape(
                batch.batch_size(), self.draft_token_num, self.draft_token_num
            )
            for i, req in enumerate(batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                req_mask = torch.ones((self.draft_token_num, seq_len - 1)).cuda()
                req_mask = torch.cat(
                    (req_mask, torch.from_numpy(mask[i]).cuda()), dim=1
                ).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        batch.spec_algorithm = SpeculativeAlgorithm.from_string("SUFFIX")
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = SuffixVerifyInput(
            draft_tokens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            self.draft_token_num,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: SuffixVerifyInput,
        logits_output: LogitsProcessorOutput,
    ):
        """Add logprob values to the batch (same as NGRAMWorker)."""
        # Extract args
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        accepted_indices = res.accept_index
        assert len(accepted_indices) == len(logits_output.next_token_logits)

        temperatures = batch.sampling_info.temperatures
        num_draft_tokens = batch.spec_info.draft_token_num
        # acceptance indices are the indices in a "flattened" batch.
        # dividing it to num_draft_tokens will yield the actual batch index.
        temperatures = temperatures[accepted_indices // num_draft_tokens]
        if envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
            logprobs = torch.nn.functional.log_softmax(
                logits_output.next_token_logits, dim=-1
            )
        else:
            logprobs = torch.nn.functional.log_softmax(
                logits_output.next_token_logits / temperatures, dim=-1
            )
        batch_next_token_ids = res.verified_id
        accept_length_per_req_cpu = res.accept_length.tolist()
        num_tokens_per_req = [accept + 1 for accept in accept_length_per_req_cpu]

        # We should repeat top_logprobs_nums to match num_tokens_per_req.
        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        # Extract logprobs
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(
                logprobs,
                top_logprobs_nums_repeat_interleaved,
            )

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                logprobs,
                token_ids_logprobs_repeat_interleaved,
            )

        logits_output.next_token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

        # Add output logprobs to the request
        pt = 0
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        verified_ids = batch_next_token_ids.tolist()
        for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
            for _ in range(num_tokens):
                if req.return_logprob:
                    req.output_token_logprobs_val.append(next_token_logprobs[pt])
                    req.output_token_logprobs_idx.append(verified_ids[pt])
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            logits_output.next_token_top_logprobs_val[pt]
                        )
                        req.output_top_logprobs_idx.append(
                            logits_output.next_token_top_logprobs_idx[pt]
                        )
                pt += 1

    def _update_suffix_cache(self, batch: ScheduleBatch):
        """Update suffix cache with accepted tokens."""
        # Arctic Inference handles cache updates internally during add_active_response
        # No explicit batch_put needed like in NGRAMWorker
        pass

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """
        Forward pass for batch generation (same as NGRAMWorker).
        """
        self._prepare_for_speculative_decoding(batch)
        model_worker_batch = batch.get_model_worker_batch()
        num_accepted_tokens = 0

        if model_worker_batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
            verify_input = model_worker_batch.spec_info
            logits_output, next_token_ids, num_accepted_tokens = verify_input.verify(
                batch, logits_output, self.page_size
            )
            if batch.return_logprob:
                self.add_logprob_values(batch, verify_input, logits_output)
            self._update_suffix_cache(batch)
            batch.forward_mode = ForwardMode.DECODE

        else:
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
        )
