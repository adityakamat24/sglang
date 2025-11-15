""" 
Cache adapter that wraps Arctic Inference SuffixDecodingCache
to provide the same interface as NgramCache.

This allows NGRAMWorker to use suffix decoding without modification.
"""

import logging
import os
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SuffixCacheAdapter:
    """
    Adapter that wraps SuffixDecodingCache to match NgramCache interface.

    NGRAMWorker expects:
    - batch_get(batch_tokens: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]
      Returns (draft_tokens, tree_mask) as flat numpy arrays
    - batch_put(batch_tokens: List[List[int]]) -> None
      Updates cache with verified tokens
    - synchronize() -> None
      No-op for suffix cache
    - reset() -> None
      Clears all cached data
    """

    def __init__(
        self,
        draft_token_num: int,
        max_tree_depth: int = 24,
        max_cached_requests: int = 10000,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ):
        """
        Args:
            draft_token_num: Fixed number of draft tokens (for padding)
            max_tree_depth: Maximum depth for suffix tree
            max_cached_requests: Maximum number of cached requests
            max_spec_factor: Maximum speculation factor
            min_token_prob: Minimum token probability threshold
        """
        # Lazy import to avoid error when Suffix Decoding is not used
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
        )
        self.draft_token_num = draft_token_num
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob

        # Debug toggles (set env e.g. SUFFIX_DEBUG_TREE=1 to dump first batch)
        self.debug_tree_dump_remaining = int(os.environ.get("SUFFIX_DEBUG_TREE", "0"))

        # Track state by SGlang request ID (stable identifier)
        # Map: sglang_req_id → (arctic_req_id, last_length)
        self.req_state = {}

    def _cleanup_inactive_requests(self, active_req_ids: set[str]):
        """Stop Arctic requests that are no longer active in SGlang."""
        inactive_req_ids = [rid for rid in self.req_state.keys() if rid not in active_req_ids]
        for rid in inactive_req_ids:
            arctic_req_id, _ = self.req_state.pop(rid)
            if arctic_req_id in getattr(self.suffix_cache, "active_requests", set()):
                logger.info(
                    f"[CLEANUP] Stopping Arctic request {arctic_req_id} for inactive SGlang req {rid}"
                )
                self.suffix_cache.stop_request(arctic_req_id)

    def _get_or_create_arctic_req_id(self, sglang_req_id: str, prompt: List[int], tokens: List[int]) -> tuple:
        """Get or create an Arctic request ID for the given SGlang request.

        Args:
            sglang_req_id: Stable request ID from SGlang
            prompt: Prompt tokens only (no generated tokens)
            tokens: Full token sequence (prompt + outputs)

        Returns: (arctic_req_id, last_length)
        """
        if sglang_req_id not in self.req_state:
            # Use SGlang request ID directly as Arctic request ID
            arctic_req_id = sglang_req_id

            logger.info(
                f"[NEW_REQUEST] Creating Arctic request for SGlang req {sglang_req_id}, "
                f"prompt_len={len(prompt)}, total_len={len(tokens)}, "
                f"first_10_prompt={prompt[:10]}, last_10_tokens={tokens[-10:]}"
            )

            # Initialize the request in suffix cache with ONLY the prompt
            self.suffix_cache.start_request(arctic_req_id, prompt)

            # Track: [arctic_req_id, last_length]
            # IMPORTANT: Set last_length to prompt length since Arctic already has the prompt
            self.req_state[sglang_req_id] = [arctic_req_id, len(prompt)]

            logger.info(
                f"[NEW_REQUEST] Arctic request {arctic_req_id} initialized, "
                f"active_requests={self.suffix_cache.active_requests}"
            )

        arctic_req_id, last_length = self.req_state[sglang_req_id]
        return arctic_req_id, last_length

    def batch_get(
        self, batch_req_ids: List[str], batch_prompts: List[List[int]], batch_tokens: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get draft tokens for a batch of token sequences.

        This is called BEFORE verification with the current state.
        We speculate based on the current tokens.

        Args:
            batch_req_ids: List of SGlang request IDs (stable)
            batch_prompts: List of prompt tokens (no generated tokens)
            batch_tokens: List of token sequences (prompt + output tokens)

        Returns:
            Tuple of:
            - draft_tokens: np.ndarray of shape (batch_size * draft_token_num,)
            - tree_mask: np.ndarray of shape (batch_size * draft_token_num * draft_token_num,)
        """
        all_drafts = []
        all_masks = []

        active_req_ids = set(batch_req_ids)
        self._cleanup_inactive_requests(active_req_ids)

        for idx, (sglang_req_id, prompt, tokens) in enumerate(zip(batch_req_ids, batch_prompts, batch_tokens)):
            arctic_req_id, last_length = self._get_or_create_arctic_req_id(sglang_req_id, prompt, tokens)

            # Extract pattern from end of tokens (up to max_tree_depth)
            # Note: Cache should already be updated from previous batch_put() call
            pattern_start = max(0, len(tokens) - self.max_tree_depth)
            pattern = tokens[pattern_start:]

            logger.info(
                f"[BATCH_GET {idx}] sglang_req={sglang_req_id}, arctic_req={arctic_req_id}, "
                f"total_len={len(tokens)}, last_length={last_length}, pattern_len={len(pattern)}, "
                f"pattern_start={pattern_start}, last_10_tokens={tokens[-10:]}"
            )

            # Speculate using suffix cache
            # Cache should have been updated in previous iteration's batch_put()
            draft = self.suffix_cache.speculate(
                arctic_req_id,
                pattern,
                max_spec_tokens=self.draft_token_num,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            # Convert to fixed-size arrays
            draft_ids = draft.token_ids
            draft_parents = draft.parents

            logger.info(
                f"[BATCH_GET {idx}] Arctic returned {len(draft_ids)} drafts: {draft_ids[:8]}"
            )

            # Pad or truncate to match draft_token_num
            original_draft_len = len(draft_ids)
            if original_draft_len == 0:
                # No speculation this round: return zeros with empty masks so the worker skips verify.
                draft_ids = [0] * self.draft_token_num
                draft_parents = [-1] * self.draft_token_num
                logger.info("[BATCH_GET %d] No drafts from Arctic; returning zeroed tensors", idx)
            elif len(draft_ids) < self.draft_token_num:
                pad_len = self.draft_token_num - len(draft_ids)
                last_token = tokens[-1] if tokens else 0
                draft_ids = draft_ids + [last_token] * pad_len
                draft_parents = draft_parents + [-1] * pad_len
                logger.info(f"[BATCH_GET {idx}] Padded with {pad_len} copies of token {last_token}")
            elif len(draft_ids) > self.draft_token_num:
                draft_ids = draft_ids[: self.draft_token_num]
                draft_parents = draft_parents[: self.draft_token_num]
                logger.info(f"[BATCH_GET {idx}] Truncated from {original_draft_len} to {self.draft_token_num}")

            all_drafts.extend(draft_ids)

            # Build tree mask from parent structure
            # Token i can attend to token j if j is an ancestor of i
            mask = np.zeros((self.draft_token_num, self.draft_token_num), dtype=bool)
            if original_draft_len > 0:
                for i in range(self.draft_token_num):
                    mask[i, i] = True  # Self-attention
                    parent_idx = draft_parents[i]
                    while parent_idx >= 0 and parent_idx < self.draft_token_num:
                        mask[i, parent_idx] = True
                        parent_idx = draft_parents[parent_idx]

            all_masks.append(mask.flatten())

            if self.debug_tree_dump_remaining > 0:
                logger.warning(
                    "[SUFFIX DEBUG] req=%s, original_draft_len=%d, masked_len=%d, draft_ids=%s",
                    sglang_req_id,
                    original_draft_len,
                    len(draft_ids),
                    draft_ids,
                )
                logger.warning(
                    "[SUFFIX DEBUG] mask=\n%s",
                    mask.astype(int),
                )
                self.debug_tree_dump_remaining -= 1

        # Convert to numpy arrays (must be int64 for NGRAMWorker)
        req_drafts = np.array(all_drafts, dtype=np.int64)
        tree_mask = np.concatenate(all_masks)

        logger.info(f"[BATCH_GET] Returning {len(req_drafts)} total draft tokens")
        return req_drafts, tree_mask

    def batch_put(self, batch_req_ids: List[str], batch_tokens: List[List[int]]):
        """
        Update cache with verified tokens (matching ngram pattern).

        This is called AFTER verification. We update the cache with newly verified tokens
        so they're available for the NEXT iteration's batch_get() call.

        Args:
            batch_req_ids: List of SGlang request IDs (stable)
            batch_tokens: List of token sequences after verification
        """
        for idx, (sglang_req_id, tokens) in enumerate(zip(batch_req_ids, batch_tokens)):
            if sglang_req_id not in self.req_state:
                # This shouldn't happen (batch_get should have been called first)
                logger.error(
                    f"[BATCH_PUT {idx}] Called for unknown request {sglang_req_id}! "
                    f"This should not happen - batch_get must be called first."
                )
                continue

            arctic_req_id, last_length = self.req_state[sglang_req_id]
            current_length = len(tokens)

            # Add new tokens to cache (for next iteration's batch_get)
            if current_length > last_length:
                new_tokens = tokens[last_length:current_length]
                logger.info(
                    f"[BATCH_PUT {idx}] Adding {len(new_tokens)} new tokens to cache: {new_tokens}"
                )
                if arctic_req_id in self.suffix_cache.active_requests:
                    self.suffix_cache.add_active_response(arctic_req_id, new_tokens)
                    # Update tracked length
                    self.req_state[sglang_req_id][1] = current_length
                else:
                    logger.warning(
                        f"[BATCH_PUT {idx}] Arctic req {arctic_req_id} not in active_requests! "
                        f"Active: {self.suffix_cache.active_requests}"
                    )
            elif current_length == last_length:
                logger.debug(
                    f"[BATCH_PUT {idx}] No new tokens (length unchanged: {current_length})"
                )
            else:
                logger.warning(
                    f"[BATCH_PUT {idx}] Length decreased! current={current_length}, last={last_length}"
                )

    def synchronize(self):
        """No-op for suffix cache (no async operations)."""
        pass

    def reset(self):
        """Clear all cached data."""
        # Stop all active requests
        for arctic_req_id in list(self.suffix_cache.active_requests):
            self.suffix_cache.stop_request(arctic_req_id)
        # Clear tracking
        self.req_state.clear()
        logger.info("[SUFFIX ADAPTER] Cache reset")
