""" 
Cache adapter that wraps Arctic Inference SuffixDecodingCache
to provide the same interface as NgramCache.

This allows NGRAMWorker to use suffix decoding without modification.
"""

import logging
import os
from collections import deque
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

            # Ensure cache includes the latest verified tokens before speculation.
            current_length = len(tokens)
            if current_length > last_length:
                new_tokens = tokens[last_length:current_length]
                logger.info(
                    f"[BATCH_GET {idx}] Adding {len(new_tokens)} new tokens before speculate: {new_tokens}"
                )
                if arctic_req_id in self.suffix_cache.active_requests:
                    self.suffix_cache.add_active_response(arctic_req_id, new_tokens)
                    self.req_state[sglang_req_id][1] = current_length
                    last_length = current_length
                else:
                    logger.warning(
                        f"[BATCH_GET {idx}] Arctic req {arctic_req_id} not active when updating!"
                    )

            # Extract pattern from end of tokens (up to max_tree_depth)
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
            draft_ids = list(draft.token_ids)
            draft_parents = list(draft.parents)
            draft_ids, draft_parents = self._reorder_tree_bfs(draft_ids, draft_parents)

            context_token = tokens[-1] if tokens else 0
            draft_ids, draft_parents = self._inject_root_node(draft_ids, draft_parents, context_token)

            logger.info(
                f"[BATCH_GET {idx}] Arctic returned {max(0, len(draft_ids) - 1)} drafts (excl. root)"
            )

            # Pad or truncate to match draft_token_num (includes root node at index 0)
            original_draft_len = len(draft_ids)
            if original_draft_len == 1:
                logger.info(f"[BATCH_GET {idx}] No suffix drafts available for this step (root only)")
            if original_draft_len < self.draft_token_num:
                pad_len = self.draft_token_num - original_draft_len
                draft_ids.extend([0] * pad_len)
                draft_parents.extend([0] * pad_len)
                logger.info(f"[BATCH_GET {idx}] Padded with {pad_len} zero tokens attached to root")
            elif original_draft_len > self.draft_token_num:
                draft_ids = draft_ids[: self.draft_token_num]
                draft_parents = draft_parents[: self.draft_token_num]
                logger.info(
                    f"[BATCH_GET {idx}] Truncated tree (with root) from {original_draft_len} to {self.draft_token_num}"
                )
                original_draft_len = self.draft_token_num

            all_drafts.extend(draft_ids)

            # Build tree mask from parent structure
            # Token i can attend to token j if j is an ancestor of i
            mask = np.zeros((self.draft_token_num, self.draft_token_num), dtype=bool)
            if original_draft_len > 0:
                for i in range(original_draft_len):
                    mask[i, i] = True  # Self-attention
                    parent_idx = draft_parents[i]
                    while parent_idx >= 0 and parent_idx < self.draft_token_num:
                        mask[i, parent_idx] = True
                        parent_idx = draft_parents[parent_idx]

            all_masks.append(mask.flatten())

            if self.debug_tree_dump_remaining > 0 and original_draft_len > 0:
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
        No-op: cache updates now happen inside batch_get before speculation.
        Kept for interface compatibility with NGRAMWorker.
        """
        _ = batch_tokens  # Intentional placeholder to satisfy interface.
        for idx, sglang_req_id in enumerate(batch_req_ids):
            if sglang_req_id not in self.req_state:
                logger.error(
                    f"[BATCH_PUT {idx}] Called for unknown request {sglang_req_id}! "
                    f"This should not happen - batch_get must be called first."
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

    def _reorder_tree_bfs(self, token_ids: List[int], parents: List[int]) -> Tuple[List[int], List[int]]:
        """
        Reorder the speculative tree so that every parent appears before its children.

        NGRAMWorker expects draft nodes to be topologically sorted because
        reconstruct_indices_from_tree_mask only scans columns < row when
        computing ancestors. Arctic returns nodes in score order, so we enforce
        a BFS order rooted at the original parents to satisfy that constraint.
        """
        n = len(token_ids)
        if n <= 1:
            return token_ids, parents

        children: List[List[int]] = [[] for _ in range(n)]
        roots: List[int] = []
        for idx, parent in enumerate(parents):
            if parent is None or parent < 0 or parent >= n:
                roots.append(idx)
            else:
                children[parent].append(idx)

        if not roots:
            roots = [0]

        order: List[int] = []
        visited = [False] * n
        for root in roots:
            if visited[root]:
                continue
            queue = deque([root])
            while queue:
                node = queue.popleft()
                if visited[node]:
                    continue
                visited[node] = True
                order.append(node)
                for child in children[node]:
                    if not visited[child]:
                        queue.append(child)

        # Append any detached nodes (should not happen, but keep deterministic order).
        for idx in range(n):
            if not visited[idx]:
                order.append(idx)

        if order == list(range(n)):
            return token_ids, parents

        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
        reordered_ids = [token_ids[old_idx] for old_idx in order]
        reordered_parents: List[int] = []
        for old_idx in order:
            parent = parents[old_idx]
            if parent is None or parent < 0:
                reordered_parents.append(-1)
            else:
                reordered_parents.append(remap.get(parent, -1))

        logger.debug(
            "[SUFFIX ADAPTER] Reordered draft tree (size=%d) to enforce parent-before-child ordering",
            n,
        )
        return reordered_ids, reordered_parents

    def _inject_root_node(
        self, token_ids: List[int], parents: List[int], context_token: int
    ) -> Tuple[List[int], List[int]]:
        """
        Insert a synthetic root node (the latest verified token) to mimic NGRAM cache layout.

        NGRAMWorker assumes index 0 always points to the already-verified token that new drafts
        branch from. We prepend that context token and shift all parent indices accordingly.
        """
        if context_token is None:
            context_token = 0

        rooted_ids = [context_token]
        rooted_parents = [-1]
        for parent_idx in parents:
            if parent_idx is None or parent_idx < 0:
                rooted_parents.append(0)
            else:
                rooted_parents.append(parent_idx + 1)
        rooted_ids.extend(token_ids)
        return rooted_ids, rooted_parents
