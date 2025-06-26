import hashlib
from typing import List

class ContentDeduplicator:
    def __init__(self, block_size: int = 300, hash_window: int = 2):
        """
        Args:
            block_size: Minimum block length to consider for deduplication.
            hash_window: How many previous hashes to compare for similarity.
        """
        self.block_size = block_size
        self.hash_window = hash_window
        self._seen_hashes = []

    def _hash_block(self, text: str) -> str:
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    def deduplicate(self, text: str) -> str:
        """
        Removes blocks of text seen in previous pages.
        
        Args:
            text: Full text content of a page.
        
        Returns:
            Deduplicated text content.
        """
        blocks = text.split("\n\n")  # Paragraph blocks
        deduped_blocks = []
        
        for block in blocks:
            block = block.strip()
            if len(block) < self.block_size:
                deduped_blocks.append(block)
                continue

            block_hash = self._hash_block(block)
            if block_hash not in self._seen_hashes:
                deduped_blocks.append(block)
                self._seen_hashes.append(block_hash)

                # Maintain window size
                if len(self._seen_hashes) > self.hash_window * 1000:
                    self._seen_hashes = self._seen_hashes[-self.hash_window * 1000:]
        
        return "\n\n".join(deduped_blocks)
