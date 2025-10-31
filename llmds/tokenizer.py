"""Simple BPE-style tokenizer interface."""

from typing import Optional


class Tokenizer:
    """
    Simple tokenizer interface with BPE-style stub implementation.

    Provides a pluggable interface for tokenization that can be
    extended with real tokenizers (e.g., tiktoken, transformers).
    """

    def __init__(self, vocab_size: int = 50257):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Vocabulary size (default GPT-2 like)
        """
        self.vocab_size = vocab_size
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}
        self._build_simple_vocab()

    def _build_simple_vocab(self) -> None:
        """Build a simple vocabulary for testing."""
        # Simple vocabulary: common words + special tokens
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        common_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
        ]

        all_tokens = special_tokens + common_words
        for i, token in enumerate(all_tokens[: self.vocab_size]):
            self._word_to_id[token] = i
            self._id_to_word[i] = token

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        # Simple whitespace-based tokenization
        words = text.lower().split()
        token_ids = []
        unk_id = self._word_to_id.get("<unk>", 0)

        for word in words:
            # Simple BPE-like: try full word, then fallback to char-level
            if word in self._word_to_id:
                token_ids.append(self._word_to_id[word])
            else:
                # Character-level fallback
                for char in word:
                    char_token = f"<char_{char}>"
                    if char_token in self._word_to_id:
                        token_ids.append(self._word_to_id[char_token])
                    else:
                        token_ids.append(unk_id)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        words = []
        for token_id in token_ids:
            if token_id in self._id_to_word:
                word = self._id_to_word[token_id]
                if not word.startswith("<"):
                    words.append(word)
        return " ".join(words)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        return len(self.encode(text))

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

