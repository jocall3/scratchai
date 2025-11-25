```python
import os
import re

class DataLoader:
    """
    Base class for data loaders.
    Handles common functionalities like batching and shuffling.
    Designed to be extended by specific data type loaders (e.g., text, numeric).
    """
    def __init__(self, batch_size: int, shuffle: bool = True):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean.")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = []  # Stores processed data (e.g., token IDs, numeric arrays)

    def _load_raw_data(self, file_path: str):
        """
        Abstract method to load raw data from a file.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _load_raw_data method.")

    def _process_data(self):
        """
        Abstract method to preprocess raw data into a format suitable for models.
        Must be implemented by subclasses to populate self.data.
        """
        raise NotImplementedError("Subclasses must implement _process_data method.")

    def _collate_fn(self, batch_data: list):
        """
        Abstract method to collate a list of individual samples into a batch.
        Must be implemented by subclasses. For 'no dependencies', this typically
        returns a list of lists or other basic Python structures.
        """
        raise NotImplementedError("Subclasses must implement _collate_fn method.")

    def __len__(self):
        """Returns the number of batches."""
        if not self.data:
            return 0
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Yields batches of data."""
        if not self.data:
            return

        indices = list(range(len(self.data)))
        if self.shuffle:
            # Custom simple shuffle if random module is strictly forbidden,
            # but usually 'random' from standard library is allowed.
            # Using random.shuffle for simplicity and efficiency.
            try:
                import random
                random.shuffle(indices)
            except ImportError:
                print("Warning: 'random' module not available for shuffling. Data will not be shuffled.")
                # Implement a manual shuffle if 'random' is truly forbidden and needed
                pass

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_samples = [self.data[idx] for idx in batch_indices]
            yield self._collate_fn(batch_samples)


class TextDataLoader(DataLoader):
    """
    Data loader for text data.
    Handles vocabulary building, tokenization, padding, and batching.
    """
    PAD_TOKEN = '<pad>'  # Used for padding sequences to max_seq_len
    UNK_TOKEN = '<unk>'  # Used for out-of-vocabulary tokens
    SOS_TOKEN = '<sos>'  # Start of Sequence token
    EOS_TOKEN = '<eos>'  # End of Sequence token

    def __init__(self, file_path: str, batch_size: int, max_seq_len: int,
                 vocab_min_freq: int = 1, shuffle: bool = True):
        super().__init__(batch_size, shuffle)
        if not isinstance(file_path, str) or not file_path:
            raise ValueError("file_path must be a non-empty string.")
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")
        if not isinstance(vocab_min_freq, int) or vocab_min_freq <= 0:
            raise ValueError("vocab_min_freq must be a positive integer.")

        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.vocab_min_freq = vocab_min_freq

        self.raw_sentences = []
        self.vocab = {}          # Maps token string to integer ID
        self.id_to_token = {}    # Maps integer ID to token string
        self.vocab_size = 0      # Total number of unique tokens in vocabulary

        self._load_raw_data(self.file_path)
        self._build_vocabulary()
        self._process_data()     # Populates self.data with token ID sequences

    def _load_raw_data(self, file_path: str):
        """Loads raw text sentences from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    self.raw_sentences.append(stripped_line)
        if not self.raw_sentences:
            raise ValueError(f"No valid data found in {file_path}")

    def _tokenize(self, text: str) -> list:
        """
        Basic word tokenizer using regex.
        Splits text into words and punctuation, keeping punctuation as separate tokens.
        Converts all tokens to lowercase.
        Example: "Hello, world!" -> ["hello", ",", "world", "!"]
        """
        # Finds sequences of word characters (\w+) or non-whitespace, non-word characters ([^\w\s])
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text.lower())
        return tokens

    def _build_vocabulary(self):
        """
        Builds a vocabulary from the raw sentences, including special tokens
        and filtering words by minimum frequency.
        """
        token_counts = {}
        for sentence in self.raw_sentences:
            tokens = self._tokenize(sentence)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

        # Add special tokens first to ensure they have fixed, low IDs
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        current_id = 0
        for token in special_tokens:
            self.vocab[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Add words based on minimum frequency, ensuring no overwrite of special tokens
        sorted_tokens_by_freq = sorted(token_counts.items(), key=lambda item: item[1], reverse=True)
        for token, count in sorted_tokens_by_freq:
            if count >= self.vocab_min_freq:
                if token not in self.vocab:
                    self.vocab[token] = current_id
                    self.id_to_token[current_id] = token
                    current_id += 1
        self.vocab_size = current_id

    def _token_to_id(self, token: str) -> int:
        """Converts a token string to its numerical ID. Uses UNK_TOKEN_ID if not found."""
        return self.vocab.get(token, self.vocab[self.UNK_TOKEN])

    def _id_to_token(self, token_id: int) -> str:
        """Converts a numerical ID back to its token string."""
        return self.id_to_token.get(token_id, self.UNK_TOKEN)

    def _pad_sequence(self, token_ids: list) -> list:
        """Pads or truncates a sequence of token IDs to the predefined max_seq_len."""
        if len(token_ids) > self.max_seq_len:
            return token_ids[:self.max_seq_len]
        else:
            padding_needed = self.max_seq_len - len(token_ids)
            return token_ids + [self.vocab[self.PAD_TOKEN]] * padding_needed

    def _process_data(self):
        """Tokenizes all raw sentences, adds SOS/EOS tokens, and pads them."""
        processed_samples = []
        for sentence in self.raw_sentences:
            tokens = self._tokenize(sentence)
            # Prepend SOS and append EOS tokens
            token_ids = ([self.vocab[self.SOS_TOKEN]] +
                         [self._token_to_id(token) for token in tokens] +
                         [self.vocab[self.EOS_TOKEN]])

            padded_ids = self._pad_sequence(token_ids)
            processed_samples.append(padded_ids)
        self.data = processed_samples

    def _collate_fn(self, batch_data: list) -> list:
        """
        Collates a list of padded token ID sequences into a single batch structure.
        For "no dependencies", this simply returns the list of lists.
        """
        return batch_data

    def get_vocab(self) -> dict:
        """Returns the token-to-ID vocabulary mapping."""
        return self.vocab

    def get_id_to_token_map(self) -> dict:
        """Returns the ID-to-token mapping."""
        return self.id_to_token

    def get_vocab_size(self) -> int:
        """Returns the total size of the vocabulary."""
        return self.vocab_size


class NumericDataLoader(DataLoader):
    """
    Data loader for simple numeric data.
    Assumes each line in the input file represents a single sample,
    with values separated by a specified delimiter (e.g., CSV-like).
    """
    def __init__(self, file_path: str, batch_size: int, separator: str = ',', shuffle: bool = True):
        super().__init__(batch_size, shuffle)
        if not isinstance(file_path, str) or not file_path:
            raise ValueError("file_path must be a non-empty string.")
        if not isinstance(separator, str) or not separator:
            raise ValueError("separator must be a non-empty string.")

        self.file_path = file_path
        self.separator = separator
        self.raw_data_lines = []
        self.feature_count = 0  # To store the expected number of features per sample

        self._load_raw_data(self.file_path)
        self._process_data()    # Populates self.data with lists of floats

    def _load_raw_data(self, file_path: str):
        """Loads raw numeric data lines from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    self.raw_data_lines.append(stripped_line)

        if not self.raw_data_lines:
            raise ValueError(f"No valid data found in {file_path}")

        # Determine feature count from the first line for consistency check
        first_sample_str_parts = self.raw_data_lines[0].split(self.separator)
        self.feature_count = len(first_sample_str_parts)

    def _process_data(self):
        """
        Converts raw data lines into lists of floats.
        Performs basic validation on feature count per line.
        """
        processed_samples = []
        for line in self.raw_data_lines:
            try:
                numeric_values = [float(val.strip()) for val in line.split(self.separator)]
                if len(numeric_values) != self.feature_count:
                    raise ValueError(
                        f"Inconsistent feature count in line: '{line}'. "
                        f"Expected {self.feature_count}, got {len(numeric_values)}."
                    )
                processed_samples.append(numeric_values)
            except ValueError as e:
                print(f"Warning: Could not parse line '{line}' as numeric data. Skipping. Error: {e}")
                continue
        self.data = processed_samples

    def _collate_fn(self, batch_data: list) -> list:
        """
        Collates a list of numeric samples into a batch.
        For "no dependencies", this remains a list of lists.
        """
        return batch_data

    def get_feature_count(self) -> int:
        """Returns the number of features expected per sample."""
        return self.feature_count
```