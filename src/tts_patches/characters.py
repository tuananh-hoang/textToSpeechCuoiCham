from dataclasses import replace
from typing import Dict

from TTS.tts.configs.shared_configs import CharactersConfig


def parse_symbols():
    return {
        "pad": _pad,
        "eos": _eos,
        "bos": _bos,
        "characters": _characters,
        "punctuations": _punctuations,
        "phonemes": _phonemes,
    }


# DEFAULT SET OF GRAPHEMES
_pad = "<PAD>"
_eos = "<EOS>"
_bos = "<BOS>"
_blank = "<BLNK>"  # TODO: check if we need this alongside with PAD
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_punctuations = "!'(),-.:;? "


# DEFAULT SET OF IPA PHONEMES
# Phonemes definition (All IPA characters)
_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "ˈˌːˑ"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧʲ"
_diacrilics = "ɚ˞ɫ"
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics


class BaseVocabulary:
    """Base Vocabulary class.

    This class only needs a vocabulary dictionary without specifying the characters.

    Args:
        vocab (Dict): A dictionary of characters and their corresponding indices.
    """

    def __init__(self, vocab: Dict, pad: str = None, blank: str = None, bos: str = None, eos: str = None):
        self.vocab = vocab
        self.pad = pad
        self.blank = blank
        self.bos = bos
        self.eos = eos

    @property
    def pad_id(self) -> int:
        """Return the index of the padding character. If the padding character is not specified, return the length
        of the vocabulary."""
        return self.char_to_id(self.pad) if self.pad else len(self.vocab)

    @property
    def blank_id(self) -> int:
        """Return the index of the blank character. If the blank character is not specified, return the length of
        the vocabulary."""
        return self.char_to_id(self.blank) if self.blank else len(self.vocab)

    @property
    def bos_id(self) -> int:
        """Return the index of the bos character. If the bos character is not specified, return the length of the
        vocabulary."""
        return self.char_to_id(self.bos) if self.bos else len(self.vocab)

    @property
    def eos_id(self) -> int:
        """Return the index of the eos character. If the eos character is not specified, return the length of the
        vocabulary."""
        return self.char_to_id(self.eos) if self.eos else len(self.vocab)

    @property
    def vocab(self):
        """Return the vocabulary dictionary."""
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        """Set the vocabulary dictionary and character mapping dictionaries."""
        self._vocab, self._char_to_id, self._id_to_char = None, None, None
        if vocab is not None:
            self._vocab = vocab
            self._char_to_id = {char: idx for idx, char in enumerate(self._vocab)}
            self._id_to_char = {
                idx: char for idx, char in enumerate(self._vocab)  # pylint: disable=unnecessary-comprehension
            }

    @staticmethod
    def init_from_config(config, **kwargs):
        """Initialize from the given config."""
        if config.characters is not None and "vocab_dict" in config.characters and config.characters.vocab_dict:
            return (
                BaseVocabulary(
                    config.characters.vocab_dict,
                    config.characters.pad,
                    config.characters.blank,
                    config.characters.bos,
                    config.characters.eos,
                ),
                config,
            )
        return BaseVocabulary(**kwargs), config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            vocab_dict=self._vocab,
            pad=self.pad,
            eos=self.eos,
            bos=self.bos,
            blank=self.blank,
            is_unique=False,
            is_sorted=False,
        )

    @property
    def num_chars(self):
        """Return number of tokens in the vocabulary."""
        return len(self._vocab)

    def char_to_id(self, char: str) -> int:
        """Map a character to an token ID."""
        try:
            return self._char_to_id[char]
        except KeyError as e:
            raise KeyError(f" [!] {repr(char)} is not in the vocabulary.") from e

    def id_to_char(self, idx: int) -> str:
        """Map an token ID to a character."""
        return self._id_to_char[idx]


class BaseCharacters:
    """🐸BaseCharacters class

        Every new character class should inherit from this.

        Characters are oredered as follows ```[PAD, EOS, BOS, BLANK, CHARACTERS, PUNCTUATIONS]```.

        If you need a custom order, you need to define inherit from this class and override the ```_create_vocab``` method.

        Args:
            characters (str):
                Main set of characters to be used in the vocabulary.

            punctuations (str):
                Characters to be treated as punctuation.

            pad (str):
                Special padding character that would be ignored by the model.

            eos (str):
                End of the sentence character.

            bos (str):
                Beginning of the sentence character.

            blank (str):
                Optional character used between characters by some models for better prosody.

            is_unique (bool):
                Remove duplicates from the provided characters. Defaults to True.
    el
            is_sorted (bool):
                Sort the characters in alphabetical order. Only applies to `self.characters`. Defaults to True.
    """

    def __init__(
        self,
        characters: str = None,
        punctuations: str = None,
        pad: str = None,
        eos: str = None,
        bos: str = None,
        blank: str = None,
        is_unique: bool = False,
        is_sorted: bool = True,
    ) -> None:
        self._characters = characters
        self._punctuations = punctuations
        self._pad = pad
        self._eos = eos
        self._bos = bos
        self._blank = blank
        self.is_unique = is_unique
        self.is_sorted = is_sorted
        self._create_vocab()

    @property
    def pad_id(self) -> int:
        return self.char_to_id(self.pad) if self.pad else len(self.vocab)

    @property
    def blank_id(self) -> int:
        return self.char_to_id(self.blank) if self.blank else len(self.vocab)

    @property
    def eos_id(self) -> int:
        return self.char_to_id(self.eos) if self.eos else len(self.vocab)

    @property
    def bos_id(self) -> int:
        return self.char_to_id(self.bos) if self.bos else len(self.vocab)

    @property
    def characters(self):
        return self._characters

    @characters.setter
    def characters(self, characters):
        self._characters = characters
        self._create_vocab()

    @property
    def punctuations(self):
        return self._punctuations

    @punctuations.setter
    def punctuations(self, punctuations):
        self._punctuations = punctuations
        self._create_vocab()

    @property
    def pad(self):
        return self._pad

    @pad.setter
    def pad(self, pad):
        self._pad = pad
        self._create_vocab()

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, eos):
        self._eos = eos
        self._create_vocab()

    @property
    def bos(self):
        return self._bos

    @bos.setter
    def bos(self, bos):
        self._bos = bos
        self._create_vocab()

    @property
    def blank(self):
        return self._blank

    @blank.setter
    def blank(self, blank):
        self._blank = blank
        self._create_vocab()

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        self._vocab = vocab
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self._id_to_char = {
            idx: char for idx, char in enumerate(self.vocab)  # pylint: disable=unnecessary-comprehension
        }

    @property
    def num_chars(self):
        return len(self._vocab)

    def _create_vocab(self):
        _vocab = self._characters
        # Nếu là chuỗi thì chuyển thành list từng ký tự, nếu là list thì giữ nguyên
        if isinstance(_vocab, str):
            _vocab = list(_vocab)
        elif isinstance(_vocab, list):
            pass
        else:
            raise TypeError("characters must be str or list")
        if self.is_unique:
            _vocab = list(set(_vocab))
        if self.is_sorted:
            _vocab = sorted(_vocab)
        _vocab = [self._blank] + _vocab if self._blank is not None and len(self._blank) > 0 else _vocab
        _vocab = [self._bos] + _vocab if self._bos is not None and len(self._bos) > 0 else _vocab
        _vocab = [self._eos] + _vocab if self._eos is not None and len(self._eos) > 0 else _vocab
        _vocab = [self._pad] + _vocab if self._pad is not None and len(self._pad) > 0 else _vocab
        # Nếu là chuỗi, list(self._punctuations) sẽ tách từng ký tự, nếu là list thì nối luôn
        if isinstance(self._punctuations, str):
            self.vocab = _vocab + list(self._punctuations)
        elif isinstance(self._punctuations, list):
            self.vocab = _vocab + self._punctuations
        else:
            self.vocab = _vocab
        if self.is_unique:
            duplicates = {x for x in self.vocab if self.vocab.count(x) > 1}
            assert (
                len(self.vocab) == len(self._char_to_id) == len(self._id_to_char)
            ), f" [!] There are duplicate characters in the character set. {duplicates}"
    def char_to_id(self, char: str) -> int:
        try:
            return self._char_to_id[char]
        except KeyError as e:
            raise KeyError(f" [!] {repr(char)} is not in the vocabulary.") from e

    def id_to_char(self, idx: int) -> str:
        return self._id_to_char[idx]

    def print_log(self, level: int = 0):
        """
        Prints the vocabulary in a nice format.
        """
        indent = "\t" * level
        print(f"{indent}| > Characters: {self._characters}")
        print(f"{indent}| > Punctuations: {self._punctuations}")
        print(f"{indent}| > Pad: {self._pad}")
        print(f"{indent}| > EOS: {self._eos}")
        print(f"{indent}| > BOS: {self._bos}")
        print(f"{indent}| > Blank: {self._blank}")
        print(f"{indent}| > Vocab: {self.vocab}")
        print(f"{indent}| > Num chars: {self.num_chars}")

    @staticmethod
    def init_from_config(config: "Coqpit"):  # pylint: disable=unused-argument
        """Init your character class from a config.

        Implement this method for your subclass.
        """
        # use character set from config
        if config.characters is not None:
            return BaseCharacters(**config.characters), config
        # return default character set
        characters = BaseCharacters()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            characters=self._characters,
            punctuations=self._punctuations,
            pad=self._pad,
            eos=self._eos,
            bos=self._bos,
            blank=self._blank,
            is_unique=self.is_unique,
            is_sorted=self.is_sorted,
        )


class IPAPhonemes(BaseCharacters):
    """🐸IPAPhonemes class to manage `TTS.tts` model vocabulary

    Intended to be used with models using IPAPhonemes as input.
    It uses system defaults for the undefined class arguments.

    Args:
        characters (str):
            Main set of case-sensitive characters to be used in the vocabulary. Defaults to `_phonemes`.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to `_punctuations`.

        pad (str):
            Special padding character that would be ignored by the model. Defaults to `_pad`.

        eos (str):
            End of the sentence character. Defaults to `_eos`.

        bos (str):
            Beginning of the sentence character. Defaults to `_bos`.

        blank (str):
            Optional character used between characters by some models for better prosody. Defaults to `_blank`.

        is_unique (bool):
            Remove duplicates from the provided characters. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Defaults to True.
    """

    def __init__(
        self,
        characters: str = _phonemes,
        punctuations: str = _punctuations,
        pad: str = _pad,
        eos: str = _eos,
        bos: str = _bos,
        blank: str = _blank,
        is_unique: bool = False,
        is_sorted: bool = True,
    ) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted)

    @staticmethod
    def init_from_config(config: "Coqpit"):
        """Init a IPAPhonemes object from a model config

        If characters are not defined in the config, it will be set to the default characters and the config
        will be updated.
        """
        # band-aid for compatibility with old models
        if "characters" in config and config.characters is not None:
            if "phonemes" in config.characters and config.characters.phonemes is not None:
                config.characters["characters"] = config.characters["phonemes"]
            return (
                IPAPhonemes(
                    characters=config.characters["characters"],
                    punctuations=config.characters["punctuations"],
                    pad=config.characters["pad"],
                    eos=config.characters["eos"],
                    bos=config.characters["bos"],
                    blank=config.characters["blank"],
                    is_unique=config.characters["is_unique"],
                    is_sorted=config.characters["is_sorted"],
                ),
                config,
            )
        # use character set from config
        if config.characters is not None:
            return IPAPhonemes(**config.characters), config
        # return default character set
        characters = IPAPhonemes()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config


class Graphemes(BaseCharacters):
    """🐸Graphemes class to manage `TTS.tts` model vocabulary

    Intended to be used with models using graphemes as input.
    It uses system defaults for the undefined class arguments.

    Args:
        characters (str):
            Main set of case-sensitive characters to be used in the vocabulary. Defaults to `_characters`.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to `_punctuations`.

        pad (str):
            Special padding character that would be ignored by the model. Defaults to `_pad`.

        eos (str):
            End of the sentence character. Defaults to `_eos`.

        bos (str):
            Beginning of the sentence character. Defaults to `_bos`.

        is_unique (bool):
            Remove duplicates from the provided characters. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Defaults to True.
    """

    def __init__(
        self,
        characters: str = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        eos: str = _eos,
        bos: str = _bos,
        blank: str = _blank,
        is_unique: bool = False,
        is_sorted: bool = True,
    ) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted)

    @staticmethod
    def init_from_config(config: "Coqpit"):
        """Init a Graphemes object from a model config

        If characters are not defined in the config, it will be set to the default characters and the config
        will be updated.
        """
        if config.characters is not None:
            # band-aid for compatibility with old models
            if "phonemes" in config.characters:
                return (
                    Graphemes(
                        characters=config.characters["characters"],
                        punctuations=config.characters["punctuations"],
                        pad=config.characters["pad"],
                        eos=config.characters["eos"],
                        bos=config.characters["bos"],
                        blank=config.characters["blank"],
                        is_unique=config.characters["is_unique"],
                        is_sorted=config.characters["is_sorted"],
                    ),
                    config,
                )
            return Graphemes(**config.characters), config
        characters = Graphemes()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config


if __name__ == "__main__":
    gr = Graphemes()
    ph = IPAPhonemes()
    gr.print_log()
    ph.print_log()


import unicodedata
from dataclasses import replace
from TTS.tts.utils.text.characters import BaseCharacters
from TTS.tts.configs.shared_configs import CharactersConfig
import unicodedata
from dataclasses import replace

class CuoiChamPhonemes(BaseCharacters):

    def __init__(self, pad="<PAD>", eos="<EOS>", bos="<BOS>", blank="<BLNK>", **kwargs):
        
        # initial_clusters = ['pʰr', 'kʰr', 'bl', 'pl', 'kl']
        
        # long_vowels = ['iː', 'eː', 'ɛː', 'aː', 'ɔː', 'oː', 'uː', 'əː', 'ɒː', 'ɐː', 'ɨː']

        # diphthongs = ['ia', 'ua', 'ɨə']
        
        # single_consonants = [
        #     'p', 'b', 't', 'd', 's', 'c', 'j', 'k', 'kʰ', 'tʰ', 'm', 'n', 'ŋ', 
        #     'l', 'r', 'w', 'h', 'ɲ', 'ʈ', 'ɣ', 'ʔ', 'v', 'z', 'f'
        # ]
          
        # single_vowels = ['i', 'ɨ', 'u', 'e', 'ə', 'o', 'ɛ', 'ɔ', 'a', 'ɐ', 'ɒ', 'ʌ']
        
        # tones_and_symbols = ['¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', 'ʰ', 'ˢ', '̆']
        initial_clusters = ['pʰr', 'kʰr', 'bl', 'pl', 'kl']
        long_vowels = ['iː', 'eː', 'ɛː', 'aː', 'ɔː', 'oː', 'uː', 'əː', 'ɒː', 'ɐː', 'ɨː']
        diphthongs  = ['ia', 'ua', 'ɨə']
        single_consonants = [
            'p', 'b', 't', 'd', 's', 'c', 'j', 'k', 'm', 'n', 'ŋ',
            'l', 'r', 'w', 'h', 'ɲ', 'ʈ', 'ɣ', 'ʔ', 'v', 'z','f'
        ]
        single_vowels = ['i', 'ɨ', 'u', 'e', 'ə', 'o', 'ɛ', 'ɔ', 'a', 'ɐ', 'ɒ', 'ʌ']
        tones_and_symbols = ['¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', 'ʰ', 'ˢ','̆']
        # Unicode normalization cho consistency
        def normalize_phoneme(phoneme):
            return unicodedata.normalize('NFC', phoneme)
        
        # Normalize all phonemes theo từng category
        all_phonemes = []
        for phoneme_list in [initial_clusters, long_vowels, diphthongs, single_consonants, single_vowels, tones_and_symbols]:
            all_phonemes.extend([normalize_phoneme(p) for p in phoneme_list])
        
        # Remove duplicates while preserving order (giống code cũ)
        seen = set()
        unique_phonemes = []
        for phoneme in all_phonemes:
            if phoneme not in seen:
                seen.add(phoneme)
                unique_phonemes.append(phoneme)
        
        # ========================= CRITICAL: SORT BY LENGTH =========================
        # Sắp xếp từ dài đến ngắn cho greedy matching (giống code cũ)
        self.PHONEME_INVENTORY = sorted(unique_phonemes, key=len, reverse=True)
        
        # Normalize special tokens
        normalized_pad = normalize_phoneme(pad)
        normalized_eos = normalize_phoneme(eos)
        normalized_bos = normalize_phoneme(bos)
        normalized_blank = normalize_phoneme(blank)
        
        # Initialize parent class
        super().__init__(
            characters=self.PHONEME_INVENTORY,
            punctuations=[],
            pad=normalized_pad,
            eos=normalized_eos,
            bos=normalized_bos,
            blank=normalized_blank,
            is_unique=True,
            is_sorted=False  # We handle sorting manually
        )
        
        # print(f"[INFO] CuoiChamPhonemes initialized:")
        # print(f"  - Phonemes: {len(self.PHONEME_INVENTORY)}")  
        # print(f"  - Vocabulary: {len(self.vocab)} (including special tokens)")
        # print(f"  - Categories: clusters={len(initial_clusters)}, long_vowels={len(long_vowels)}, diphthongs={len(diphthongs)}")
        # print(f"  - Single: consonants={len(single_consonants)}, vowels={len(single_vowels)}, tones={len(tones_and_symbols)}")
        # print(f"  - Longest phonemes: {[p for p in self.PHONEME_INVENTORY if len(p) >= 3]}")

    def _create_vocab(self):
        """
        Optimized vocabulary creation từ logic cũ
        """
        # Start with phoneme inventory  
        _vocab = self._characters.copy() if isinstance(self._characters, list) else list(self._characters)
        
        # Remove duplicates if needed
        if self.is_unique:
            seen = set()
            unique_vocab = []
            for item in _vocab:
                if item not in seen:
                    seen.add(item)
                    unique_vocab.append(item)
            _vocab = unique_vocab
        
        # Add special tokens in optimal order
        if self._blank is not None and len(self._blank) > 0:
            _vocab = [self._blank] + _vocab
        if self._bos is not None and len(self._bos) > 0:
            _vocab = [self._bos] + _vocab  
        if self._eos is not None and len(self._eos) > 0:
            _vocab = [self._eos] + _vocab
        if self._pad is not None and len(self._pad) > 0:
            _vocab = [self._pad] + _vocab
        
        # Add punctuations
        if isinstance(self._punctuations, str):
            _vocab = _vocab + list(self._punctuations)
        elif isinstance(self._punctuations, list):
            _vocab = _vocab + self._punctuations
        
        self.vocab = _vocab
        
        # Validate uniqueness
        if self.is_unique:
            duplicates = {x for x in self.vocab if self.vocab.count(x) > 1}
            if len(duplicates) > 0:
                raise ValueError(f"Duplicate characters in vocabulary: {duplicates}")

    def tokenize(self, text, **kwargs):
        """
        Optimized tokenization - Pure greedy approach
        
        Why this is sufficient:
        1. Phoneme inventory is pre-sorted (longest first)
        2. Whitespace handling is built-in  
        3. No cross-phoneme ambiguity in IPA notation
        4. Simpler = faster = more reliable
        
        Performance: O(n * m) where n=text_length, m=inventory_size
        """
        if not isinstance(text, str):
            return []
        
        if not text.strip():  # Empty or whitespace-only
            return []
        
        # Unicode normalization for input consistency
        text = unicodedata.normalize('NFC', text)
        
        tokens = []
        i = 0
        
        while i < len(text):
            match_found = False
            
            # Try to match longest phoneme first (greedy)
            for phoneme in self.PHONEME_INVENTORY:
                if text[i:].startswith(phoneme):
                    tokens.append(phoneme)
                    i += len(phoneme)
                    match_found = True
                    break  # Found match, continue with next position
            
            if not match_found:
                # Handle non-phoneme characters
                if text[i].isspace():
                    i += 1  # Skip whitespace silently
                else:
                    # Unknown character - log warning và skip
                    print(f"[WARNING] Unknown character '{text[i]}' at position {i} in text: '{text}'")
                    i += 1  # Skip unknown character
        
        return tokens

    def validate_text(self, text):
        """
        Quick validation cho data preprocessing
        """
        try:
            tokens = self.tokenize(text)
            if not tokens:
                return False
            
            # Check if all tokens are in vocabulary
            for token in tokens:
                if token not in self.vocab:
                    print(f"[ERROR] Token '{token}' not in vocabulary")
                    return False
            
            return True
        except Exception as e:
            print(f"[ERROR] Validation failed for '{text}': {e}")
            return False

    def debug_vocabulary(self):
        """Debug method để kiểm tra vocabulary"""
        print("=== PHONEME INVENTORY ===")
        for i, phoneme in enumerate(self.PHONEME_INVENTORY):
            print(f"{i:3d}: '{phoneme}' (len={len(phoneme)}, unicode={[ord(c) for c in phoneme]})")
        
        print("\n=== VOCABULARY ===")
        for i, token in enumerate(self.vocab):
            print(f"{i:3d}: '{token}' (len={len(token)})")

    def debug_tokenization(self, text):
        """
        Step-by-step tokenization debugging
        """
        print(f"\n=== TOKENIZATION DEBUG: '{text}' ===")
        
        if not isinstance(text, str) or not text.strip():
            print("Empty or invalid input")
            return
        
        text = unicodedata.normalize('NFC', text)
        print(f"Normalized input: '{text}'")
        
        tokens = []
        i = 0
        step = 0
        
        while i < len(text):
            step += 1
            print(f"\nStep {step}: Position {i}, remaining: '{text[i:]}'")
            
            match_found = False
            for phoneme in self.PHONEME_INVENTORY:
                if text[i:].startswith(phoneme):
                    tokens.append(phoneme)
                    print(f"  ✓ Matched: '{phoneme}' (length {len(phoneme)})")
                    i += len(phoneme)
                    match_found = True
                    break
            
            if not match_found:
                if text[i].isspace():
                    print(f"  → Skipping whitespace")
                    i += 1
                else:
                    print(f"  ✗ Unknown character: '{text[i]}'")
                    i += 1
        
        print(f"\nFinal tokens: {tokens}")
        print(f"Token count: {len(tokens)}")

    def benchmark_tokenization(self, test_texts, iterations=1000):
        """
        Performance benchmark
        """
        import time
        
        print(f"=== TOKENIZATION BENCHMARK ===")
        print(f"Test texts: {len(test_texts)}")
        print(f"Iterations per text: {iterations}")
        
        total_chars = sum(len(text) for text in test_texts)
        total_time = 0
        
        for text in test_texts:
            start_time = time.time()
            
            for _ in range(iterations):
                self.tokenize(text)
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            chars_per_sec = (len(text) * iterations) / elapsed
            print(f"'{text[:30]}...': {chars_per_sec:.0f} chars/sec")
        
        overall_chars_per_sec = (total_chars * iterations) / total_time
        print(f"\nOverall performance: {overall_chars_per_sec:.0f} characters/second")

    def to_config(self):
        """
        Export configuration cho serialization
        """
        from TTS.tts.configs.shared_configs import CharactersConfig
        
        return CharactersConfig(
            characters=self.PHONEME_INVENTORY,
            punctuations=self._punctuations,
            pad=self._pad,
            eos=self._eos,
            bos=self._bos,
            blank=self._blank,
            is_unique=self.is_unique,
            is_sorted=self.is_sorted,
        )

    @staticmethod
    def init_from_config(config):
        """
        Safe factory method từ logic cũ
        """
        try:
            if hasattr(config, 'characters') and config.characters is not None:
                char_config = config.characters
                
                # Extract parameters safely
                init_params = {}
                for param in ['pad', 'eos', 'bos', 'blank']:
                    if hasattr(char_config, param):
                        init_params[param] = getattr(char_config, param)
                    elif isinstance(char_config, dict) and param in char_config:
                        init_params[param] = char_config[param]
                
                characters = CuoiChamPhonemes(**init_params)
            else:
                characters = CuoiChamPhonemes()
            
            # Update config
            try:
                updated_config = replace(config, characters=characters.to_config())
            except:
                config.characters = characters.to_config()
                updated_config = config
            
            return characters, updated_config
            
        except Exception as e:
            print(f"[ERROR] init_from_config failed: {e}")
            return CuoiChamPhonemes(), config


# # ========================= TESTING =========================
# if __name__ == "__main__":
#     print("=== CuoiChamPhonemes - Simplified Version Test ===")
    
#     # Initialize
#     phonemes = CuoiChamPhonemes()
    
#     # Debug vocabulary structure
#     phonemes.debug_vocabulary()
    
#     # Test cases
#     test_cases = [
#         "kʰraː¹ maː³ ciː⁴",
#         "pʰraː² klaː¹", 
#         "bləːj¹ cəːp⁷",
#         "tʰaːw³ heːt⁷ˢ daːk⁷ saː¹",
#         "ɲɐː² ɣɛɲ¹ kɛːn¹"
#     ]
    
#     print("\n=== TOKENIZATION TESTS ===")
#     for i, test_text in enumerate(test_cases, 1):
#         print(f"\n--- Test {i}: '{test_text}' ---")
        
#         tokens = phonemes.tokenize(test_text)
#         is_valid = phonemes.validate_text(test_text)
        
#         print(f"Tokens: {tokens}")
#         print(f"Valid: {is_valid}")
#         print(f"Token count: {len(tokens)}")
        
#         # Reconstruction check
#         reconstructed = ''.join(tokens)
#         original_no_space = test_text.replace(' ', '')
#         matches = (reconstructed == original_no_space)
#         print(f"Reconstruction matches: {matches}")
        
#         if not matches or not is_valid:
#             phonemes.debug_tokenization(test_text)
    
#     # Performance test
#     print("\n=== PERFORMANCE TEST ===")
#     phonemes.benchmark_tokenization(test_cases[:3], iterations=100)
    
#     print("\n=== Test completed ===")



#### test ######
class CuoiChamPhonemesWithLabeling(BaseCharacters):
    """
    Space-based tokenization với phoneme inventory được phân loại theo cấu trúc ngôn ngữ học.
    
    Architectural differences from CuoiChamPhonemes:
    1. Tokenization: Space-splitting (O(n)) vs Greedy matching (O(n×m))
    2. Phoneme organization: Category-based (initials/nuclei/codas/tones) vs Flat inventory
    3. Use case: Pre-labeled data vs Raw text
    
    Linguistic motivation:
    - Initials (phụ âm đầu): Onset consonants/clusters
    - Nuclei (nguyên âm): Vowel nucleus của syllable
    - Codas (phụ âm cuối): Syllable-final consonants với labeling
    - Tones (thanh điệu): Suprasegmental features
    """
    
    def __init__(self, pad="<PAD>", eos="<EOS>", bos="<BOS>", blank="<BLNK>", **kwargs):
        
        # Phụ âm đầu (Initials): Onsets including clusters
        initials = [
            # 'p', 'b', 't', 'd', 'k', 'c',
            #  'tʰ', 'kʰ',            
            # 'pʰr', 'kʰr',            
            # 'bl', 'pl', 'kl',                 
            # 'm', 'n', 'ŋ', 'ɲ',            
            # 'l', 'r', 'f',                  
            # 's', 'h', 'ɣ',                 
            # 'v', 'z',                       
            # 'j', 'w',                       
            # 'ʈ', 'ʔ'      
            'pʰr', 'kʰr', 'bl', 'pl', 'kl', 
            
            # 2. Các tổ hợp có -w- được Nguyễn Hữu Hoành xác nhận, rất quan trọng cho tính đầy đủ
            'kʰw', 'kw', 'hw',
            

            'pʰ', 'tʰ', 'kʰ',
            
            # 4. Các phụ âm tắc (stops)
            'p', 't', 'c', 'k', 'ʔ',  
            'b', 'd',
            
            # 5. Các phụ âm mũi (nasals)
            'm', 'n', 'ɲ', 'ŋ',
            
            # 6. Các phụ âm xát, bên, rung và tiếp cận
            'v', 's', 'z', 'h', 'ɣ', 'l', 'r', 'j', 'w',
            
            # 7. Phụ âm quặt lưỡi, được cả hai xác nhận gián tiếp qua các ví dụ
            'ʈ'                 
        ]
       
        # Nguyên âm (Nuclei): 
        nuclei = [
            
            # 'i', 'ɨ', 'u', 
            # 'e', 'ə', 'o',
            # 'ɛ', 'ɔ', 'a', 'ɐ', 'ɒ',
            # 'ʌ',
            # 'iː', 'ɨː', 'uː',
            # 'eː', 'əː', 'oː',
            # 'ɛː', 'ɔː', 'aː', 'ɐː', 'ɒː',
            # 'ia', 'ua', 'ɨə'
            'iː', 'eː', 'ɛː', 'aː', 'ɔː', 'oː', 'uː', 
            'ɨː', 'əː', 'ɐː', 'ɒː',
            'ia', 'ua', 'ɨə', # Âm vị được Ferlus và Nguyễn Hữu Hoành xác nhận
            'ie',       # Được Nguyễn Hữu Hoành xác nhận có trong từ bản địa và xuất hiện thật trong dataset

            'i', 'e', 'ɛ', 'a', 'ɔ', 'o', 'u',
            'ɨ', 'ə', 'ɐ', 'ɒ', 'ʌ'
        ]
      
        codas = [
            'p_coda', 't_coda', 'k_coda', 
            'm_coda', 'n_coda', 'ŋ_coda',
            'j_coda', 'w_coda', 'l_coda'             
        ]
        
        # Thanh điệu (Tones): Tone markers
        tones = [
            'T¹', 'T²', 'T³', 'T⁴', 'T⁵', 'T⁶', 
            'T⁷',   'T⁸' ,'ˢ'
        ]
        
        def normalize_phoneme(phoneme):
            """
            NFC normalization ensures consistent representation.
            Critical for IPA symbols với combining diacritics.
            """
            return unicodedata.normalize('NFC', phoneme)
        
        # Normalize all categories
        self.INITIALS = sorted([normalize_phoneme(p) for p in initials], key=len, reverse=True)
        self.NUCLEI = sorted([normalize_phoneme(p) for p in nuclei], key=len, reverse=True)
        self.CODAS = sorted([normalize_phoneme(p) for p in codas], key=len, reverse=True)
        self.TONES = sorted([normalize_phoneme(p) for p in tones], key=len, reverse=True)
        
        # Combine into single inventory (maintaining category order for debugging)
        all_phonemes = self.INITIALS + self.NUCLEI + self.CODAS + self.TONES
        
        # Remove duplicates while preserving order
        seen = set()
        unique_phonemes = []
        for phoneme in all_phonemes:
            if phoneme not in seen:
                seen.add(phoneme)
                unique_phonemes.append(phoneme)
        
        # CRITICAL: Sort by length for potential future greedy fallback
        self.PHONEME_INVENTORY = sorted(unique_phonemes, key=len, reverse=True)
        
        # Normalize special tokens
        normalized_pad = normalize_phoneme(pad)
        normalized_eos = normalize_phoneme(eos)
        normalized_bos = normalize_phoneme(bos)
        normalized_blank = normalize_phoneme(blank)
        
        # Initialize parent class
        super().__init__(
            characters=self.PHONEME_INVENTORY,
            punctuations=[],
            pad=normalized_pad,
            eos=normalized_eos,
            bos=normalized_bos,
            blank=normalized_blank,
            is_unique=True,
            is_sorted=False  # Manual sorting already done
        )
        
        print(f"[INFO] CuoiChamPhonemesWithLabeling initialized:")
        print(f"  - Initials: {len(self.INITIALS)} (includes clusters)")
        print(f"  - Nuclei: {len(self.NUCLEI)} (short/long vowels + diphthongs)")
        print(f"  - Codas: {len(self.CODAS)} (labeled with '_coda' suffix)")
        print(f"  - Tones: {len(self.TONES)}")
        print(f"  - Total phonemes: {len(self.PHONEME_INVENTORY)}")
        print(f"  - Vocabulary size: {len(self.vocab)} (including special tokens)")
        print(f"  - Longest phonemes: {[p for p in self.PHONEME_INVENTORY if len(p) >= 3][:5]}")

    def _create_vocab(self):
        """
        Optimized vocabulary creation matching CuoiChamPhonemes logic.
        Order: [PAD] [EOS] [BOS] [BLANK] + phonemes + punctuations
        """
        _vocab = self._characters.copy() if isinstance(self._characters, list) else list(self._characters)
        
        # Remove duplicates if needed
        if self.is_unique:
            seen = set()
            unique_vocab = []
            for item in _vocab:
                if item not in seen:
                    seen.add(item)
                    unique_vocab.append(item)
            _vocab = unique_vocab
        
        # Add special tokens in optimal order (matching parent class behavior)
        if self._blank is not None and len(self._blank) > 0:
            _vocab = [self._blank] + _vocab
        if self._bos is not None and len(self._bos) > 0:
            _vocab = [self._bos] + _vocab
        if self._eos is not None and len(self._eos) > 0:
            _vocab = [self._eos] + _vocab
        if self._pad is not None and len(self._pad) > 0:
            _vocab = [self._pad] + _vocab
        
        # Add punctuations
        if isinstance(self._punctuations, str):
            _vocab = _vocab + list(self._punctuations)
        elif isinstance(self._punctuations, list):
            _vocab = _vocab + self._punctuations
        
        self.vocab = _vocab
        
        # Validate uniqueness
        if self.is_unique:
            duplicates = {x for x in self.vocab if self.vocab.count(x) > 1}
            if len(duplicates) > 0:
                raise ValueError(f"Duplicate characters in vocabulary: {duplicates}")

    def tokenize(self, text, **kwargs):
        """
        Space-based tokenization: O(n) complexity.
        
        Assumption: Input text is pre-tokenized với spaces as delimiters.
        Example: "t a j_coda T⁵ kl ɔː ŋ_coda T²"
        
        Advantages:
        - Fast: Single split operation
        - Predictable: No ambiguity in token boundaries
        - Label-friendly: Explicit coda markers preserved
        
        Disadvantages:
        - Requires preprocessing
        - Cannot handle raw text
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # Unicode normalization for consistency
        text = unicodedata.normalize('NFC', text)
        
        # Simple split by whitespace
        tokens = text.strip().split()
        
        return tokens

    def validate_text(self, text):
        """
        Validation với detailed error reporting.
        """
        try:
            tokens = self.tokenize(text)
            if not tokens:
                return False
            
            invalid_tokens = []
            for token in tokens:
                if token not in self.vocab:
                    invalid_tokens.append(token)
            
            if invalid_tokens:
                print(f"[ERROR] Invalid tokens found: {invalid_tokens}")
                print(f"[ERROR] In text: '{text}'")
                return False
            
            return True
        except Exception as e:
            print(f"[ERROR] Validation failed for '{text}': {e}")
            return False

    def debug_vocabulary(self):
        """
        Category-aware vocabulary debugging.
        """
        print("\n=== PHONEME INVENTORY BY CATEGORY ===")
        
        print(f"\n[INITIALS] ({len(self.INITIALS)} items):")
        for i, p in enumerate(self.INITIALS):
            print(f"  {i:2d}: '{p}'")
        
        print(f"\n[NUCLEI] ({len(self.NUCLEI)} items):")
        for i, p in enumerate(self.NUCLEI):
            print(f"  {i:2d}: '{p}'")
        
        print(f"\n[CODAS] ({len(self.CODAS)} items):")
        for i, p in enumerate(self.CODAS):
            print(f"  {i:2d}: '{p}'")
        
        print(f"\n[TONES] ({len(self.TONES)} items):")
        for i, p in enumerate(self.TONES):
            print(f"  {i:2d}: '{p}'")
        
        print(f"\n=== FULL VOCABULARY ({len(self.vocab)} items) ===")
        for i, token in enumerate(self.vocab):
            print(f"{i:3d}: '{token}'")

    def debug_tokenization(self, text):
        """
        Step-by-step tokenization debugging với category identification.
        """
        print(f"\n=== TOKENIZATION DEBUG: '{text}' ===")
        
        if not isinstance(text, str) or not text.strip():
            print("Empty or invalid input")
            return
        
        text = unicodedata.normalize('NFC', text)
        print(f"Normalized input: '{text}'")
        
        tokens = self.tokenize(text)
        print(f"\nTokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        
        # Category analysis
        for i, token in enumerate(tokens):
            category = "UNKNOWN"
            if token in self.INITIALS:
                category = "INITIAL"
            elif token in self.NUCLEI:
                category = "NUCLEUS"
            elif token in self.CODAS:
                category = "CODA"
            elif token in self.TONES:
                category = "TONE"
            elif token in [self._pad, self._eos, self._bos, self._blank]:
                category = "SPECIAL"
            
            in_vocab = token in self.vocab
            status = "✓" if in_vocab else "✗"
            print(f"  {i:2d}: {status} '{token}' → {category}")

    def benchmark_tokenization(self, test_texts, iterations=1000):
        """
        Performance benchmark (should be faster than greedy approach).
        """
        import time
        
        print(f"\n=== TOKENIZATION BENCHMARK (space-based) ===")
        print(f"Test texts: {len(test_texts)}")
        print(f"Iterations per text: {iterations}")
        
        total_tokens = 0
        total_time = 0
        
        for text in test_texts:
            start_time = time.time()
            
            for _ in range(iterations):
                tokens = self.tokenize(text)
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            tokens_per_sec = (len(tokens) * iterations) / elapsed
            total_tokens += len(tokens)
            print(f"'{text[:40]}...': {tokens_per_sec:.0f} tokens/sec")
        
        overall_tokens_per_sec = (total_tokens * iterations) / total_time
        print(f"\nOverall performance: {overall_tokens_per_sec:.0f} tokens/second")

    def to_config(self):
        """
        Export configuration for serialization.
        """
        return CharactersConfig(
            characters=self.PHONEME_INVENTORY,
            punctuations=self._punctuations,
            pad=self._pad,
            eos=self._eos,
            bos=self._bos,
            blank=self._blank,
            is_unique=self.is_unique,
            is_sorted=self.is_sorted,
        )

    @staticmethod
    def init_from_config(config):
        """
        Safe factory method matching CuoiChamPhonemes pattern.
        """
        try:
            if hasattr(config, 'characters') and config.characters is not None:
                char_config = config.characters
                
                init_params = {}
                for param in ['pad', 'eos', 'bos', 'blank']:
                    if hasattr(char_config, param):
                        init_params[param] = getattr(char_config, param)
                    elif isinstance(char_config, dict) and param in char_config:
                        init_params[param] = char_config[param]
                
                characters = CuoiChamPhonemesWithLabeling(**init_params)
            else:
                characters = CuoiChamPhonemesWithLabeling()
            
            try:
                updated_config = replace(config, characters=characters.to_config())
            except:
                config.characters = characters.to_config()
                updated_config = config
            
            return characters, updated_config
            
        except Exception as e:
            print(f"[ERROR] init_from_config failed: {e}")
            return CuoiChamPhonemesWithLabeling(), config


# ========================= COMPREHENSIVE TESTING =========================
if __name__ == '__main__':
    print("=== CuoiChamPhonemesWithLabeling - Refactored Version Test ===\n")

    # 1. Initialize
    try:
        phonemes_processor = CuoiChamPhonemesWithLabeling()
        print("\n[SUCCESS] Initialization completed.\n")
    except Exception as e:
        print(f"\n[FAILURE] Initialization error: {e}")
        exit()

    # 2. Test cases (space-separated phonemes)
    test_cases = [
        "t a j_coda T⁵",
        "kl ɔː ŋ_coda T² t ɨə j_coda T³ d aː k_coda T⁷",
        "h aː j_coda T¹",
        "pʰr aː T² kʰr aː T¹",
        "invalid_token s aː T¹"  # Invalid token test
    ]

    # 3. Basic tokenization tests
    print("=== TOKENIZATION TESTS ===")
    for i, text in enumerate(test_cases, 1):
        print(f"\n>>> Test Case {i}: '{text}'")
        
        tokens = phonemes_processor.tokenize(text)
        print(f"    Tokens: {tokens}")
        print(f"    Count: {len(tokens)}")
        
        is_valid = phonemes_processor.validate_text(text)
        print(f"    Valid: {is_valid}")
        
        if not is_valid:
            phonemes_processor.debug_tokenization(text)

    # 4. Category structure verification
    print("\n=== CATEGORY STRUCTURE ===")
    print(f"Initials: {len(phonemes_processor.INITIALS)}")
    print(f"Nuclei: {len(phonemes_processor.NUCLEI)}")
    print(f"Codas: {len(phonemes_processor.CODAS)}")
    print(f"Tones: {len(phonemes_processor.TONES)}")
    print(f"Total unique: {len(phonemes_processor.PHONEME_INVENTORY)}")

    # 5. Performance benchmark
    print("\n=== PERFORMANCE BENCHMARK ===")
    valid_tests = [t for t in test_cases if phonemes_processor.validate_text(t)]
    phonemes_processor.benchmark_tokenization(valid_tests[:3], iterations=1000)

    # 6. Detailed debugging (optional)
    print("\n=== DETAILED VOCABULARY DUMP (first valid test) ===")
    phonemes_processor.debug_tokenization(test_cases[0])

    print("\n=== Test completed successfully ===")