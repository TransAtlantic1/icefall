#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Callable, List, Union

import sentencepiece as spm
from k2 import SymbolTable

from hybrid_text import hybrid_tokenize


class Tokenizer:
    text2word: Callable[[str], List[str]]

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Lang related options")
        group.add_argument("--lang", type=Path, help="Path to lang directory.")
        group.add_argument(
            "--lang-type",
            type=str,
            default=None,
            help="Tokenizer type. If omitted, use lang_dir/lang_type.",
        )

    @staticmethod
    def Load(lang_dir: Path, lang_type: str = "", oov: str = "<unk>"):
        if not lang_type:
            if (lang_dir / "lang_type").exists():
                lang_type = (lang_dir / "lang_type").read_text().strip()
            elif (lang_dir / "bpe.model").exists():
                lang_type = "bpe"
            else:
                raise AssertionError("lang_type not specified.")

        if lang_type == "hybrid":
            return HybridTokenizer(lang_dir, oov=oov)
        if lang_type == "bpe":
            assert (lang_dir / "bpe.model").exists(), (
                f"Missing bpe.model in {lang_dir}"
            )
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(str(lang_dir / "bpe.model"))
            return tokenizer

        raise NotImplementedError(f"{lang_type} not supported.")

    load = Load

    def piece_to_id(self, piece: str) -> int:
        raise NotImplementedError

    def id_to_piece(self, idx: int) -> str:
        raise NotImplementedError

    def get_piece_size(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.get_piece_size()

    def EncodeAsIdsBatch(self, texts: List[str]) -> List[List[int]]:
        raise NotImplementedError

    def EncodeAsPiecesBatch(self, texts: List[str]) -> List[List[str]]:
        raise NotImplementedError

    def EncodeAsIds(self, text: str) -> List[int]:
        return self.EncodeAsIdsBatch([text])[0]

    def EncodeAsPieces(self, text: str) -> List[str]:
        return self.EncodeAsPiecesBatch([text])[0]

    def encode(
        self, inputs: Union[str, List[str]], out_type=int
    ) -> Union[List, List[List]]:
        if not inputs:
            return []
        if isinstance(inputs, list):
            if out_type is int:
                return self.EncodeAsIdsBatch(inputs)
            if out_type is str:
                return self.EncodeAsPiecesBatch(inputs)
        if out_type is int:
            return self.EncodeAsIds(inputs)
        if out_type is str:
            return self.EncodeAsPieces(inputs)
        raise RuntimeError(f"Unsupported out_type: {out_type}")

    def DecodeIdsBatch(self, ids_batch: List[List[int]]) -> List[str]:
        raise NotImplementedError

    def DecodePiecesBatch(self, pieces_batch: List[List[str]]) -> List[str]:
        raise NotImplementedError

    def decode(
        self,
        inputs: Union[int, List[int], List[str], List[List[int]], List[List[str]]],
    ) -> Union[List[str], str]:
        if inputs is None or inputs == []:
            return ""
        if isinstance(inputs, int):
            return self.id_to_piece(inputs)
        if isinstance(inputs, str):
            raise TypeError("Cannot decode from type str.")
        if isinstance(inputs[0], list):
            if not inputs[0] or isinstance(inputs[0][0], int):
                return self.DecodeIdsBatch(inputs)
            if isinstance(inputs[0][0], str):
                return self.DecodePiecesBatch(inputs)
        if isinstance(inputs[0], int):
            return self.DecodeIdsBatch([inputs])[0]
        if isinstance(inputs[0], str):
            return self.DecodePiecesBatch([inputs])[0]
        raise RuntimeError("Unknown input type")


class HybridTokenizer(Tokenizer):
    def __init__(self, lang_dir: Path, oov: str = "<unk>"):
        assert (lang_dir / "tokens.txt").exists(), f"Missing tokens.txt in {lang_dir}"
        assert (lang_dir / "english_bpe.model").exists(), (
            f"Missing english_bpe.model in {lang_dir}"
        )

        token_table = SymbolTable.from_file(lang_dir / "tokens.txt")
        self._id2sym = token_table._id2sym
        self._sym2id = token_table._sym2id
        self.oov = oov
        self.oov_id = self._sym2id[oov]

        self.english_sp = spm.SentencePieceProcessor()
        self.english_sp.load(str(lang_dir / "english_bpe.model"))
        english_unk_id = self.english_sp.unk_id()
        self.english_pieces = {
            self.english_sp.id_to_piece(i)
            for i in range(self.english_sp.vocab_size())
            if i != english_unk_id
        }

    def _encode_english(self, run: str) -> List[str]:
        return self.english_sp.encode(run, out_type=str)

    def _text_to_pieces(self, text: str) -> List[str]:
        return hybrid_tokenize(text, english_encoder=self._encode_english)

    def _pieces_to_text(self, pieces: List[str]) -> str:
        english_buffer: List[str] = []
        parts: List[str] = []

        def flush_english() -> None:
            if english_buffer:
                parts.append(self.english_sp.decode_pieces(english_buffer))
                english_buffer.clear()

        for piece in pieces:
            if piece in self.english_pieces:
                english_buffer.append(piece)
                continue
            flush_english()
            parts.append(piece)

        flush_english()
        return "".join(parts)

    def piece_to_id(self, piece: str) -> int:
        return self._sym2id.get(piece, self.oov_id)

    def id_to_piece(self, idx: int) -> str:
        return self._id2sym[idx]

    def get_piece_size(self) -> int:
        return len(self._sym2id)

    def EncodeAsIdsBatch(self, texts: List[str]) -> List[List[int]]:
        return [
            [self.piece_to_id(piece) for piece in self._text_to_pieces(text)]
            for text in texts
        ]

    def EncodeAsPiecesBatch(self, texts: List[str]) -> List[List[str]]:
        return [self._text_to_pieces(text) for text in texts]

    def DecodeIdsBatch(self, ids_batch: List[List[int]]) -> List[str]:
        return [
            self._pieces_to_text([self.id_to_piece(idx) for idx in ids])
            for ids in ids_batch
        ]

    def DecodePiecesBatch(self, pieces_batch: List[List[str]]) -> List[str]:
        return [self._pieces_to_text(pieces) for pieces in pieces_batch]
