#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import k2
import sentencepiece as spm
import torch

_ICEFALL_ROOT = Path(__file__).resolve().parents[5]
if str(_ICEFALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ICEFALL_ROOT))

from icefall.lexicon import write_lexicon


Lexicon = List[Tuple[str, List[str]]]


def write_mapping(filename: Path, sym2id: Dict[str, int]) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for sym, i in sym2id.items():
            f.write(f"{sym} {i}\n")


def add_disambig_symbols(lexicon: Lexicon) -> Tuple[Lexicon, int]:
    count = defaultdict(int)
    for _, tokens in lexicon:
        count[" ".join(tokens)] += 1

    issubseq = defaultdict(int)
    for _, tokens in lexicon:
        tokens = tokens.copy()
        tokens.pop()
        while tokens:
            issubseq[" ".join(tokens)] = 1
            tokens.pop()

    ans = []
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_symbol_of = defaultdict(int)

    for word, tokens in lexicon:
        tokenseq = " ".join(tokens)
        assert tokenseq != ""
        if issubseq[tokenseq] == 0 and count[tokenseq] == 1:
            ans.append((word, tokens))
            continue

        cur_disambig = last_used_disambig_symbol_of[tokenseq]
        if cur_disambig == 0:
            cur_disambig = first_allowed_disambig
        else:
            cur_disambig += 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
        last_used_disambig_symbol_of[tokenseq] = cur_disambig
        tokenseq += f" #{cur_disambig}"
        ans.append((word, tokenseq.split()))

    return ans, max_disambig


def add_self_loops(
    arcs: List[List[Any]], disambig_token: int, disambig_word: int
) -> List[List[Any]]:
    states_needs_self_loops = set()
    for arc in arcs:
        src, _, _, olabel, _ = arc
        if olabel != 0:
            states_needs_self_loops.add(src)

    ans = []
    for state in states_needs_self_loops:
        ans.append([state, state, disambig_token, disambig_word, 0])

    return arcs + ans


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang-dir", type=Path, required=True)
    return parser.parse_args()


def lexicon_to_fst_no_sil(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    loop_state = 0
    next_state = 1
    arcs = []

    assert token2id["<blk>"] == 0
    assert word2id["<eps>"] == 0
    eps = 0

    for word, pieces in lexicon:
        cur_state = loop_state
        word_id = word2id[word]
        piece_ids = [token2id[piece] for piece in pieces]
        for i in range(len(piece_ids) - 1):
            out_label = word_id if i == 0 else eps
            arcs.append([cur_state, next_state, piece_ids[i], out_label, 0])
            cur_state = next_state
            next_state += 1

        out_label = word_id if len(piece_ids) == 1 else eps
        arcs.append([cur_state, loop_state, piece_ids[-1], out_label, 0])

    if need_self_loops:
        arcs = add_self_loops(
            arcs,
            disambig_token=token2id["#0"],
            disambig_word=word2id["#0"],
        )

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = "\n".join(" ".join(str(item) for item in arc) for arc in sorted(arcs))
    return k2.Fsa.from_str(arcs, acceptor=False)


def main():
    args = get_args()
    lang_dir = args.lang_dir

    char_tokens_path = lang_dir / "char_tokens.txt"
    english_model_path = lang_dir / "english_bpe.model"
    assert char_tokens_path.is_file(), char_tokens_path
    assert english_model_path.is_file(), english_model_path

    char_tokens = [
        line.strip()
        for line in char_tokens_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    sp = spm.SentencePieceProcessor()
    sp.load(str(english_model_path))
    unk_id = sp.unk_id()
    english_pieces = [
        sp.id_to_piece(i)
        for i in range(sp.vocab_size())
        if i != unk_id
    ]

    token_sym_table: Dict[str, int] = {
        "<blk>": 0,
        "<sos/eos>": 1,
        "<unk>": 2,
    }
    for token in sorted(char_tokens):
        if token not in token_sym_table:
            token_sym_table[token] = len(token_sym_table)
    for piece in english_pieces:
        if piece not in token_sym_table:
            token_sym_table[piece] = len(token_sym_table)

    lexicon: Lexicon = []
    word_sym_table: Dict[str, int] = {
        "<eps>": 0,
        "!SIL": 1,
        "<SPOKEN_NOISE>": 2,
        "<UNK>": 3,
    }

    words = sorted(
        token for token in token_sym_table if token not in {"<blk>", "<sos/eos>", "<unk>"}
    )
    for word in words:
        word_sym_table[word] = len(word_sym_table)
        lexicon.append((word, [word]))

    lexicon.append(("<UNK>", ["<unk>"]))

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    next_token_id = max(token_sym_table.values()) + 1
    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        if disambig not in token_sym_table:
            token_sym_table[disambig] = next_token_id
            next_token_id += 1

    for special_word in ("#0", "<s>", "</s>"):
        if special_word not in word_sym_table:
            word_sym_table[special_word] = len(word_sym_table)

    write_mapping(lang_dir / "tokens.txt", token_sym_table)
    write_mapping(lang_dir / "words.txt", word_sym_table)
    write_lexicon(lang_dir / "lexicon.txt", lexicon)
    write_lexicon(lang_dir / "lexicon_disambig.txt", lexicon_disambig)
    (lang_dir / "lang_type").write_text("hybrid", encoding="utf-8")

    L = k2.arc_sort(
        lexicon_to_fst_no_sil(lexicon, token_sym_table, word_sym_table, False)
    )
    L_disambig = k2.arc_sort(
        lexicon_to_fst_no_sil(
            lexicon_disambig, token_sym_table, word_sym_table, True
        )
    )

    torch.save(L.as_dict(), lang_dir / "L.pt")
    torch.save(L_disambig.as_dict(), lang_dir / "L_disambig.pt")


if __name__ == "__main__":
    main()
