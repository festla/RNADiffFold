# -*- coding: utf-8 -*-
"""
Convert BPSEQ files to sequence and dot-bracket.
- Supports pseudoknots by assigning different bracket tiers: (), [], {}, <>, Aa, Bb, ...
- Can also run in "simple" mode to drop pseudoknots (keep a greedy non-crossing subset).
Usage:
    python bpseq_to_dbn.py /path/to/file_or_dir --mode pk
    python bpseq_to_dbn.py ./bpseq_dir --mode simple
Outputs:
    For each input *.bpseq, writes a *.dbn (3 lines: name, sequence, structure)
"""

import argparse
from pathlib import Path

BRACKETS = [
    ('(', ')'), ('[', ']'), ('{', '}'), ('<', '>'),
    ('A', 'a'), ('B', 'b'), ('C', 'c'), ('D', 'd'),
    ('E', 'e'), ('F', 'f')
]

def parse_bpseq_text(text: str):
    """Parse BPSEQ content -> (sequence:str, pairs:list[int])"""
    triples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '>', ';')):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        i = int(parts[0])
        base = parts[1].upper()
        pair = int(parts[2])
        triples.append((i, base, pair))

    # sort by index (robust if file unsorted)
    triples.sort(key=lambda x: x[0])
    # sanity: ensure 1..N
    if triples and (triples[0][0] != 1 or triples[-1][0] != len(triples)):
        # still proceed; we’ll map by given order
        pass

    seq = ''.join([b for _, b, _ in triples])
    pairs = [p for _, _, p in triples]   # 1-based indices; 0 for unpaired
    return seq, pairs

def has_cross(a, b, i, j):
    """Return True if pairs (a,b) and (i,j) are crossing. (1-based, a<b, i<j)"""
    return (a < i < b < j) or (i < a < j < b)

def assign_levels_non_crossing(pairs):
    """
    Assign base pairs to multiple pseudoknot levels so that
    each level itself is non-crossing. Greedy by ascending left index.
    Returns: levels = [ [(i,j), ...], ... ]
    """
    # Collect pairs as sorted tuples (left<right)
    arcs = []
    for i, j in enumerate(pairs, start=1):
        if j > 0 and j > i:
            arcs.append((i, j))
    arcs.sort(key=lambda x: x[0])  # sort by left index

    levels = []  # list of lists of arcs
    for i, j in arcs:
        placed = False
        for lvl in levels:
            # place into first level with no crossing
            if all(not has_cross(a, b, i, j) for (a, b) in lvl):
                lvl.append((i, j))
                placed = True
                break
        if not placed:
            levels.append([(i, j)])
    return levels

def to_dotbracket_with_pseudoknot(seq, pairs):
    """
    Convert to dot-bracket with multiple bracket tiers if pseudoknots exist.
    """
    L = len(seq)
    dot = ['.'] * L
    levels = assign_levels_non_crossing(pairs)

    if len(levels) > len(BRACKETS):
        print(f"[WARN] Pseudoknot depth {len(levels)} exceeds supported {len(BRACKETS)}. "
              f"Extra levels will be dropped.")
    for lvl_idx, lvl in enumerate(levels):
        if lvl_idx >= len(BRACKETS):
            break
        left_sym, right_sym = BRACKETS[lvl_idx]
        for i, j in lvl:
            dot[i-1] = left_sym
            dot[j-1] = right_sym
    return ''.join(dot)

def to_dotbracket_simple(seq, pairs):
    """
    Convert to standard dot-bracket '()' only (drop pseudoknots).
    Greedy keep a non-crossing subset: place (i,j) if it doesn't cross any kept.
    """
    L = len(seq)
    dot = ['.'] * L
    kept = []
    for i, j in sorted([(i, j) for i, j in enumerate(pairs, start=1) if j > i], key=lambda x: x[0]):
        if all(not has_cross(a, b, i, j) for (a, b) in kept):
            kept.append((i, j))
    for i, j in kept:
        dot[i-1] = '('
        dot[j-1] = ')'
    return ''.join(dot)

def write_dbn(out_path: Path, name: str, seq: str, dot: str):
    out_path.write_text(f">{name}\n{seq}\n{dot}\n", encoding="utf-8")
    print(f"[OK] {out_path}")

def process_file(path: Path, mode: str):
    name = path.stem
    text = path.read_text(encoding="utf-8", errors="ignore")
    seq, pairs = parse_bpseq_text(text)
    if mode == "pk":
        dot = to_dotbracket_with_pseudoknot(seq, pairs)
    else:
        dot = to_dotbracket_simple(seq, pairs)
    out = path.with_suffix(".dbn")
    write_dbn(out, name, seq, dot)

def main():
    ap = argparse.ArgumentParser(description="BPSEQ -> sequence + dot-bracket")
    ap.add_argument("inpath", type=str, help="BPSEQ file or directory")
    ap.add_argument("--mode", choices=["pk", "simple"], default="pk",
                    help="pk: keep pseudoknots with multi-brackets; simple: only () (drop pseudoknots)")
    ap.add_argument("--glob", default="*.bpseq", help="When inpath is a dir, glob pattern (default: *.bpseq)")
    args = ap.parse_args()

    p = Path(args.inpath)
    if p.is_file():
        process_file(p, args.mode)
    elif p.is_dir():
        files = sorted(p.rglob(args.glob))
        if not files:
            print(f"[WARN] No files matched {args.glob} under {p}")
        for f in files:
            process_file(f, args.mode)
    else:
        print(f"[ERR] Not found: {p}")

if __name__ == "__main__":
    main()
