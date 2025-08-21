#!/usr/bin/env python3
"""
qv_extract_flag.py
------------------
Recover the clean qv{...} flag by:
  1) Sifting: keep vB>=0 and bA==bB (optionally restrict basis).
  2) Selecting the top tail by |charge| (>= percentile).
  3) Packing MSB-first for all byte phases 0..7.
  4) Searching each packed stream for a fully-printable 'qv{...}' ending at the first '}'.

We then choose the best candidate (longest; earliest if tie) and print only the exact flag.
"""

import argparse
import numpy as np
import pandas as pd

def pack_bits_msb_first(bits: np.ndarray, phase: int = 0) -> np.ndarray:
    """Pack 0/1 bits into uint8 bytes (MSB-first) after dropping `phase` bits."""
    if phase:
        bits = bits[phase:]
    n = len(bits) - (len(bits) % 8)
    if n <= 0:
        return np.zeros(0, dtype=np.uint8)
    b8 = bits[:n].astype(np.uint8).reshape(-1, 8)
    shifts = np.array([7,6,5,4,3,2,1,0], dtype=np.uint8)
    return np.sum(b8 << shifts, axis=1).astype(np.uint8)

def is_printable_byte(x: int) -> bool:
    return 32 <= x < 127

def to_printable_text(buf: np.ndarray) -> str:
    return ''.join(chr(x) if is_printable_byte(int(x)) else '\x00' for x in buf)

def find_qv_flags(text: str):
    """Yield (start_idx, end_idx_inclusive, flag_text) for fully printable qv{...} segments."""
    i = 0
    out = []
    while True:
        i = text.find("qv{", i)
        if i < 0:
            break
        j = text.find("}", i + 3)
        if j < 0:
            break
        seg = text[i:j+1]
        if '\x00' not in seg:  # fully printable
            out.append((i, j, seg))
        i = i + 1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", required=True, help="Path to exchange_obf.csv")
    ap.add_argument("--charge",   required=True, help="Path to charge_histogram.npy")
    ap.add_argument("--pct", type=float, default=70.0,
                    help="Percentile on |charge| for tail selection (keep >= pct). Default 70.")
    ap.add_argument("--basis", choices=["ALL","Z","X"], default="ALL",
                    help="Restrict to Bob's basis (ALL/Z/X). Default ALL.")
    args = ap.parse_args()

    # Load
    ex = pd.read_csv(args.exchange)
    charge = np.load(args.charge)

    # Sift: valid click & basis match
    sift = (ex["vB"] >= 0) & (ex["bA"] == ex["bB"])
    exk = ex.loc[sift].copy()
    ck  = charge[sift.values]
    bBk = exk["bB"].to_numpy()

    if args.basis in ("Z","X"):
        m = (bBk == args.basis)
        exk = exk.loc[m].reset_index(drop=True)
        ck  = ck[m]

    vB = exk["vB"].to_numpy(np.uint8)

    print(f"[info] total pulses: {len(ex)}")
    print(f"[info] kept after sifting (vB>=0 & bA==bB, basis={args.basis}): {len(exk)}")

    # Tail select by |charge|
    abs_c = np.abs(ck)
    thr = np.percentile(abs_c, args.pct)
    sel = (abs_c >= thr)
    bits = vB[sel]
    print(f"[info] |charge| percentile {args.pct:.1f} -> threshold = {thr:.6g}")
    print(f"[info] selected pulses: {sel.sum()} ({100.0*sel.mean():.1f}% of sifted-in-basis)")

    # Scan all phases
    candidates = []  # (phase, byte_pos, length, flag_text)
    for phase in range(8):
        packed = pack_bits_msb_first(bits, phase=phase)
        if packed.size == 0:
            continue
        text = to_printable_text(packed)
        hits = find_qv_flags(text)
        for (i, j, seg) in hits:
            candidates.append((phase, i, len(seg), seg))

    if not candidates:
        print("[miss] No fully-printable qv{...} found in any phase (0..7).")
        # For debugging, show a short preview from phase 0
        packed0 = pack_bits_msb_first(bits, phase=0)
        preview = to_printable_text(packed0)[:128].replace('\x00', '.')
        print(f"[peek] phase=0 preview: {preview}")
        return

    # Choose the best: longest; if tie, earliest byte_pos
    candidates.sort(key=lambda t: (t[2], -t[1]), reverse=True)
    phase, pos, L, flag = candidates[0]
    print(f"[FLAG] {flag}")
    print(f"[align] phase={phase} byte_pos={pos} length={L}")

    # If there were alternates, list briefly
    alts = candidates[1:4]
    if alts:
        print("[also found]")
        for (p, i, l, seg) in alts:
            print(f"  phase={p} byte_pos={i} len={l}  {seg}")

    print("[done]")

if __name__ == "__main__":
    main()
