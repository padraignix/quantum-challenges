#!/usr/bin/env python3
"""
qv_extract_flag_resync_fix.py
-----------------------------
Recover 'qv{...}' from sift+|charge| tail with local re-sync and gentle per-byte fixes.

Pipeline:
  1) Sift: vB>=0 and bA==bB (optional basis ALL/Z/X).
  2) Tail-select: keep |charge| >= percentile (default 70th).
  3) Find earliest 'qv{' across initial phases 0..7.
  4) Decode forward MSB-first. While inside braces:
       - If byte not in expected charset [a-z_}] (plus optional ASCII letters/digits),
         try local re-sync by shifting the bit boundary ±1..±2 bits (scored by lookahead printability).
       - If still implausible, try gentle bit repairs: flip 0x40; optionally flip 0x04 for e↔a type errors.
     Stop at first '}' and print the clean flag.
"""

import argparse
import numpy as np
import pandas as pd
import string

ALLOWED_BODY = set(string.ascii_lowercase + string.digits + "_")  # generous but biased to expected
ALLOWED_ANY  = set(string.printable)  # generic printable fallback

def pack_byte_msb(bits_view) -> int:
    b = 0
    for k, v in enumerate(bits_view):  # 8 bits expected
        b |= (int(v) & 1) << (7 - k)
    return b

def is_printable(b: int) -> bool:
    return 32 <= b < 127

def score_future(bits, start_idx, lookahead):
    """Return count of printable/brace bytes over next 'lookahead' bytes from start_idx."""
    score = 0
    i = start_idx
    for _ in range(lookahead):
        if i + 8 > len(bits): break
        bb = pack_byte_msb(bits[i:i+8])
        if is_printable(bb) or bb == ord('}'):
            score += 1
        i += 8
    return score

def find_initial_qv(bits):
    target = b"qv{"
    best = None
    for p in range(8):
        nbytes = (len(bits) - p) // 8
        if nbytes <= 0: continue
        buf = bytearray()
        idx = p
        limit = min(nbytes, 4096)
        for _ in range(limit):
            buf.append(pack_byte_msb(bits[idx:idx+8]))
            idx += 8
        pos = bytes(buf).find(target)
        if pos >= 0:
            bit_start = p + pos * 8
            if best is None or pos < best[1]:
                best = (p, pos, bit_start)
    return best  # (phase, byte_pos, bit_start) or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--charge",   required=True)
    ap.add_argument("--pct", type=float, default=70.0)
    ap.add_argument("--basis", choices=["ALL","Z","X"], default="ALL")
    ap.add_argument("--lookahead", type=int, default=12, help="bytes to score for re-sync")
    ap.add_argument("--max_slips", type=int, default=6, help="max local re-sync attempts")
    args = ap.parse_args()

    ex = pd.read_csv(args.exchange)
    charge = np.load(args.charge)

    # Sift
    sift = (ex["vB"] >= 0) & (ex["bA"] == ex["bB"])
    exk = ex.loc[sift].copy()
    ck  = charge[sift.values]
    if args.basis in ("Z","X"):
        m = (exk["bB"].to_numpy() == args.basis)
        exk = exk.loc[m].reset_index(drop=True)
        ck  = ck[m]
    vB = exk["vB"].to_numpy(np.uint8)

    print(f"[info] total pulses: {len(ex)}")
    print(f"[info] kept after sifting (vB>=0 & bA==bB, basis={args.basis}): {len(exk)}")

    # Tail
    abs_c = np.abs(ck)
    thr = np.percentile(abs_c, args.pct)
    sel = (abs_c >= thr)
    bits = vB[sel]
    print(f"[info] |charge| percentile {args.pct:.1f} -> threshold={thr:.6g}")
    print(f"[info] selected pulses: {sel.sum()} ({100.0*sel.mean():.1f}% of sifted-in-basis)")

    # Initial 'qv{'
    hit = find_initial_qv(bits)
    if hit is None:
        print("[miss] No 'qv{' found in any initial phase (0..7).")
        return
    init_phase, init_byte_pos, i = hit
    print(f"[hit] initial 'qv{{' at phase={init_phase}, byte_pos={init_byte_pos}, bit_pos={i}")

    # Decode with local re-sync + gentle fixes
    out = []
    slips = 0
    inside = True
    while i + 8 <= len(bits):
        b = pack_byte_msb(bits[i:i+8])
        ch = chr(b) if is_printable(b) else None

        if ch is None or (inside and ch not in ALLOWED_ANY):
            # Try local re-sync ±1..±2 bits
            best_shift = 0
            best_score = -1
            best_b = b
            for s in (-2, -1, 1, 2):
                j = i + s
                if j < 0 or j + 8 > len(bits): continue
                bb = pack_byte_msb(bits[j:j+8])
                sc = score_future(bits, j+8, args.lookahead)
                if sc > best_score and (is_printable(bb) or bb == ord('}')):
                    best_score = sc
                    best_shift = s
                    best_b = bb
            if best_shift != 0 and slips < args.max_slips:
                slips += 1
                i = i + best_shift
                b = best_b
                ch = chr(b) if is_printable(b) else None

        # Gentle bit repairs inside braces (common 0x40 / 0x04 flips)
        if inside and ch is not None and ch not in ALLOWED_BODY and b != ord('}'):
            repaired = None
            for delta in (0x40, 0x04, 0x40 ^ 0x04):
                bb = b ^ delta
                cc = chr(bb) if is_printable(bb) else None
                if cc is not None and cc in ALLOWED_BODY:
                    repaired = (bb, cc, delta)
                    break
            if repaired:
                b, ch, delta = repaired
            # else leave as-is; it will appear as punctuation but we continue

        out.append(b)
        i += 8
        if b == ord('}'):
            break

    text = ''.join(chr(x) if is_printable(x) or x == ord('}') else '.' for x in out)
    s = text.find("qv{")
    e = text.find("}", s+3) if s >= 0 else -1
    if s >= 0 and e >= 0:
        flag = text[s:e+1]
        print(f"[FLAG] {flag}")
        print(f"[resync] slips_used={slips}")
    else:
        print("[warn] Did not locate a complete 'qv{...}'. Preview:")
        print(text[:120])

    print("[done]")

if __name__ == "__main__":
    main()
