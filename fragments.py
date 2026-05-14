from __future__ import annotations

import csv
import hashlib
import io
from typing import Dict, List, Set

import dna_codec


BASES = "ACGT"


def clean_dna(seq: str) -> str:
    return dna_codec.clean_dna_text(seq)


def split_dna(seq: str, payload_len: int) -> List[str]:
    seq = clean_dna(seq)
    if payload_len <= 0:
        payload_len = 100
    return [seq[i:i + payload_len] for i in range(0, len(seq), payload_len)]


def max_homopolymer(seq: str) -> int:
    seq = clean_dna(seq)
    if not seq:
        return 0

    cur = 1
    mx = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 1
    return mx


def gc_fraction(seq: str) -> float:
    seq = clean_dna(seq)
    if not seq:
        return 0.0
    return sum(b in "GC" for b in seq) / len(seq)


def _hash_to_dna(seed: str, length: int) -> str:
    """
    Deterministically generate a DNA candidate from a seed.
    This is not random at runtime; same seed gives same sequence.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    bits = []
    for byte in digest:
        for shift in (6, 4, 2, 0):
            bits.append((byte >> shift) & 0b11)

    out = []
    i = 0
    while len(out) < length:
        if i >= len(bits):
            digest = hashlib.sha256((seed + "|" + str(i)).encode("utf-8")).digest()
            bits = []
            for byte in digest:
                for shift in (6, 4, 2, 0):
                    bits.append((byte >> shift) & 0b11)
            i = 0

        out.append(BASES[bits[i]])
        i += 1

    return "".join(out)


def _index_score(index_seq: str, left_context: str, right_context: str) -> float:
    """
    Lower score is better.
    Penalizes homopolymers, poor GC balance, and boundary repeats.
    """
    full = clean_dna(left_context) + clean_dna(index_seq) + clean_dna(right_context)
    idx = clean_dna(index_seq)

    score = 0.0

    hp_full = max_homopolymer(full)
    hp_idx = max_homopolymer(idx)

    score += max(0, hp_full - 2) * 100.0
    score += max(0, hp_idx - 2) * 80.0

    gc = gc_fraction(idx)
    score += abs(gc - 0.50) * 20.0

    # Boundary penalty: avoid FBR ending with same base as index start
    if left_context and idx and clean_dna(left_context)[-1:] == idx[:1]:
        score += 10.0

    # Boundary penalty: avoid index ending with same base as payload/RBR start
    if right_context and idx and idx[-1:] == clean_dna(right_context)[:1]:
        score += 10.0

    return score


def generate_strand_index(
    strand_no: int,
    index_len: int,
    used_indices: Set[str],
    left_context: str = "",
    right_context: str = "",
    max_hp_allowed: int = 2,
) -> str:
    """
    Generate a unique DNA strand index with low homopolymer tendency.

    It tries multiple deterministic candidates and selects the best one.
    The index is checked together with neighboring sequence context:
        FBR + strand_index + payload_start
    """
    if index_len <= 0:
        return ""

    best_seq = ""
    best_score = float("inf")

    for salt in range(5000):
        seed = f"strand_index|{strand_no}|{index_len}|{salt}"
        cand = _hash_to_dna(seed, index_len)

        if cand in used_indices:
            continue

        score = _index_score(cand, left_context, right_context)

        if score < best_score:
            best_seq = cand
            best_score = score

        full_context = clean_dna(left_context) + cand + clean_dna(right_context)
        if (
            score <= 1e-9
            and max_homopolymer(cand) <= max_hp_allowed
            and max_homopolymer(full_context) <= max_hp_allowed
        ):
            used_indices.add(cand)
            return cand

    if not best_seq:
        raise RuntimeError("Could not generate a unique strand index.")

    used_indices.add(best_seq)
    return best_seq


def prepare_dna_strands(
    dna: str,
    payload_len: int,
    fbr: str,
    rbr: str,
    index_len: int = 8,
) -> List[Dict[str, str]]:
    """
    Prepare DNA strands as:
        FBR + strand_index + payload + RBR

    CSV columns:
        No., FBR, Strand index, Payload, RBR
    """
    dna = clean_dna(dna)
    fbr = clean_dna(fbr)
    rbr = clean_dna(rbr)

    payloads = split_dna(dna, payload_len)
    used_indices: Set[str] = set()
    rows: List[Dict[str, str]] = []

    for i, payload in enumerate(payloads, start=1):
        # right_context starts from payload, because the index touches payload directly.
        right_context = payload[:8] if payload else rbr[:8]

        strand_index = generate_strand_index(
            strand_no=i,
            index_len=index_len,
            used_indices=used_indices,
            left_context=fbr[-8:],
            right_context=right_context,
            max_hp_allowed=2,
        )

        rows.append(
            {
                "No.": str(i),
                "FBR": fbr,
                "Strand index": strand_index,
                "Payload": payload,
                "RBR": rbr,
            }
        )

    return rows


def strand_rows_to_csv(rows: List[Dict[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["No.", "FBR", "Strand index", "Payload", "RBR"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def strand_rows_to_preview(rows: List[Dict[str, str]], max_rows: int = 5) -> str:
    lines = []
    for row in rows[:max_rows]:
        full_strand = (
            row["FBR"]
            + row["Strand index"]
            + row["Payload"]
            + row["RBR"]
        )
        lines.append(
            f"{row['No.']}. "
            f"FBR({len(row['FBR'])}) + "
            f"Index({row['Strand index']}) + "
            f"Payload({len(row['Payload'])}) + "
            f"RBR({len(row['RBR'])})"
        )
        lines.append(full_strand)
        lines.append("")
    return "\n".join(lines)