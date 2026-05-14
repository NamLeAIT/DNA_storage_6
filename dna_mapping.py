from __future__ import annotations

import gzip
import io
import lzma
import bz2
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from PIL import Image
except Exception:
    Image = None

import dna_codec
from utils_bits_v2 import bytes_to_bitstring, bitstring_to_bytes, detect_magic
from config import MAPPING_OPTIONS, IMAGE_KINDS


def mapping_to_config(mapping_name: str) -> Dict[str, Any]:
    if mapping_name == "Simple Mapping":
        return {
            "mode": "SIMPLE",
            "scheme_name": "RINF_B16",
            "init_dimer": "TA",
            "whiten": False,
        }
    return {
        "mode": "TABLE",
        "scheme_name": mapping_name,
        "init_dimer": "TA",
        "whiten": False,
    }

def encode_bytes_to_dna(data: bytes, mapping_name: str) -> Tuple[str, str, Dict[str, Any]]:
    bits = bytes_to_bitstring(data)
    if bits == "":
        bits = "0"
    cfg = mapping_to_config(mapping_name)
    dna, digits = dna_codec.encode_bits_to_dna(
        bits,
        scheme_name=cfg["scheme_name"],
        mode=cfg["mode"],
        seed="rn",
        init_dimer=cfg["init_dimer"],
        prepend_one=True,
        whiten=cfg["whiten"],
        target_gc=0.50,
        w_gc=0.0,
        w_motif=0.0,
        ks=(4, 6),
    )
    meta = {
        "mapping": mapping_name,
        "mode": cfg["mode"],
        "scheme_name": cfg["scheme_name"],
        "init_dimer": "TA",
        "bits_len": len(bits),
        "digits_len": len(digits) if isinstance(digits, list) else None,
        "bytes_len": len(data),
    }
    return dna, bits, meta

def decode_dna_with_mapping(dna: str, mapping_name: str) -> Tuple[bytes, str, Dict[str, Any]]:
    cfg = mapping_to_config(mapping_name)
    bits, digits = dna_codec.decode_dna_to_bits(
        dna,
        scheme_name=cfg["scheme_name"],
        mode=cfg["mode"],
        seed="rn",
        init_dimer=cfg["init_dimer"],
        remove_leading_one=True,
        whiten=cfg["whiten"],
        target_gc=0.50,
        w_gc=0.0,
        w_motif=0.0,
        ks=(4, 6),
    )
    data, pad_bits = bitstring_to_bytes(bits, pad_to_byte=True)
    meta = {
        "mapping": mapping_name,
        "bits_len": len(bits),
        "bytes_len": len(data),
        "pad_bits_to_byte": pad_bits,
        "digits_len": len(digits) if isinstance(digits, list) else None,
    }
    return data, bits, meta

def validate_container(data: bytes, magic_kind: str) -> Tuple[bool, str]:
    """Lightweight validation beyond magic signature."""
    try:
        if magic_kind == "zip" or magic_kind in {"docx", "pptx", "xlsx", "epub"}:
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    return False, f"ZIP test failed at {bad}"
            return True, "ZIP container opened successfully"
        if magic_kind == "gzip":
            gzip.decompress(data)
            return True, "GZIP decompressed successfully"
        if magic_kind == "xz":
            lzma.decompress(data, format=lzma.FORMAT_XZ)
            return True, "XZ decompressed successfully"
        if magic_kind == "bz2":
            bz2.decompress(data)
            return True, "BZ2 decompressed successfully"
        if magic_kind in IMAGE_KINDS and Image is not None:
            img = Image.open(io.BytesIO(data))
            img.verify()
            return True, "Image verified successfully"
        return True, "Magic signature accepted"
    except Exception as e:
        return False, str(e)

def blind_decode_dna(dna_text: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    dna = dna_codec.clean_dna_text(dna_text)
    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for mapping in MAPPING_OPTIONS:
        row: Dict[str, Any] = {"Mapping": mapping}
        try:
            data, bits, meta = decode_dna_with_mapping(dna, mapping)
            m = detect_magic(data)
            score = 0.0
            if data:
                score += 1.0
            if m:
                score += 10.0 * float(m.confidence)
                ok, note = validate_container(data, m.kind)
                if ok:
                    score += 5.0
                else:
                    score -= 2.0
                row.update({
                    "Status": "Valid" if ok else "Weak",
                    "Magic": m.kind,
                    "Ext": m.ext,
                    "Confidence": m.confidence,
                    "Bytes": len(data),
                    "Score": score,
                    "Note": note,
                })
            else:
                row.update({
                    "Status": "No magic",
                    "Magic": "—",
                    "Ext": "—",
                    "Confidence": 0.0,
                    "Bytes": len(data),
                    "Score": score,
                    "Note": "No recognizable file/container signature",
                })

            candidate = {
                "mapping": mapping,
                "data": data,
                "bits": bits,
                "meta": meta,
                "magic": m,
                "score": score,
                "row": row,
            }
            if m is not None and (best is None or candidate["score"] > best["score"]):
                best = candidate

        except Exception as e:
            row.update({
                "Status": "Failed",
                "Magic": "—",
                "Ext": "—",
                "Confidence": 0.0,
                "Bytes": 0,
                "Score": -1.0,
                "Note": str(e)[:160],
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    if best is None:
        raise ValueError("Auto-detection failed: no mapping produced a recognizable self-describing byte stream.")
    return best, df
