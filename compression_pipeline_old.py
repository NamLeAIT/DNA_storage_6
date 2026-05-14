from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import lzma
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from PIL import Image
except Exception:
    Image = None

from utils_bits_v2 import detect_magic, safe_basename
from config import SELF_DESCRIBING_KINDS
from ui_helpers import fmt_bytes, get_domain


@dataclass
class CompressionCandidate:
    rank: int
    method: str
    data: bytes
    ext: str
    kind: str
    mime: str
    lossy: bool
    size_bytes: int
    compression_ratio: float
    saving_pct: float
    estimated_dna_nt: int
    note: str = ""

    def public_row(self) -> Dict[str, Any]:
        return {
            "Rank": self.rank,
            "Method": self.method,
            "Type": self.kind,
            "Ext": self.ext,
            "Lossy": "Yes" if self.lossy else "No",
            "Size": fmt_bytes(self.size_bytes),
            "Ratio": f"{self.compression_ratio:.2f}×",
            "Saving": f"{self.saving_pct:.2f}%",
            "Est. DNA nt": f"{self.estimated_dna_nt:,}",
            "Note": self.note,
        }

def _candidate_from_bytes(method: str, data: bytes, original_size: int, lossy: bool, note: str = "") -> Optional[CompressionCandidate]:
    if not data:
        return None
    m = detect_magic(data)
    if not m:
        return None
    if m.kind not in SELF_DESCRIBING_KINDS:
        return None
    size = len(data)
    ratio = (original_size / size) if size else 0.0
    saving = (1.0 - (size / original_size)) * 100.0 if original_size else 0.0
    return CompressionCandidate(
        rank=0,
        method=method,
        data=data,
        ext=m.ext,
        kind=m.kind,
        mime=m.mime,
        lossy=lossy,
        size_bytes=size,
        compression_ratio=ratio,
        saving_pct=saving,
        estimated_dna_nt=size * 4,  # exact for Simple Mapping, estimate for rule-based mappings
        note=note or getattr(m, "note", ""),
    )

def zip_store_bytes(name: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(safe_basename(name), data)
    return buf.getvalue()

def zip_deflate_bytes(name: str, data: bytes, level: int = 6) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=int(level)) as zf:
        zf.writestr(safe_basename(name), data)
    return buf.getvalue()

def image_candidates(raw: bytes, original_name: str, original_size: int, quality_mode: str = "Compression") -> List[CompressionCandidate]:
    out: List[CompressionCandidate] = []
    if Image is None:
        return out

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return out

    # PNG lossless
    try:
        im = img
        if im.mode not in ("RGB", "RGBA", "L", "P"):
            im = im.convert("RGBA")
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        c = _candidate_from_bytes("png_lossless", buf.getvalue(), original_size, lossy=False)
        if c:
            out.append(c)
    except Exception:
        pass

    # WebP lossless
    try:
        buf = io.BytesIO()
        img.save(buf, format="WEBP", lossless=True, quality=100)
        c = _candidate_from_bytes("webp_lossless", buf.getvalue(), original_size, lossy=False)
        if c:
            out.append(c)
    except Exception:
        pass

    # WebP / JPEG lossy candidates
    for q in (50, 60, 70, 80, 90):
        # WebP lossy
        try:
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=int(q), lossless=False)
            c = _candidate_from_bytes(f"webp_q{q}", buf.getvalue(), original_size, lossy=True)
            if c:
                out.append(c)
        except Exception:
            pass

        # JPEG lossy
        try:
            im = img
            if im.mode != "RGB":
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=int(q), optimize=True)
            c = _candidate_from_bytes(f"jpeg_q{q}", buf.getvalue(), original_size, lossy=True)
            if c:
                out.append(c)
        except Exception:
            pass

    return out

def has_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        p = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        return p.returncode == 0
    except Exception:
        return False


def run_ffmpeg_transcode(input_path: str, output_ext: str, args: List[str]) -> Optional[bytes]:
    """Run ffmpeg and return output bytes. Returns None if ffmpeg or the requested codec fails."""
    try:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / f"out{output_ext}"
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
            ] + args + [str(out_path)]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            if p.returncode != 0 or not out_path.exists():
                return None
            return out_path.read_bytes()
    except Exception:
        return None


def audio_compression_candidates(input_path: str, raw: bytes, original_size: int, quality_mode: str) -> List[CompressionCandidate]:
    """
    Audio-native candidates for WAV/uncompressed audio.

    Lossless mode:
      - FLAC

    Lossy mode:
      - OGG/Opus
      - MP3
      - AAC/M4A

    Requires ffmpeg on PATH. If ffmpeg is not installed, returns [] and the benchmark
    falls back to generic containers such as gzip/xz/zip.
    """
    out: List[CompressionCandidate] = []
    if not has_ffmpeg():
        return out

    # Lossless audio-native candidate
    flac = run_ffmpeg_transcode(input_path, ".flac", ["-vn", "-c:a", "flac"])
    if flac:
        c = _candidate_from_bytes("flac_lossless", flac, original_size, lossy=False, note="audio-native lossless")
        if c:
            out.append(c)

    if quality_mode in {"Lossy", "Lossy allowed"}:
        # OGG/Opus candidates. Usually very efficient for WAV/audio.
        for br in (32, 64, 96, 128):
            ogg = run_ffmpeg_transcode(
                input_path,
                ".ogg",
                ["-vn", "-c:a", "libopus", "-b:a", f"{br}k"],
            )
            if ogg:
                c = _candidate_from_bytes(f"opus_ogg_{br}k", ogg, original_size, lossy=True, note="audio-native lossy")
                if c:
                    out.append(c)

        # MP3 fallback
        for br in (96, 128, 192):
            mp3 = run_ffmpeg_transcode(
                input_path,
                ".mp3",
                ["-vn", "-c:a", "libmp3lame", "-b:a", f"{br}k"],
            )
            if mp3:
                c = _candidate_from_bytes(f"mp3_{br}k", mp3, original_size, lossy=True, note="audio-native lossy")
                if c:
                    out.append(c)

        # AAC in M4A container. detect_magic identifies it as mp4/ftyp.
        for br in (96, 128):
            aac = run_ffmpeg_transcode(
                input_path,
                ".m4a",
                ["-vn", "-c:a", "aac", "-b:a", f"{br}k"],
            )
            if aac:
                c = _candidate_from_bytes(f"aac_m4a_{br}k", aac, original_size, lossy=True, note="audio-native lossy")
                if c:
                    out.append(c)

    return out


def video_compression_candidates(input_path: str, raw: bytes, original_size: int, quality_mode: str) -> List[CompressionCandidate]:
    """Small video-native benchmark grid. Requires ffmpeg."""
    out: List[CompressionCandidate] = []
    if quality_mode not in {"Lossy", "Lossy allowed"} or not has_ffmpeg():
        return out

    for crf in (28, 32):
        mp4 = run_ffmpeg_transcode(
            input_path,
            ".mp4",
            [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-b:a", "128k",
            ],
        )
        if mp4:
            c = _candidate_from_bytes(f"h264_mp4_crf{crf}", mp4, original_size, lossy=True, note="video-native lossy")
            if c:
                out.append(c)

    webm = run_ffmpeg_transcode(
        input_path,
        ".webm",
        [
            "-c:v", "libvpx-vp9",
            "-crf", "32",
            "-b:v", "0",
            "-row-mt", "1",
            "-cpu-used", "4",
            "-pix_fmt", "yuv420p",
            "-c:a", "libopus",
            "-b:a", "96k",
        ],
    )
    if webm:
        c = _candidate_from_bytes("vp9_webm_crf32", webm, original_size, lossy=True, note="video-native lossy")
        if c:
            out.append(c)

    return out



def generic_container_candidates(raw: bytes, original_name: str, original_size: int) -> List[CompressionCandidate]:
    out: List[CompressionCandidate] = []

    # gzip
    for lvl in (1, 6, 9):
        try:
            b = gzip.compress(raw, compresslevel=lvl)
            c = _candidate_from_bytes(f"gzip_lvl{lvl}", b, original_size, lossy=False)
            if c:
                out.append(c)
        except Exception:
            pass

    # bz2
    for lvl in (1, 6, 9):
        try:
            b = bz2.compress(raw, compresslevel=lvl)
            c = _candidate_from_bytes(f"bz2_lvl{lvl}", b, original_size, lossy=False)
            if c:
                out.append(c)
        except Exception:
            pass

    # xz
    for preset in (0, 6, 9):
        try:
            b = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=preset)
            c = _candidate_from_bytes(f"xz_p{preset}", b, original_size, lossy=False)
            if c:
                out.append(c)
        except Exception:
            pass

    # zip store / deflate
    try:
        b = zip_store_bytes(original_name, raw)
        c = _candidate_from_bytes("zip_store", b, original_size, lossy=False)
        if c:
            out.append(c)
    except Exception:
        pass

    for lvl in (1, 6, 9):
        try:
            b = zip_deflate_bytes(original_name, raw, level=lvl)
            c = _candidate_from_bytes(f"zip_deflate_lvl{lvl}", b, original_size, lossy=False)
            if c:
                out.append(c)
        except Exception:
            pass

    return out

def run_compression_benchmark(input_path: str, raw: bytes, quality_mode: str = "Lossy allowed") -> Tuple[CompressionCandidate, List[CompressionCandidate]]:
    original_name = os.path.basename(input_path) or "input.bin"
    original_size = len(raw)
    candidates: List[CompressionCandidate] = []

    # Keep original if it is self-describing or text-like.
    keep = _candidate_from_bytes("keep_original", raw, original_size, lossy=False)
    if keep:
        candidates.append(keep)

    domain = get_domain(input_path, raw)

    if domain == "image":
        candidates.extend(image_candidates(raw, original_name, original_size, quality_mode=quality_mode))
        # Also allow generic containers; useful for small or unusual images.
        candidates.extend(generic_container_candidates(raw, original_name, original_size))
    elif domain == "audio":
        # Prefer audio-native codecs for WAV/uncompressed audio.
        candidates.extend(audio_compression_candidates(input_path, raw, original_size, quality_mode=quality_mode))
        # Include generic containers only for comparison/fallback.
        candidates.extend(generic_container_candidates(raw, original_name, original_size))
    elif domain == "video":
        candidates.extend(video_compression_candidates(input_path, raw, original_size, quality_mode=quality_mode))
        candidates.extend(generic_container_candidates(raw, original_name, original_size))
    elif domain in {"text", "other", "unknown", "binary"}:
        candidates.extend(generic_container_candidates(raw, original_name, original_size))
    elif domain in {"archive", "document"}:
        # Native keep usually wins for already-compressed containers, but include container tests for comparison.
        candidates.extend(generic_container_candidates(raw, original_name, original_size))
    else:
        candidates.extend(generic_container_candidates(raw, original_name, original_size))

    # Deduplicate by (method, size, sha)
    seen = set()
    unique: List[CompressionCandidate] = []
    for c in candidates:
        key = (c.method, c.size_bytes, hashlib.sha1(c.data[:4096]).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    if not unique:
        # Last-resort ZIP store to force self-describing container
        b = zip_store_bytes(original_name, raw)
        c = _candidate_from_bytes("zip_store_fallback", b, original_size, lossy=False)
        if not c:
            raise RuntimeError("Could not create any self-describing compression candidate.")
        unique = [c]

    unique.sort(key=lambda x: (x.size_bytes, x.method))
    for i, c in enumerate(unique, start=1):
        c.rank = i
    return unique[0], unique
