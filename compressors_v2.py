# compressors_v2.py
from __future__ import annotations

import gzip
import bz2
import lzma
import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from utils_bits_v2 import MagicInfo, detect_magic, safe_basename, write_bytes, zlib_wrap

# Unified candidate grids used by both Streamlit apps.
# Image lossy quality levels are intentionally standardized to 50/60/70/80/90
# so reports are comparable across apps and no backend silently introduces q55/q65.
# General-purpose compressors use a wider but still manageable level grid.


# ----------------------------
# ZIP helpers (Mode1/Mode2)
# ----------------------------

def zip_single_file(input_path: str, level: int = 6) -> Tuple[bytes, Dict[str, Any]]:
    """
    ZIP a single file (DEFLATED). This is self-describing (filename + CRC inside ZIP).
    """
    base = safe_basename(os.path.basename(input_path))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
        zf.write(input_path, arcname=base)
    meta = {"kind": "zip_single_file", "compression": "deflated", "level": level, "filename": base}
    return buf.getvalue(), meta


def zip_store_single_file(input_path: str) -> Tuple[bytes, Dict[str, Any]]:
    """
    ZIP a single file with STORED (no compression). Still self-describing (keeps extension).
    """
    base = safe_basename(os.path.basename(input_path))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(input_path, arcname=base)
    meta = {"kind": "zip_store_single_file", "compression": "stored", "filename": base}
    return buf.getvalue(), meta


def unzip_single_file(zip_bytes: bytes, out_dir: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract the first non-directory member from ZIP bytes.
    If there are multiple members, extracts the first file member.
    """
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        if not members:
            raise ValueError("ZIP has no file members")
        name = members[0]
        safe = safe_basename(name, fallback="unzipped.bin")
        dst = os.path.join(out_dir, safe)
        with zf.open(name, "r") as src, open(dst, "wb") as f:
            shutil.copyfileobj(src, f)
        return dst, {"kind": "unzip_single_file", "member": name, "out": dst}


# ----------------------------
# Domain detection
# ----------------------------

_XZ_MAGIC = b"\xFD7zXZ\x00"
_BZ2_MAGIC = b"BZh"


def _looks_text(b: bytes) -> bool:
    if not b:
        return False
    if b.count(b"\x00") > 0:
        return False
    try:
        s = b[:4096].decode("utf-8")
        printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in s)
        return (printable / max(1, len(s))) > 0.95
    except Exception:
        return False


def _is_xz_bytes(data: bytes) -> bool:
    return bool(data) and data.startswith(_XZ_MAGIC)


def _is_bz2_bytes(data: bytes) -> bool:
    return bool(data) and data.startswith(_BZ2_MAGIC)


def _rep_kind(data: bytes) -> Optional[str]:
    m = detect_magic(data)
    if m:
        return m.kind
    if _is_xz_bytes(data):
        return "xz"
    if _is_bz2_bytes(data):
        return "bz2"
    return None


def _rezip_zip_container(raw: bytes, level: int = 9) -> bytes:
    src = io.BytesIO(raw)
    out = io.BytesIO()
    with zipfile.ZipFile(src, 'r') as zin, zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=int(level)) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            zi = zipfile.ZipInfo(info.filename)
            zi.date_time = info.date_time
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.comment = info.comment
            zi.create_system = info.create_system
            zi.external_attr = info.external_attr
            zi.internal_attr = info.internal_attr
            zi.flag_bits = info.flag_bits
            zi.extra = info.extra
            zout.writestr(zi, data)
    return out.getvalue()


def _encode_bz2(raw: bytes, compresslevel: int = 9) -> bytes:
    return bz2.compress(raw, compresslevel=int(compresslevel))


def _decode_bz2(b: bytes) -> bytes:
    return bz2.decompress(b)


def _encode_xz(raw: bytes, preset: int = 6) -> bytes:
    return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=int(preset))


def _decode_xz(b: bytes) -> bytes:
    return lzma.decompress(b, format=lzma.FORMAT_XZ)


def detect_domain(input_path: str, raw_bytes: bytes) -> str:
    """
    Returns one of: image | audio | video | text | document | archive | binary | other
    """
    m = detect_magic(raw_bytes)
    if m:
        if m.kind in {"png", "jpeg", "webp", "gif", "bmp", "tiff"}:
            return "image"
        if m.kind in {"wav", "mp3", "flac", "opus_ogg", "ogg"}:
            return "audio"
        if m.kind in {"mp4", "avi", "mkv_webm"}:
            return "video"
        if m.kind in {"pdf", "docx", "pptx", "xlsx", "epub"}:
            return "document"
        if m.kind in {"zip", "gzip"}:
            return "archive"
        if m.kind == "text":
            return "text"

    if _is_xz_bytes(raw_bytes) or _is_bz2_bytes(raw_bytes):
        return "archive"

    ext = os.path.splitext(input_path)[1].lower()
    image_ext = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
    audio_ext = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}
    video_ext = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    text_ext = {".txt", ".md", ".json", ".csv", ".tsv", ".log", ".xml", ".yaml", ".yml", ".html", ".htm", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".ini"}
    document_ext = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".rtf", ".epub", ".odt", ".ods", ".odp"}
    archive_ext = {".zip", ".gz", ".tgz", ".tar", ".xz", ".bz2", ".7z", ".rar"}
    binary_ext = {".bin", ".dat", ".npy", ".npz", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".hdf5", ".mat"}

    if ext in image_ext:
        return "image"
    if ext in audio_ext:
        return "audio"
    if ext in video_ext:
        return "video"
    if ext in text_ext:
        return "text"
    if ext in document_ext:
        return "document"
    if ext in archive_ext:
        return "archive"
    if ext in binary_ext:
        return "binary"

    if _looks_text(raw_bytes):
        return "text"
    return "other"


# ----------------------------
# Representation encoding
# ----------------------------

@dataclass
class RepresentationResult:
    rep_bytes: bytes
    rep_meta: Dict[str, Any]


def _encode_gzip(raw: bytes, level: int = 9) -> bytes:
    return gzip.compress(raw, compresslevel=int(level))


def _decode_gzip(gz: bytes) -> bytes:
    return gzip.decompress(gz)


def _run_ffmpeg(args: list[str]) -> None:
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + p.stderr.decode("utf-8", errors="ignore")[:2000])


def _encode_image_webp(raw: bytes, quality: int, lossless: bool, allow_external_ffmpeg: bool) -> bytes:
    # Try PIL first
    try:
        from PIL import Image  # type: ignore
        img = Image.open(io.BytesIO(raw))
        out = io.BytesIO()
        img.save(out, format="WEBP", quality=int(quality), lossless=bool(lossless))
        return out.getvalue()
    except Exception:
        if not allow_external_ffmpeg:
            raise
    # Fallback: ffmpeg -> webp
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.bin")
        outp = os.path.join(td, "out.webp")
        with open(inp, "wb") as f:
            f.write(raw)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", inp]
        if lossless:
            cmd += ["-lossless", "1"]
        else:
            cmd += ["-q:v", str(max(0, min(100, int(quality))))]
        cmd += [outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_audio_opus(input_path: str, allow_external_ffmpeg: bool, bitrate_kbps: int = 64) -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode Opus.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.ogg")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:a", "libopus", "-b:a", f"{int(bitrate_kbps)}k", outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_audio_flac(input_path: str, allow_external_ffmpeg: bool) -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode FLAC.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.flac")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:a", "flac", outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_video_h264_mp4(input_path: str, allow_external_ffmpeg: bool, crf: int = 28, preset: str = "veryfast") -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode MP4.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.mp4")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
               "-pix_fmt", "yuv420p",
               "-movflags", "+faststart",
               "-c:a", "aac", "-b:a", "128k",
               outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_video_vp9_webm(input_path: str, allow_external_ffmpeg: bool, crf: int = 32, speed: int = 4) -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode WebM/VP9.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.webm")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:v", "libvpx-vp9", "-crf", str(int(crf)), "-b:v", "0", "-row-mt", "1",
               "-cpu-used", str(int(speed)),
               "-pix_fmt", "yuv420p",
               "-c:a", "libopus", "-b:a", "96k",
               outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_video_av1_mkv(input_path: str, allow_external_ffmpeg: bool, crf: int = 35, preset: int = 8) -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode AV1.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.mkv")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:v", "libaom-av1", "-crf", str(int(crf)), "-b:v", "0",
               "-cpu-used", str(int(preset)),
               "-pix_fmt", "yuv420p",
               "-c:a", "libopus", "-b:a", "96k",
               outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def domain_detect_and_encode_rep(
    input_path: str,
    raw_bytes: bytes,
    *,
    image_policy: str = "webp_lossy",
    webp_quality: int = 80,
    text_policy: str = "xz",
    allow_external_ffmpeg: bool = False,
    zlib_policy: str = "auto",
    audio_policy: str = "opus_ogg",
    opus_bitrate_kbps: int = 64,
    video_policy: str = "mp4_h264",
    video_crf: int = 28,
) -> RepresentationResult:
    """
    Produce self-describing bytes or restorable compressed bytes.
    Auto Fixed chooses one deterministic codec per domain.
    """
    dom = detect_domain(input_path, raw_bytes)
    m = detect_magic(raw_bytes)

    if dom == "image":
        if image_policy == "keep":
            return RepresentationResult(raw_bytes, {"domain": "image", "policy": "image_keep", "lossy": False})
        if image_policy == "png_lossless":
            try:
                from PIL import Image  # type: ignore
                img = Image.open(io.BytesIO(raw_bytes))
                out = io.BytesIO()
                img.save(out, format="PNG", optimize=True)
                return RepresentationResult(out.getvalue(), {"domain": "image", "policy": "png_lossless", "lossy": False})
            except Exception:
                return RepresentationResult(raw_bytes, {"domain": "image", "policy": "image_keep_fallback", "lossy": False})
        if image_policy == "webp_lossless":
            rep = _encode_image_webp(raw_bytes, quality=100, lossless=True, allow_external_ffmpeg=allow_external_ffmpeg)
            return RepresentationResult(rep, {"domain": "image", "policy": "webp_lossless", "lossy": False})
        if image_policy == "jpeg_lossy":
            rep = _encode_image_jpeg(raw_bytes, quality=webp_quality)
            return RepresentationResult(rep, {"domain": "image", "policy": f"jpeg_q{int(webp_quality)}", "lossy": True, "jpeg_quality": int(webp_quality)})
        rep = _encode_image_webp(raw_bytes, quality=webp_quality, lossless=False, allow_external_ffmpeg=allow_external_ffmpeg)
        return RepresentationResult(rep, {"domain": "image", "policy": f"webp_q{int(webp_quality)}", "lossy": True, "webp_quality": int(webp_quality)})

    if dom == "text":
        if text_policy == "keep":
            return RepresentationResult(raw_bytes, {"domain": "text", "policy": "text_keep", "lossy": False})
        if text_policy == "gzip":
            rep = _encode_gzip(raw_bytes, level=9)
            return RepresentationResult(rep, {"domain": "text", "policy": "text_gzip9", "lossy": False})
        if text_policy == "bz2":
            rep = _encode_bz2(raw_bytes, compresslevel=9)
            return RepresentationResult(rep, {"domain": "text", "policy": "text_bz2_9", "lossy": False})
        rep = _encode_xz(raw_bytes, preset=6)
        return RepresentationResult(rep, {"domain": "text", "policy": "text_xz6", "lossy": False})

    if dom == "document":
        if m and m.kind in {"docx", "pptx", "xlsx", "epub"}:
            try:
                rep = _rezip_zip_container(raw_bytes, level=9)
                return RepresentationResult(rep, {"domain": "document", "policy": "ooxml_rezip9", "lossy": False, "source_kind": m.kind})
            except Exception:
                pass
        rep = _encode_xz(raw_bytes, preset=6)
        return RepresentationResult(rep, {"domain": "document", "policy": (f"{m.kind}_xz6" if m and m.kind else "document_xz6"), "lossy": False, "source_kind": (m.kind if m else None)})

    if dom == "archive":
        if m and m.kind in {"zip", "gzip"} or _is_xz_bytes(raw_bytes) or _is_bz2_bytes(raw_bytes):
            return RepresentationResult(raw_bytes, {"domain": "archive", "policy": "archive_keep", "lossy": False})
        rep = _encode_xz(raw_bytes, preset=6)
        return RepresentationResult(rep, {"domain": "archive", "policy": "archive_xz6", "lossy": False})

    if dom == "binary":
        rep = _encode_xz(raw_bytes, preset=6)
        return RepresentationResult(rep, {"domain": "binary", "policy": "binary_xz6", "lossy": False})

    if dom == "audio":
        if audio_policy == "keep" or not allow_external_ffmpeg:
            return RepresentationResult(raw_bytes, {"domain": "audio", "policy": "audio_keep", "lossy": False, "note": "ffmpeg_disabled_or_keep"})
        if audio_policy == "flac_lossless":
            rep = _encode_audio_flac(input_path, allow_external_ffmpeg=True)
            return RepresentationResult(rep, {"domain": "audio", "policy": "flac_lossless", "lossy": False})
        if audio_policy == "mp3":
            rep = _encode_audio_mp3(input_path, allow_external_ffmpeg=True, bitrate_kbps=128)
            return RepresentationResult(rep, {"domain": "audio", "policy": "mp3_128", "lossy": True, "mp3_bitrate_kbps": 128})
        if audio_policy == "aac_m4a":
            rep = _encode_audio_aac_m4a(input_path, allow_external_ffmpeg=True, bitrate_kbps=128)
            return RepresentationResult(rep, {"domain": "audio", "policy": "aac_128", "lossy": True, "aac_bitrate_kbps": 128})
        rep = _encode_audio_opus(input_path, allow_external_ffmpeg=True, bitrate_kbps=opus_bitrate_kbps)
        return RepresentationResult(rep, {"domain": "audio", "policy": f"opus_{int(opus_bitrate_kbps)}", "lossy": True, "opus_bitrate_kbps": int(opus_bitrate_kbps)})

    if dom == "video":
        if video_policy == "keep" or not allow_external_ffmpeg:
            return RepresentationResult(raw_bytes, {"domain": "video", "policy": "video_keep", "lossy": False, "note": "ffmpeg_disabled_or_keep"})
        if video_policy == "webm_vp9":
            rep = _encode_video_vp9_webm(input_path, allow_external_ffmpeg=True, crf=video_crf)
            return RepresentationResult(rep, {"domain": "video", "policy": f"vp9_crf{int(video_crf)}", "lossy": True, "crf": int(video_crf)})
        if video_policy == "mkv_av1":
            rep = _encode_video_av1_mkv(input_path, allow_external_ffmpeg=True, crf=video_crf)
            return RepresentationResult(rep, {"domain": "video", "policy": f"av1_crf{int(video_crf)}", "lossy": True, "crf": int(video_crf)})
        rep = _encode_video_h264_mp4(input_path, allow_external_ffmpeg=True, crf=video_crf)
        return RepresentationResult(rep, {"domain": "video", "policy": f"h264_crf{int(video_crf)}", "lossy": True, "crf": int(video_crf)})

    rep = _encode_xz(raw_bytes, preset=6)
    return RepresentationResult(rep, {"domain": dom, "policy": f"{dom}_xz6", "lossy": False})


# ----------------------------
# Restore representation (headerless)
# ----------------------------

# ----------------------------
# Benchmark candidate generation (Mode 3 Best)
# ----------------------------


def _encode_image_png(raw: bytes) -> bytes:
    from PIL import Image  # type: ignore
    img = Image.open(io.BytesIO(raw))
    out = io.BytesIO()
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA")
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _encode_image_jpeg(raw: bytes, quality: int) -> bytes:
    from PIL import Image  # type: ignore
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=int(quality), optimize=True)
    return out.getvalue()


def _encode_audio_mp3(input_path: str, *, allow_external_ffmpeg: bool, bitrate_kbps: int) -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode MP3.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.mp3")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:a", "libmp3lame", "-b:a", f"{int(bitrate_kbps)}k", outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_audio_aac_m4a(input_path: str, *, allow_external_ffmpeg: bool, bitrate_kbps: int) -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode AAC/M4A.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.m4a")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:a", "aac", "-b:a", f"{int(bitrate_kbps)}k", outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def benchmark_domain_encode_rep(
    input_path: str,
    raw_bytes: bytes,
    *,
    quality_mode: str = "Lossy",
    allow_external_ffmpeg: bool = False,
    zlib_policy: str = "auto",
    image_webp_qualities: Tuple[int, ...] = (50, 60, 70, 80, 90),
    image_jpeg_qualities: Tuple[int, ...] = (50, 60, 70, 80, 90),
    opus_bitrates_kbps: Tuple[int, ...] = (24, 32, 48, 64, 96, 128),
    mp3_bitrates_kbps: Tuple[int, ...] = (96, 128, 160, 192, 256),
    aac_bitrates_kbps: Tuple[int, ...] = (96, 128, 160, 192),
    video_crfs: Tuple[int, ...] = (18, 23, 28, 32),
    video_vp9_crfs: Tuple[int, ...] = (28, 32, 36, 40),
    video_av1_crfs: Tuple[int, ...] = (30, 35, 40, 45),
) -> Tuple[RepresentationResult, Dict[str, Any]]:
    dom = detect_domain(input_path, raw_bytes)
    m = detect_magic(raw_bytes)
    qmode = str(quality_mode or "Lossy")
    qmode_norm = "Lossless" if qmode.lower().startswith("lossless") else "Lossy"

    candidates: list[Tuple[str, bytes, Dict[str, Any]]] = []

    def _try_add(name: str, fn, meta: Dict[str, Any]):
        try:
            b = fn()
            if not isinstance(b, (bytes, bytearray)) or len(b) == 0:
                return
            candidates.append((name, bytes(b), dict(meta or {})))
        except Exception:
            return

    keep_policy = f"{dom}_keep"
    if dom == "document" and m and m.kind:
        keep_policy = f"{m.kind}_keep"
    candidates.append(("keep", raw_bytes, {"domain": dom, "policy": keep_policy, "lossy": False, "kind": _rep_kind(raw_bytes)}))

    if dom == "text":
        for lvl in (1, 3, 6, 9):
            _try_add(f"gzip_lvl{lvl}", lambda lvl=lvl: _encode_gzip(raw_bytes, level=lvl), {"domain": dom, "policy": f"text_gzip{int(lvl)}", "lossy": False})
        for lvl in (1, 3, 5, 7, 9):
            _try_add(f"bz2_lvl{lvl}", lambda lvl=lvl: _encode_bz2(raw_bytes, compresslevel=lvl), {"domain": dom, "policy": f"text_bz2_{int(lvl)}", "lossy": False})
        for preset in (0, 3, 6, 9):
            _try_add(f"xz_p{preset}", lambda preset=preset: _encode_xz(raw_bytes, preset=preset), {"domain": dom, "policy": f"text_xz{int(preset)}", "lossy": False})
        for lvl in (1, 3, 6, 9):
            _try_add(f"zip_lvl{lvl}", lambda lvl=lvl: zip_single_file(input_path, level=lvl)[0], {"domain": dom, "policy": f"text_zip{int(lvl)}", "lossy": False})

    elif dom == "document":
        kind = m.kind if m else None
        is_pdf = bool(kind == "pdf")
        is_ooxml = bool(kind in {"docx", "pptx", "xlsx", "epub"})
        for lvl in (1, 3, 6, 9):
            _try_add(f"gzip_lvl{lvl}", lambda lvl=lvl: _encode_gzip(raw_bytes, level=lvl), {"domain": dom, "policy": (f"pdf_gzip{int(lvl)}" if is_pdf else f"document_gzip{int(lvl)}"), "lossy": False, "source_kind": kind})
        for lvl in (1, 3, 5, 7, 9):
            _try_add(f"bz2_lvl{lvl}", lambda lvl=lvl: _encode_bz2(raw_bytes, compresslevel=lvl), {"domain": dom, "policy": (f"pdf_bz2_{int(lvl)}" if is_pdf else f"document_bz2_{int(lvl)}"), "lossy": False, "source_kind": kind})
        for preset in (0, 3, 6, 9):
            _try_add(f"xz_p{preset}", lambda preset=preset: _encode_xz(raw_bytes, preset=preset), {"domain": dom, "policy": (f"pdf_xz{int(preset)}" if is_pdf else f"document_xz{int(preset)}"), "lossy": False, "source_kind": kind})
        if is_ooxml:
            for lvl in (1, 3, 6, 9):
                _try_add(f"rezip_lvl{lvl}", lambda lvl=lvl: _rezip_zip_container(raw_bytes, level=lvl), {"domain": dom, "policy": f"ooxml_rezip{int(lvl)}", "lossy": False, "source_kind": kind})
        for lvl in (1, 3, 6, 9):
            _try_add(f"zip_lvl{lvl}", lambda lvl=lvl: zip_single_file(input_path, level=lvl)[0], {"domain": dom, "policy": (f"pdf_zip{int(lvl)}" if is_pdf else f"document_zip{int(lvl)}"), "lossy": False, "source_kind": kind})

    elif dom == "archive":
        for lvl in (1, 3, 6, 9):
            _try_add(f"gzip_lvl{lvl}", lambda lvl=lvl: _encode_gzip(raw_bytes, level=lvl), {"domain": dom, "policy": f"archive_gzip{int(lvl)}", "lossy": False})
        for lvl in (1, 3, 5, 7, 9):
            _try_add(f"bz2_lvl{lvl}", lambda lvl=lvl: _encode_bz2(raw_bytes, compresslevel=lvl), {"domain": dom, "policy": f"archive_bz2_{int(lvl)}", "lossy": False})
        for preset in (0, 3, 6, 9):
            _try_add(f"xz_p{preset}", lambda preset=preset: _encode_xz(raw_bytes, preset=preset), {"domain": dom, "policy": f"archive_xz{int(preset)}", "lossy": False})
        for lvl in (1, 3, 6, 9):
            _try_add(f"zip_lvl{lvl}", lambda lvl=lvl: zip_single_file(input_path, level=lvl)[0], {"domain": dom, "policy": f"archive_zip{int(lvl)}", "lossy": False})

    elif dom == "binary":
        for lvl in (1, 3, 6, 9):
            _try_add(f"gzip_lvl{lvl}", lambda lvl=lvl: _encode_gzip(raw_bytes, level=lvl), {"domain": dom, "policy": f"binary_gzip{int(lvl)}", "lossy": False})
        for lvl in (1, 3, 5, 7, 9):
            _try_add(f"bz2_lvl{lvl}", lambda lvl=lvl: _encode_bz2(raw_bytes, compresslevel=lvl), {"domain": dom, "policy": f"binary_bz2_{int(lvl)}", "lossy": False})
        for preset in (0, 3, 6, 9):
            _try_add(f"xz_p{preset}", lambda preset=preset: _encode_xz(raw_bytes, preset=preset), {"domain": dom, "policy": f"binary_xz{int(preset)}", "lossy": False})
        for lvl in (1, 3, 6, 9):
            _try_add(f"zip_lvl{lvl}", lambda lvl=lvl: zip_single_file(input_path, level=lvl)[0], {"domain": dom, "policy": f"binary_zip{int(lvl)}", "lossy": False})

    elif dom == "image":
        _try_add("png_lossless", lambda: _encode_image_png(raw_bytes), {"domain": dom, "policy": "png_lossless", "lossy": False})
        _try_add("webp_lossless", lambda: _encode_image_webp(raw_bytes, quality=100, lossless=True, allow_external_ffmpeg=allow_external_ffmpeg), {"domain": dom, "policy": "webp_lossless", "lossy": False})
        if qmode_norm == "Lossy":
            for q in image_webp_qualities:
                _try_add(f"webp_q{int(q)}", lambda q=q: _encode_image_webp(raw_bytes, quality=int(q), lossless=False, allow_external_ffmpeg=allow_external_ffmpeg), {"domain": dom, "policy": f"webp_q{int(q)}", "lossy": True, "webp_quality": int(q)})
            for q in image_jpeg_qualities:
                _try_add(f"jpeg_q{int(q)}", lambda q=q: _encode_image_jpeg(raw_bytes, quality=int(q)), {"domain": dom, "policy": f"jpeg_q{int(q)}", "lossy": True, "jpeg_quality": int(q)})

    elif dom == "audio":
        _try_add("flac_lossless", lambda: _encode_audio_flac(input_path, allow_external_ffmpeg=allow_external_ffmpeg), {"domain": dom, "policy": "flac_lossless", "lossy": False})
        if qmode_norm == "Lossy":
            for br in opus_bitrates_kbps:
                _try_add(f"opus_{int(br)}k", lambda br=br: _encode_audio_opus(input_path, allow_external_ffmpeg=allow_external_ffmpeg, bitrate_kbps=int(br)), {"domain": dom, "policy": f"opus_{int(br)}", "lossy": True, "opus_bitrate_kbps": int(br)})
            for br in mp3_bitrates_kbps:
                _try_add(f"mp3_{int(br)}k", lambda br=br: _encode_audio_mp3(input_path, allow_external_ffmpeg=allow_external_ffmpeg, bitrate_kbps=int(br)), {"domain": dom, "policy": f"mp3_{int(br)}", "lossy": True, "mp3_bitrate_kbps": int(br)})
            for br in aac_bitrates_kbps:
                _try_add(f"aac_{int(br)}k", lambda br=br: _encode_audio_aac_m4a(input_path, allow_external_ffmpeg=allow_external_ffmpeg, bitrate_kbps=int(br)), {"domain": dom, "policy": f"aac_{int(br)}", "lossy": True, "aac_bitrate_kbps": int(br)})

    elif dom == "video":
        if qmode_norm == "Lossy":
            for crf in video_crfs:
                _try_add(f"h264_crf{int(crf)}", lambda crf=crf: _encode_video_h264_mp4(input_path, allow_external_ffmpeg=allow_external_ffmpeg, crf=int(crf)), {"domain": dom, "policy": f"h264_crf{int(crf)}", "lossy": True, "crf": int(crf)})
            for crf in video_vp9_crfs:
                _try_add(f"vp9_crf{int(crf)}", lambda crf=crf: _encode_video_vp9_webm(input_path, allow_external_ffmpeg=allow_external_ffmpeg, crf=int(crf)), {"domain": dom, "policy": f"vp9_crf{int(crf)}", "lossy": True, "crf": int(crf)})
            for crf in video_av1_crfs:
                _try_add(f"av1_crf{int(crf)}", lambda crf=crf: _encode_video_av1_mkv(input_path, allow_external_ffmpeg=allow_external_ffmpeg, crf=int(crf)), {"domain": dom, "policy": f"av1_crf{int(crf)}", "lossy": True, "crf": int(crf)})

    else:
        for lvl in (1, 3, 6, 9):
            _try_add(f"gzip_lvl{lvl}", lambda lvl=lvl: _encode_gzip(raw_bytes, level=lvl), {"domain": dom, "policy": f"{dom}_gzip{int(lvl)}", "lossy": False})
        for lvl in (1, 3, 5, 7, 9):
            _try_add(f"bz2_lvl{lvl}", lambda lvl=lvl: _encode_bz2(raw_bytes, compresslevel=lvl), {"domain": dom, "policy": f"{dom}_bz2_{int(lvl)}", "lossy": False})
        for preset in (0, 3, 6, 9):
            _try_add(f"xz_p{preset}", lambda preset=preset: _encode_xz(raw_bytes, preset=preset), {"domain": dom, "policy": f"{dom}_xz{int(preset)}", "lossy": False})
        for lvl in (1, 3, 6, 9):
            _try_add(f"zip_lvl{lvl}", lambda lvl=lvl: zip_single_file(input_path, level=lvl)[0], {"domain": dom, "policy": f"{dom}_zip{int(lvl)}", "lossy": False})

    scored: list[Tuple[int, int, str, bytes, Dict[str, Any]]] = []
    for (name, rep_bytes, meta) in candidates:
        try:
            z, _zmeta = zlib_wrap(rep_bytes, policy=str(zlib_policy or "auto"))
            scored.append((len(z), len(rep_bytes), name, rep_bytes, meta))
        except Exception:
            continue
    if not scored:
        raise RuntimeError(f"No candidates could be framed by zlib (policy={zlib_policy}).")

    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    best_zlib_size, best_rep_size, best_name, best_bytes, best_meta = scored[0]

    candidates_sorted = []
    for (zsz, rsz, name, rep_bytes, meta) in scored:
        row = {
            "name": name,
            "rep_size_bytes": int(rsz),
            "zlib_size_bytes": int(zsz),
            **(meta or {}),
        }
        candidates_sorted.append(row)

    bench_meta = {
        "detected_domain": dom,
        "quality_mode": qmode_norm,
        "chosen_candidate": best_name,
        "candidates": candidates_sorted,
    }

    best_meta = dict(best_meta or {})
    best_meta.setdefault("domain", dom)
    best_meta.setdefault("lossy", bool(best_meta.get("lossy", False)))
    best_meta["chosen_candidate"] = best_name
    best_meta["quality_mode"] = qmode_norm
    best_meta["candidates"] = candidates_sorted

    return RepresentationResult(best_bytes, best_meta), bench_meta
# ----------------------------

def _write_with_magic(data: bytes, out_dir: str, stem: str) -> Tuple[str, Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    m = detect_magic(data)
    ext = m.ext if m else ".bin"
    out_path = os.path.join(out_dir, safe_basename(stem + ext, fallback=stem + ".bin"))
    write_bytes(out_path, data)
    return out_path, {"detected_magic": (m.kind if m else None), "ext": ext, "restore_kind": "write_bytes"}


def restore_rep(inner_bytes: bytes, out_dir: str, preferred_stem: str = "restored") -> Tuple[str, Dict[str, Any]]:
    """
    Convert inner bytes -> a concrete file on disk, using standard magic signatures
    plus native handling for XZ/BZ2 representations.
    """
    m = detect_magic(inner_bytes)

    if _is_xz_bytes(inner_bytes):
        payload = _decode_xz(inner_bytes)
        outp, meta = restore_rep(payload, out_dir=out_dir, preferred_stem=preferred_stem)
        meta.update({"input_magic": "xz", "restore_kind": "unxz_then_restore"})
        return outp, meta

    if _is_bz2_bytes(inner_bytes):
        payload = _decode_bz2(inner_bytes)
        outp, meta = restore_rep(payload, out_dir=out_dir, preferred_stem=preferred_stem)
        meta.update({"input_magic": "bz2", "restore_kind": "bunzip2_then_restore"})
        return outp, meta

    if m and m.kind == "gzip":
        payload = _decode_gzip(inner_bytes)
        outp, meta = restore_rep(payload, out_dir=out_dir, preferred_stem=preferred_stem)
        meta.update({"input_magic": "gzip", "restore_kind": "gunzip_then_restore"})
        return outp, meta

    if m and m.kind in {"docx", "pptx", "xlsx", "epub"}:
        outp, meta = _write_with_magic(inner_bytes, out_dir, preferred_stem)
        meta.update({"restore_kind": "write_container"})
        return outp, meta

    if m and m.kind == "zip":
        try:
            with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as zf:
                members = [n for n in zf.namelist() if not n.endswith("/")]
                if len(members) == 1:
                    extracted, meta = unzip_single_file(inner_bytes, out_dir=out_dir)
                    meta.update({"detected_magic": "zip", "restore_kind": "unzip_single_file"})
                    return extracted, meta
        except Exception:
            pass
        outp, meta = _write_with_magic(inner_bytes, out_dir, preferred_stem)
        meta.update({"restore_kind": "write_zip_bytes"})
        return outp, meta

    return _write_with_magic(inner_bytes, out_dir, preferred_stem)
