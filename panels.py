from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

try:
    from PIL import Image
except Exception:
    Image = None

import dna_codec

from utils_bits_v2 import detect_magic, sha256_bytes
from config import (
    WORK_ROOT,
    IMAGE_KINDS,
    MAPPING_OPTIONS,
    DNA_PREVIEW_HEIGHT,
    FRAGMENT_PREVIEW_HEIGHT,
)
from ui_helpers import (
    fmt_bytes,
    step_header,
    save_upload,
    magic_dict,
    get_domain,
    preview_file,
    download_bytes_button,
)
from compression_pipeline import (
    CompressionCandidate,
    _candidate_from_bytes,
    run_compression_benchmark,
)
from dna_mapping import encode_bytes_to_dna, blind_decode_dna
from fragments import (
    clean_dna,
    prepare_dna_strands,
    strand_rows_to_csv,
    strand_rows_to_preview,
)
from restore_analysis import write_restored_file, image_metrics, text_similarity


def render_panel_1_upload() -> None:
    with st.container(border=True):
        step_header(1, "Upload Data")
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("### Upload")
            uploaded = st.file_uploader("Upload file", type=None, key="upload_input_file")
            if uploaded is not None:
                path, data = save_upload(uploaded)
                st.session_state["input_path"] = path
                st.session_state["input_bytes"] = data
                st.session_state["input_name"] = os.path.basename(path)
                #st.success("File loaded.")

        with right:
            st.markdown("### Detected information and preview")
            path = st.session_state.get("input_path")
            data = st.session_state.get("input_bytes")
            if path and data:
                m = magic_dict(data)
                domain = get_domain(path, data)
                c1, c2, c3 = st.columns(3)
                c1.metric("Data type", domain)
                c2.metric("File extension", m["kind"])
                c3.metric("File size", fmt_bytes(len(data)))
                #c1, c2, c3 = st.columns(3)
                #c1.metric("Extension", m["ext"])
                #c2.metric("MIME", m["mime"][:24] + ("..." if len(m["mime"]) > 24 else ""))
                #c3.metric("Confidence", f"{float(m['confidence']):.2f}")

                if Image is not None and domain == "image":
                    try:
                        img = Image.open(io.BytesIO(data))
                        st.caption(f"Image size: {img.size[0]} × {img.size[1]} px ")
                    except Exception:
                        pass

                preview_file(path, "Input preview")
            else:
                st.info("Upload a file to start.")

def render_panel_2_compression() -> None:
    with st.container(border=True):
        step_header(2, "Compression")
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("### Mode:")
            mode = st.radio(
                " ",
                ["Non-compression", "Compression"],
                horizontal=True,
                key="storage_mode",
            )
            st.markdown(
                """
""",
                unsafe_allow_html=True,
            )
            quality_mode = "Lossy"
            run = st.button("Run", type="primary", use_container_width=True, key="btn_run_compression")

            if run:
                path = st.session_state.get("input_path")
                raw = st.session_state.get("input_bytes")
                if not path or raw is None:
                    st.error("Upload a file first.")
                else:
                    run_dir = WORK_ROOT / "compression" / uuid.uuid4().hex
                    run_dir.mkdir(parents=True, exist_ok=True)

                    if mode == "Non-compression":
                        stored_bytes = raw
                        m = detect_magic(raw)
                        ext = m.ext if m else Path(path).suffix or ".bin"
                        out_name = "noncompressed_input" + ext
                        out_path = run_dir / out_name
                        out_path.write_bytes(stored_bytes)
                        cand = _candidate_from_bytes("non_compression_original_bytes", stored_bytes, len(raw), lossy=False)
                        if cand is None:
                            # still allow non-compression, even if no magic; decoding may not self-identify.
                            cand = CompressionCandidate(
                                rank=1,
                                method="non_compression_original_bytes",
                                data=stored_bytes,
                                ext=ext,
                                kind=m.kind if m else "unknown",
                                mime=m.mime if m else "application/octet-stream",
                                lossy=False,
                                size_bytes=len(stored_bytes),
                                compression_ratio=1.0,
                                saving_pct=0.0,
                                estimated_dna_nt=len(stored_bytes) * 4,
                                note="No magic detected; exact byte recovery still possible if mapping is known.",
                            )
                        st.session_state["compression_candidates"] = [cand]
                        st.session_state["selected_candidate"] = cand
                        st.session_state["stored_bytes"] = stored_bytes
                        st.session_state["stored_file_path"] = str(out_path)
                        st.session_state["stored_mode"] = mode
                    else:
                        with st.spinner("Running compression benchmark..."):
                            best, all_candidates = run_compression_benchmark(path, raw, quality_mode=quality_mode)
                            out_path = run_dir / f"selected_{best.method}{best.ext}"
                            out_path.write_bytes(best.data)

                        st.session_state["compression_candidates"] = all_candidates
                        st.session_state["selected_candidate"] = best
                        st.session_state["stored_bytes"] = best.data
                        st.session_state["stored_file_path"] = str(out_path)
                        st.session_state["stored_mode"] = mode

                    #st.success("Storage bytes prepared.")

        with right:
            st.markdown("### Compression result")
            cand: Optional[CompressionCandidate] = st.session_state.get("selected_candidate")
            if cand:
                c1, c2, c3 = st.columns(3)
                c1.metric("Method", cand.method)
                c2.metric("File extension", cand.kind)
                c3.metric("File size", fmt_bytes(cand.size_bytes))
                c1, c2, c3 = st.columns(3)
                c1.metric("Compression ratio", f"{cand.compression_ratio:.2f}")
                #c2.metric("File saving", f"{cand.saving_pct:.2f}%")
                c2.metric("DNA length", f"{cand.estimated_dna_nt:,} nt")

                rows = [c.public_row() for c in st.session_state.get("compression_candidates", [])[:3]]
                if rows:
                    st.markdown("#### Methods")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                stored_path = st.session_state.get("stored_file_path")
                preview_file(stored_path, "File preview")
                stored_bytes = st.session_state.get("stored_bytes")
                if stored_bytes:
                    download_bytes_button(
                        "Download image",
                        stored_bytes,
                        f"Compressed_{cand.method}{cand.ext}",
                        mime=cand.mime,
                    )
            else:
                st.info("Run Panel 2 to download image.")

def render_panel_3_encoding() -> None:
    with st.container(border=True):
        step_header(3, "Encoding")
        left, right = st.columns(2, gap="large")

        with left:
            #st.markdown("### Bits → DNA")
            mapping = st.radio(
                "DNA design method",
                MAPPING_OPTIONS,
                horizontal=False,
                key="encoding_mapping",
            )
            #st.caption("Fixed settings: init_dimer=TA, whiten=False, prepend_one=True.")
            run = st.button("Run", type="primary", use_container_width=True, key="btn_run_encoding")

            if run:
                stored_bytes = st.session_state.get("stored_bytes")
                if not stored_bytes:
                    st.error("Run Panel 2 first.")
                else:
                    with st.spinner("Encoding bytes to DNA..."):
                        dna, bits, meta = encode_bytes_to_dna(stored_bytes, mapping)
                    st.session_state["bits"] = bits
                    st.session_state["dna"] = dna
                    st.session_state["encoding_meta"] = meta
                    #st.success("DNA encoding completed.")

        with right:
            st.markdown("### DNA preview and statistics")
            dna = st.session_state.get("dna", "")
            bits = st.session_state.get("bits", "")
            if dna:
                hp = dna_codec.homopolymer_stats(dna)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("DNA length", f"{len(dna):,} nt")
                c2.metric("GC content", f"{dna_codec.gc_content(dna):.4f}")
                c3.metric("Longest HP", f"{hp.get('longest', 0)} nt")
                c4.metric("HP count ≥2", f"{hp.get('count_ge2', 0):,}")
                st.text_area(
                    "DNA preview",
                    value=dna[:1000],
                    height=DNA_PREVIEW_HEIGHT,
                    key="dna_preview_text_area",
                )
                download_bytes_button("Download binary string", bits.encode("utf-8"), "encoded_bits.txt", "text/plain")
                download_bytes_button("Download DNA string", dna.encode("utf-8"), "encoded_sequence.dna", "text/plain")
            else:
                st.info("Run Panel 3 to encode DNA.")

def render_panel_4_experiment() -> None:
    with st.container(border=True):
        step_header(4, "DNA Strand Preparation")

        dna = st.session_state.get("dna", "")
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("### Strand design")

            payload_len = st.number_input(
                "Payload length per strand",
                min_value=40,
                max_value=1000,
                value=150,
                step=10,
                key="strand_payload_len",
            )

            fbr = st.text_input(
                "Forward binding region (FBR)",
                value="ACACGACGCTCTTCCGATCT",
                key="strand_fbr",
            )

            rbr = st.text_input(
                "Reverse binding region (RBR)",
                value="AGATCGGAAGAGCACACGTCT",
                key="strand_rbr",
            )

            index_len = st.number_input(
                "Strand index length",
                min_value=4,
                max_value=20,
                value=8,
                step=1,
                key="strand_index_len",
            )

            run = st.button(
                "Prepare DNA strands",
                use_container_width=True,
                key="btn_prepare_dna_strands",
            )

            if run:
                if not dna:
                    st.error("Run DNA encoding first.")
                else:
                    rows = prepare_dna_strands(
                        dna=dna,
                        payload_len=int(payload_len),
                        fbr=fbr,
                        rbr=rbr,
                        index_len=int(index_len),
                    )

                    csv_text = strand_rows_to_csv(rows)
                    preview_text = strand_rows_to_preview(rows, max_rows=5)

                    st.session_state["strand_rows"] = rows
                    st.session_state["strand_csv"] = csv_text
                    st.session_state["strand_preview"] = preview_text

                    st.success(f"DNA strands prepared: {len(rows):,}")

        with right:
            st.markdown("### Strand output")

            rows = st.session_state.get("strand_rows", [])
            csv_text = st.session_state.get("strand_csv", "")
            preview_text = st.session_state.get("strand_preview", "")

            if rows:
                st.metric("DNA strands", f"{len(rows):,}")

                # Show table preview
                df = pd.DataFrame(rows)
                st.dataframe(df.head(10), use_container_width=True, hide_index=True)

                # Optional text preview of complete strands
                st.text_area(
                    "Complete strand preview",
                    value=preview_text,
                    height=220,
                    key="strand_preview_text",
                )

                download_bytes_button(
                    "Download DNA strands CSV",
                    csv_text.encode("utf-8"),
                    "dna_strand_preparation.csv",
                    "text/csv",
                )
            else:
                st.info("Prepare DNA strands after encoding.")


def render_panel_5_decoding() -> None:
    with st.container(border=True):
        step_header(5, "Decoding")
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("### Input")

            source = st.radio(
                "Decode source",
                [
                    "Auto from previous step",
                    "Upload file",
                ],
                horizontal=True,
                key="decode_source",
            )

            input_dna = ""

            if source == "Auto from previous step":
                rows = st.session_state.get("strand_rows", [])

                if rows:
                    # Only payload is used for decoding.
                    # FBR, strand index, and RBR are sequencing/preparation regions.
                    input_dna = "".join(row.get("Payload", "") for row in rows)
                    st.success(
                        f"Loaded {len(rows):,} prepared strands "
                        f"({len(input_dna):,} payload nt)."
                    )
                else:
                    # Fallback: if Panel 4 has not been run, use direct DNA from Encoding.
                    input_dna = st.session_state.get("dna", "")
                    if input_dna:
                        st.warning(
                            f"No prepared strands found. Loaded DNA directly from Encoding "
                            f"({len(input_dna):,} nt)."
                        )
                    else:
                        st.info("No DNA is available yet. Run Encoding and Strand Preparation first.")

            else:
                up = st.file_uploader(
                    "Upload DNA file",
                    type=["txt", "dna", "fasta", "fa", "csv"],
                    key="decode_upload_file",
                )

                if up is not None:
                    text = read_uploaded_text(up)

                    # If uploaded file is CSV from Panel 4, extract Payload column.
                    if up.name.lower().endswith(".csv"):
                        try:
                            import io
                            df = pd.read_csv(io.StringIO(text))

                            if "Payload" in df.columns:
                                input_dna = "".join(
                                    clean_dna(str(x))
                                    for x in df["Payload"].fillna("").tolist()
                                )
                                st.success(
                                    f"Loaded payload from CSV "
                                    f"({len(input_dna):,} nt)."
                                )
                            else:
                                # fallback: clean all DNA-looking characters from file
                                input_dna = clean_dna(text)
                                st.warning(
                                    "CSV does not contain a Payload column. "
                                    "Loaded all DNA-like characters instead."
                                )
                        except Exception as e:
                            input_dna = clean_dna(text)
                            st.warning(f"CSV parsing failed; loaded DNA-like text only. {e}")
                    else:
                        input_dna = clean_dna(text)
                        st.success(f"Loaded DNA file ({len(input_dna):,} nt).")

            st.session_state["decode_input_dna"] = input_dna

            run = st.button(
                "Decoding",
                type="primary",
                use_container_width=True,
                key="btn_run_decoding",
            )

            if run:
                if not input_dna:
                    st.error("No DNA input is available for decoding.")
                else:
                    with st.spinner("Trying 5 mappings and detecting output file..."):
                        try:
                            best, table = blind_decode_dna(input_dna)
                            out_dir = WORK_ROOT / "decoded" / uuid.uuid4().hex
                            restore = write_restored_file(
                                best["data"],
                                str(out_dir),
                                preferred_name="restored",
                            )

                            st.session_state["decode_table"] = table
                            st.session_state["decoded_best"] = best
                            st.session_state["restored_info"] = restore

                            st.success(f"Decoded with {best['mapping']}.")

                        except Exception as e:
                            st.error(str(e))

        with right:
            st.markdown("### Result")

#            table = st.session_state.get("decode_table")
  #          if table is not None:
    #            st.markdown("#### Mapping auto-detection")
       #         st.dataframe(table, use_container_width=True, hide_index=True)

            restore = st.session_state.get("restored_info")
            best = st.session_state.get("decoded_best")

            if restore and best:
                magic = restore.get("magic", {})

                c1, c2, c3 = st.columns(3)
                c1.metric("Design rule", best["mapping"])
                c2.metric("File extension", magic.get("kind", "unknown"))
                c3.metric("File size", fmt_bytes(restore.get("size_bytes")))

                preview_file(restore.get("preview_path"), "preview")

                file_path = restore.get("file_path")
                if file_path and os.path.exists(file_path):
                    download_bytes_button(
                        "Download  file",
                        Path(file_path).read_bytes(),
                        os.path.basename(file_path),
                        magic.get("mime", "application/octet-stream"),
                    )

                extracted_path = restore.get("extracted_path")
                if extracted_path and os.path.exists(extracted_path):
                    download_bytes_button(
                        "Download extracted file",
                        Path(extracted_path).read_bytes(),
                        os.path.basename(extracted_path),
                        "application/octet-stream",
                    )
            else:
                st.info("Run decoding to restore the file.")

def render_panel_6_analysis() -> None:
    with st.container(border=True):
        step_header(6, "Analysis")
        left, right = st.columns(2, gap="large")

        input_path = st.session_state.get("input_path")
        stored_path = st.session_state.get("stored_file_path")
        restore = st.session_state.get("restored_info") or {}
        restored_path = restore.get("file_path")
        preview_path = restore.get("preview_path")
        mode = st.session_state.get("stored_mode", st.session_state.get("storage_mode", "—"))

        with left:
            st.markdown("### Result")
            c1, c2 = st.columns(2)
            with c1:
                preview_file(stored_path, "Encoded image")
            with c2:
                preview_file(restored_path, "Decoded image")

        with right:
            #st.markdown("### Validation")
            if not input_path or not restored_path:
                st.info("Complete upload, encoding, and decoding first.")
                return

            input_bytes = Path(stored_path).read_bytes()
            restored_bytes = Path(restored_path).read_bytes()
            stored_bytes = st.session_state.get("stored_bytes", b"")

            #st.markdown("#### DNA recovery validation")
            if mode == "Non-compression":
                exact = sha256_bytes(input_bytes) == sha256_bytes(restored_bytes)
                st.metric("Exact input recovery", "Passed" if exact else "Failed")
                st.caption("Non-compression expects restored file bytes to match original input bytes 100%.")
            else:
                exact = sha256_bytes(stored_bytes) == sha256_bytes(restored_bytes)
                #st.metric("Recovery", "Passed" if exact else "Failed")
                #st.caption("Compression expects restored bytes to match the selected compressed output bytes 100%.")

            c1, c2 = st.columns(2)
            #c1.metric("Original size", fmt_bytes(len(input_bytes)))
            c2.metric("Decoded file size", fmt_bytes(len(restored_bytes)))

#            st.text_area(
#                "SHA256",
#                value=(
#                    f"input:    {sha256_bytes(input_bytes)}\n"
#                    f"stored:   {sha256_bytes(stored_bytes) if stored_bytes else '—'}\n"
#                    f"restored: {sha256_bytes(restored_bytes)}"
#                ),
#                height=95,
#            )

            st.markdown("#### Quality Validation")
            # Prefer preview path for decompressed archive containers.
            compare_restored = preview_path or restored_path
            m_in = detect_magic(input_bytes)
            m_res = detect_magic(Path(compare_restored).read_bytes()) if compare_restored and os.path.exists(compare_restored) else None

            if m_in and m_in.kind in IMAGE_KINDS and compare_restored:
                metrics = image_metrics(input_path, compare_restored)
                if metrics.get("ok"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                    c2.metric("SSIM", f"{metrics['ssim']:.4f}")
                    c3.metric("MAE", f"{metrics['mae']:.3f}")
                else:
                    st.info(f"Image metrics unavailable: {metrics.get('reason')}")
            elif m_in and m_in.kind == "text" and compare_restored:
                metrics = text_similarity(input_path, compare_restored)
                if metrics.get("ok"):
                    c1, c2 = st.columns(2)
                    c1.metric("Text exact", "Passed" if metrics["exact"] else "Failed")
                    c2.metric("Position accuracy", f"{metrics['char_position_accuracy']:.4f}")
                else:
                    st.info(f"Text metrics unavailable: {metrics.get('reason')}")
            else:
                st.info("For this file type, validation is based on SHA256 and file preview/download.")
