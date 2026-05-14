# Headerless DNA Storage Pipeline — Modular Streamlit App

Run:

```bash
streamlit run app.py
```

Keep these files in the same folder as your existing backend files:

- `dna_codec.py`
- `utils_bits_v2.py`
- `compressors_v2.py`

## Files

- `app.py` — main Streamlit entry point.
- `config.py` — global settings, preview sizes, supported file kinds, mapping names.
- `styles.py` — CSS/font/card styling.
- `ui_helpers.py` — upload, preview, file detection display, download buttons.
- `compression_pipeline.py` — compression benchmark and candidate selection.
- `dna_mapping.py` — 5 DNA mapping methods and automatic blind decoding.
- `fragments.py` — Sanger/Illumina fragment formatting.
- `restore_analysis.py` — file restoration and quality metrics.
- `panels.py` — all six UI panels.

## Preview size

Edit `config.py`:

```python
IMAGE_PREVIEW_USE_CONTAINER_WIDTH = True
IMAGE_PREVIEW_WIDTH = 420
TEXT_PREVIEW_HEIGHT = 250
DNA_PREVIEW_HEIGHT = 150
FRAGMENT_PREVIEW_HEIGHT = 180
```

Set `IMAGE_PREVIEW_USE_CONTAINER_WIDTH = False` if you want fixed image width.


## Audio compression

For WAV/audio, the compression benchmark now tries audio-native codecs first:

- `flac_lossless`
- `opus_ogg_32k`, `opus_ogg_64k`, `opus_ogg_96k`, `opus_ogg_128k`
- `mp3_96k`, `mp3_128k`, `mp3_192k`
- `aac_m4a_96k`, `aac_m4a_128k`

These require `ffmpeg` on PATH. Check with:

```bash
ffmpeg -version
```

If ffmpeg is missing, the app falls back to generic containers such as gzip/xz/zip.
