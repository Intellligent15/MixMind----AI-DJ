"""Modal GPU functions for stem separation and transcription.

Runs on Modal's T4 GPUs (Linux + CUDA). Used when the host worker can't
or doesn't want to run heavy ML locally — primarily when deployed on a
small DigitalOcean droplet that has neither MPS nor enough CPU to do
Demucs in reasonable time.

The functions are deliberately self-contained: they download from DO
Spaces, run the model, upload results back, and return small
JSON-serializable summaries. The host worker writes the DB row from
that summary. No `app.*` imports happen inside Modal, so the image
stays slim.

Credentials are passed as function arguments rather than via
`modal.Secret`, so deploying the module doesn't require any one-time
`modal secret create` step. The host already has the credentials
loaded from `.env`; it forwards them per call.

Deploying: `cd backend && uv run modal deploy app/workers/modal_stubs.py`
once after pulling a new image spec. Calls from the workers use
`modal.Function.from_name(APP_NAME, ...)`. If you skip the deploy,
ephemeral apps are created per call — works but cold-start is slow.
"""

from __future__ import annotations

import modal

APP_NAME = "ai-dj-gpu-workers"

# Essentia model files for tag_audio, baked into the image at build time
# (~20 MB total). Embeddings come from Discogs-EffNet; two small
# classification heads map those to genre and mood/theme labels. The
# .json metadata files carry the class-label lists.
ESSENTIA_MODELS_DIR = "/models"
ESSENTIA_MODEL_URLS = (
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json",
)

# Image: slim Debian + Python 3.11, ffmpeg, ML libs, S3 client. boto3
# (sync) is used inside the function — async aioboto3 brings no benefit
# inside a one-shot Modal call.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "wget")
    .pip_install(
        "demucs>=4.0.1",
        "openai-whisper>=20240930",
        "soundfile>=0.13.1",
        "numpy<2",
        "torch>=2.1",
        "torchaudio>=2.1",
        # torchaudio >= ~2.9 deprecated its native audio loader in favor
        # of TorchCodec. Without this, torchaudio.load() raises
        # ImportError. Pinning here is simpler than capping torchaudio.
        "torchcodec",
        "boto3>=1.34",
        "essentia-tensorflow",
    )
    .run_commands(
        f"mkdir -p {ESSENTIA_MODELS_DIR} && "
        + " && ".join(
            f"wget -q -P {ESSENTIA_MODELS_DIR} {url}" for url in ESSENTIA_MODEL_URLS
        )
    )
)

app = modal.App(APP_NAME, image=image)


def _make_s3_client(
    endpoint: str,
    bucket: str,
    access_key: str,
    secret_key: str,
    region: str,
):
    """Mirror app.services.storage.s3._normalize_endpoint here so the
    Modal side reads/writes to the same key paths the host worker uses.

    DO Spaces has two endpoint URLs that look almost identical: the
    `<bucket>.<region>.digitaloceanspaces.com` "origin endpoint" (what
    you copy from the DO panel) and the `<region>.digitaloceanspaces.com`
    API endpoint that boto3 actually wants. Pass the former and boto3
    silently double-prefixes the bucket name into every key path; pass
    the latter with virtual-hosted addressing and keys are clean.
    """
    import re
    import boto3
    from botocore.config import Config

    m = re.match(
        r"^(https?://)([a-z0-9.\-]+)\.([a-z0-9-]+\.digitaloceanspaces\.com)/?$",
        endpoint,
    )
    if m and m.group(2) == bucket:
        endpoint = f"{m.group(1)}{m.group(3)}"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(
            s3={"addressing_style": "virtual"},
            signature_version="s3v4",
        ),
    )


@app.function(gpu="T4", timeout=1800)
def run_separation(
    audio_key: str,
    video_id: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret: str,
    s3_region: str,
) -> dict:
    """Demucs htdemucs separation. Downloads audio, splits into 4 stems,
    computes the vocal envelope, uploads everything back to S3, returns
    the summary the calling worker writes into the Stems row.
    """
    import json
    import os
    import tempfile
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    s3 = _make_s3_client(s3_endpoint, s3_bucket, s3_access_key, s3_secret, s3_region)
    model_name = "htdemucs_ft"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        local_audio = td / "audio.wav"
        s3.download_file(s3_bucket, audio_key, str(local_audio))

        waveform, sr = torchaudio.load(str(local_audio))
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            waveform = resampler(waveform)
            sr = 44100

        model = get_model(model_name)
        model.cpu()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        with torch.no_grad():
            sources = apply_model(
                model,
                waveform.unsqueeze(0).to(device),
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
        # sources: (4, channels, samples) ordered as drums, bass, other, vocals
        names = model.sources  # ["drums", "bass", "other", "vocals"]
        stems = {name: sources[i].cpu().numpy().T for i, name in enumerate(names)}

        # Per-stem envelopes: frame-wise RMS + peak at 10 Hz, same schema
        # as the local path (services/stems/service.py::_compute_stem_envelopes).
        frame_hz = 10
        hop = sr // frame_hz
        envelopes: dict = {"frame_hz": frame_hz}
        for name, arr in stems.items():
            mono = arr.mean(axis=1) if arr.ndim == 2 else arr
            n_frames = len(mono) // hop
            trimmed = mono[: n_frames * hop].reshape(n_frames, hop)
            envelopes[name] = {
                "rms": np.sqrt(np.mean(trimmed ** 2, axis=1)).tolist(),
                "peak": np.abs(trimmed).max(axis=1).tolist(),
            }
        vocals = stems["vocals"]
        vocals_mono = vocals.mean(axis=1) if vocals.ndim == 2 else vocals
        vocal_rms = float(np.sqrt(np.mean(vocals_mono ** 2)))

        keys: dict[str, str] = {}
        for name, arr in stems.items():
            dest = td / f"{name}.wav"
            sf.write(str(dest), arr.astype(np.float32), sr, subtype="PCM_16")
            key = f"stems/{video_id}/{name}.wav"
            s3.upload_file(str(dest), s3_bucket, key)
            keys[name] = key

        env_key = f"stems/{video_id}/envelopes.json"
        env_dest = td / "envelopes.json"
        env_dest.write_text(json.dumps(envelopes))
        s3.upload_file(str(env_dest), s3_bucket, env_key)

    return {
        "vocals_path": keys["vocals"],
        "drums_path": keys["drums"],
        "bass_path": keys["bass"],
        "other_path": keys["other"],
        "envelopes_path": env_key,
        "vocal_rms": vocal_rms,
        "model_name": model_name,
    }


@app.function(gpu="T4", timeout=900)
def run_transcription(
    vocals_key: str,
    video_id: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret: str,
    s3_region: str,
) -> dict:
    """Whisper large-v3 transcription of the vocal stem. Mirrors the
    shape of TranscriptionService.transcribe(): per-segment text + word-
    level timing where available.
    """
    import json
    import tempfile
    from pathlib import Path

    import whisper

    s3 = _make_s3_client(s3_endpoint, s3_bucket, s3_access_key, s3_secret, s3_region)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        local = td / "vocals.wav"
        s3.download_file(s3_bucket, vocals_key, str(local))

        model = whisper.load_model("large-v3")
        kwargs = {
            "word_timestamps": True,
            "verbose": False,
            "compression_ratio_threshold": 3.0,
            "logprob_threshold": -1.2,
            "no_speech_threshold": 0.45,
            "hallucination_silence_threshold": 1.5,
            "condition_on_previous_text": False,
            "temperature": (0.0, 0.2),
            # No language= — let Whisper auto-detect to match the local path.
            "best_of": 3,
            "suppress_tokens": [-1],
            "suppress_blank": True,
        }
        
        kwargs["initial_prompt"] = (
            "Transcribe only the sung or spoken vocals. Do not add "
            "descriptions of music, applause, silence, or instrumental "
            "sounds."
        )

        result = model.transcribe(str(local), **kwargs)

        segments = []
        for seg in result.get("segments", []):
            words = []
            for w in seg.get("words", []) or []:
                words.append(
                    {
                        "word": w.get("word", "").strip(),
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0)),
                        "probability": float(w.get("probability", 0.0)),
                    }
                )
            segments.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": seg["text"].strip(),
                    "avg_logprob": float(seg.get("avg_logprob", 0.0)),
                    "words": words,
                }
            )

        out_path = td / "segments.json"
        out_path.write_text(json.dumps(segments))
        key = f"transcriptions/{video_id}.json"
        s3.upload_file(str(out_path), s3_bucket, key)

    return {
        "transcription_path": key,
        "language": result.get("language"),
        "duration": float(result.get("duration", 0.0))
        if "duration" in result
        else None,
        "model_name": "whisper-large-v3",
        "segments": segments,
    }


# EffNet inference on a full track is a few seconds on CPU, and the
# essentia-tensorflow wheel is CPU-only anyway — no GPU requested.
@app.function(timeout=600)
def tag_audio(
    audio_key: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_access: str,
    s3_secret: str,
    s3_region: str,
) -> dict:
    """Download audio, run Essentia Discogs-EffNet tagging, and return
    ``{"genres": [...], "moods": [...]}`` (most-probable first)."""
    import json
    import os
    import tempfile

    from essentia.standard import (
        MonoLoader,
        TensorflowPredict2D,
        TensorflowPredictEffnetDiscogs,
    )

    def _labels(metadata_file: str) -> list[str]:
        path = os.path.join(ESSENTIA_MODELS_DIR, metadata_file)
        with open(path) as f:
            return json.load(f)["classes"]

    def _top(probs, labels: list[str], k: int, threshold: float) -> list[tuple[str, float]]:
        ranked = sorted(zip(labels, probs), key=lambda p: p[1], reverse=True)
        return [(label, float(p)) for label, p in ranked[:k] if p >= threshold]

    s3 = _make_s3_client(s3_endpoint, s3_bucket, s3_access, s3_secret, s3_region)
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        s3.download_file(s3_bucket, audio_key, audio_path)

        # EffNet models want 16 kHz mono input.
        audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
        embeddings = TensorflowPredictEffnetDiscogs(
            graphFilename=os.path.join(ESSENTIA_MODELS_DIR, "discogs-effnet-bs64-1.pb"),
            output="PartitionedCall:1",
        )(audio)

        genre_probs = TensorflowPredict2D(
            graphFilename=os.path.join(
                ESSENTIA_MODELS_DIR, "genre_discogs400-discogs-effnet-1.pb"
            ),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )(embeddings).mean(axis=0)

        mood_probs = TensorflowPredict2D(
            graphFilename=os.path.join(
                ESSENTIA_MODELS_DIR, "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
            ),
            input="model/Placeholder",
            output="model/Sigmoid",
        )(embeddings).mean(axis=0)

    # Discogs400 labels are "Parent---Subgenre"; keep both, most-probable
    # first, deduped (several subgenres share a parent).
    genres: list[str] = []
    for label, _ in _top(genre_probs, _labels("genre_discogs400-discogs-effnet-1.json"), k=5, threshold=0.1):
        for part in label.split("---"):
            part = part.strip().lower()
            if part and part not in genres:
                genres.append(part)

    # Mood/theme is multi-label with characteristically low sigmoid
    # activations — a lower threshold than genre is intentional.
    moods = [
        label.lower()
        for label, _ in _top(
            mood_probs,
            _labels("mtg_jamendo_moodtheme-discogs-effnet-1.json"),
            k=5,
            threshold=0.05,
        )
    ]

    return {"genres": genres, "moods": moods}
