"""TTS provider abstraction for the host voice.

Two providers behind ``settings.tts_provider``:

  * "kokoro" (default) — Kokoro-82M locally: free, no API key. Weights
    (~330 MB) download from HuggingFace on first synthesis; the pipeline
    is cached module-level so a set's handful of clips share one load.
  * "openai" — gpt-4o-mini-tts over the REST API (needs openai_api_key).

Everything returns 44.1 kHz stereo float32 or None — a missing voice
must never fail a stitch. Clips are cached in storage under
``host_clips/{sha1(provider|voice|text)}.wav`` so re-stitching a queue
doesn't re-synthesize.
"""

from __future__ import annotations

import hashlib
import io
import logging

import numpy as np
import soundfile as sf

from app.core.config import settings

logger = logging.getLogger(__name__)

TARGET_SR = 44100
KOKORO_SR = 24000
# Speech normalized to this RMS before mixdown (≈ -14 dBFS).
VOICE_TARGET_RMS = 0.2

_kokoro_pipeline = None


def clip_cache_key(text: str, voice: str, provider: str) -> str:
    digest = hashlib.sha1(
        f"{provider}|{voice}|{text}".encode("utf-8")
    ).hexdigest()
    return f"host_clips/{digest}.wav"


def _to_stereo_44k(mono_or_stereo: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(mono_or_stereo, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.column_stack((audio, audio))
    if sr != TARGET_SR:
        import librosa

        audio = np.column_stack([
            librosa.resample(audio[:, ch], orig_sr=sr, target_sr=TARGET_SR)
            for ch in range(audio.shape[1])
        ]).astype(np.float32)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms > 1e-6:
        audio = audio * (VOICE_TARGET_RMS / rms)
    return np.clip(audio, -0.99, 0.99).astype(np.float32)


def _synthesize_kokoro(text: str, voice: str) -> np.ndarray | None:
    global _kokoro_pipeline
    try:
        if _kokoro_pipeline is None:
            from kokoro import KPipeline

            # 'a' = American English. Voice ids pick accents/genders.
            _kokoro_pipeline = KPipeline(lang_code="a")
        chunks = [
            audio for _, _, audio in _kokoro_pipeline(text, voice=voice)
        ]
        if not chunks:
            return None
        mono = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
        return _to_stereo_44k(mono, KOKORO_SR)
    except Exception:
        logger.exception("host tts: kokoro synthesis failed")
        return None


def _synthesize_openai(text: str, voice: str) -> np.ndarray | None:
    if not settings.openai_api_key:
        logger.warning("host tts: tts_provider=openai but no openai_api_key")
        return None
    try:
        import httpx

        resp = httpx.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": "gpt-4o-mini-tts",
                "voice": voice,
                "input": text,
                "response_format": "wav",
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        audio, sr = sf.read(io.BytesIO(resp.content), always_2d=False,
                            dtype="float32")
        return _to_stereo_44k(audio, sr)
    except Exception:
        logger.exception("host tts: openai synthesis failed")
        return None


def synthesize(text: str, voice: str | None = None) -> np.ndarray | None:
    """Text -> (samples, 2) float32 at 44.1 kHz, or None on any failure."""
    provider = settings.tts_provider
    voice = voice or settings.tts_voice
    if provider == "kokoro":
        return _synthesize_kokoro(text, voice)
    if provider == "openai":
        return _synthesize_openai(text, voice)
    return None


async def synthesize_cached(storage, text: str, voice: str | None = None):
    """Storage-cached synthesis: hit ``host_clips/`` first, write after.

    Cache failures degrade to direct synthesis; synthesis failures return
    None (callers skip the slot)."""
    provider = settings.tts_provider
    voice = voice or settings.tts_voice
    key = clip_cache_key(text, voice, provider)
    try:
        data = await storage.read(key)
        audio, sr = sf.read(io.BytesIO(data), always_2d=True, dtype="float32")
        if sr == TARGET_SR:
            return audio
    except Exception:
        pass  # miss — synthesize below

    audio = synthesize(text, voice)
    if audio is None:
        return None
    try:
        buf = io.BytesIO()
        sf.write(buf, audio, TARGET_SR, format="WAV", subtype="PCM_16")
        await storage.write(key, buf.getvalue())
    except Exception:
        logger.warning("host tts: failed to cache clip %s", key, exc_info=True)
    return audio
