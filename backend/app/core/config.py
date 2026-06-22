from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://aidj:aidj@localhost:5432/aidj"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"

    storage_backend: str = "local"
    local_storage_path: Path = Path("./cache")
    # Phase 11 LRU cache budget. When the total footprint of generated
    # artifacts exceeds this, the evictor deletes least-recently-accessed
    # Songs (audio + stems + their renders) until back under budget. Songs
    # in any queue, songs mid-pipeline, and the mix_plan_logs/ LLM-plan
    # cache are exempt. See app/services/cache/eviction.py.
    cache_max_size_gb: float = 50.0
    s3_endpoint_url: str = ""
    s3_bucket_name: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region_name: str = "auto"
    
    modal_token_id: str = ""
    modal_token_secret: str = ""
    
    genius_access_token: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    # llama-4-scout has 30K TPM on the Groq free tier (vs 8K for
    # gpt-oss-120b). Our trimmed prompt is ~2–3K tokens, so this
    # leaves 10x headroom for retries / parallel renders.
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    # DigitalOcean Serverless Inference (OpenAI-compatible). gpt-oss-120b
    # is the best model accessible on the tier-1 subscription (all OpenAI
    # GPT-5.x and Anthropic Claude models are 403-gated behind a higher
    # tier) and is fast enough (~10s) to stay under the 30s timeout. To
    # use GPT-5.5/Claude, upgrade the DO tier and set openai-gpt-5.5 here.
    do_inference_api_key: str = ""
    do_inference_model: str = "openai-gpt-oss-120b"
    llm_provider: str = "gemini"  # gemini | groq | digitalocean
    use_llm_planner: bool = True
    # Planner architecture. "v2" (default): the LLM picks from
    # pre-validated seam candidates + a transition-style archetype and a
    # deterministic expander computes every timestamp. "legacy": the v1
    # free-form tool-call prompt (now with repair-not-reject validation).
    planner_version: str = "v2"  # v2 | legacy
    # Sampling temperature for plan generation. Re-rolls bump a nonce in
    # the cache key, so >0 keeps re-rolls genuinely different.
    llm_temperature: float = 0.7

    # Section detector backend. "allin1" uses the All-In-One music
    # structure model (real verse/chorus labels + neural beats) when the
    # optional `allin1` package is installed, falling back to librosa
    # otherwise. See app/services/analysis/sections/allin1_detector.py.
    section_detector: str = "librosa_laplacian"  # librosa_laplacian | allin1

    # How key clashes between adjacent songs are handled:
    # - "whole_song" (default): a set-level resolver assigns each song a
    #   constant semitone offset (capped at ±2) applied to its ENTIRE
    #   play, so there is never an audible mid-song key glide. Clashes
    #   beyond the cap are accepted and masked by transition style.
    # - "temporary": the v1 behavior — B is held in A's key during the
    #   crossfade then glided back to native afterwards (audible).
    # - "off": never pitch-shift; clashes are masked by style choice only.
    pitch_mode: str = "whole_song"  # whole_song | temporary | off

    # Seam loudness matching: align the incoming song's perceived level
    # to the outgoing one at the seam (no energy pothole/spike), then
    # glide back to its native level after the crossfade. Boost capped
    # at +4 dB, cut at -6 dB; gaps under ~1.25 dB are left alone.
    loudness_match: bool = True

    # Path to a Netscape-format cookies.txt that yt-dlp passes to YouTube.
    # Required on cloud hosts (the droplet) — YouTube's anti-bot system
    # rejects datacenter IPs unless an authenticated session is presented.
    # On macOS dev the residential IP is unblocked and this can stay empty.
    yt_dlp_cookies_file: str = ""

    # Phase 6: skip Whisper if the Stems row's `vocal_rms` is below this
    # threshold. 0.005 sits well below an audible monologue (~0.04 in our
    # smoke-test clip) and well above silence — only true instrumentals
    # and ambient tracks land underneath.
    whisper_vocal_rms_threshold: float = 0.005

settings = Settings()
