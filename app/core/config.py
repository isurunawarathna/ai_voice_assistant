"""Application configuration.

Every tunable lives here so that nothing in the runtime reads ``os.environ``
directly. Values are loaded from the environment (and ``.env``) using nested
delimiters, e.g. ``STT__MODEL_SIZE=medium`` or ``SESSION__BACKEND=redis``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Environment(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class SessionBackend(StrEnum):
    MEMORY = "memory"
    REDIS = "redis"


class AgentBackend(StrEnum):
    """Which conversation agent is wired in at the RAG seam."""

    ECHO = "echo"
    HYBRID_RAG = "hybrid_rag"


class TTSBackend(StrEnum):
    NONE = "none"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    json_format: bool = True
    include_timestamp: bool = True

    @field_validator("level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()


class AudioSettings(BaseModel):
    """Constraints applied to every inbound audio frame."""

    assets_dir: Path = BASE_DIR / "assets" / "audio"
    target_sample_rate: int = 16_000
    max_bytes: int = 10 * 1024 * 1024
    max_seconds: float = 60.0
    min_seconds: float = 0.20
    allow_resampling: bool = True
    # Browsers record WebM/Opus, not WAV. When enabled, non-RIFF payloads are
    # decoded through PyAV so the Angular client can stream MediaRecorder output.
    allow_compressed_input: bool = True
    # Fail startup when a declared prompt file is missing. Keep ``True`` in
    # production: a half-loaded prompt library is a broken greeting flow.
    strict_prompt_loading: bool = True


class STTSettings(BaseModel):
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 4
    num_workers: int = 1
    beam_size: int = 5
    vad_filter: bool = True
    # Whisper hallucinates on silence; drop segments below this log-prob.
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    # Transcriptions that may run in parallel. CPU inference is not reentrant
    # in a useful way, so 1 is the safe production default.
    max_concurrency: int = 1
    warmup_on_startup: bool = True
    timeout_seconds: float = 45.0
    download_root: Path | None = None


class SessionSettings(BaseModel):
    backend: SessionBackend = SessionBackend.MEMORY
    ttl_seconds: int = 1_800
    max_active_sessions: int = 100
    max_history_turns: int = 12
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "voice:session:"
    sweep_interval_seconds: int = 60


class ConversationSettings(BaseModel):
    language_max_retries: int = Field(default=3, ge=1, le=10)
    idle_timeout_seconds: float = Field(default=180.0, gt=0)
    # Longest silence tolerated between two frames during language selection.
    language_selection_timeout_seconds: float = Field(default=120.0, gt=0)
    # Hard ceiling for the whole selection phase, however busy the client
    # keeps the socket. Per-frame timeouts alone can be reset forever.
    language_selection_deadline_seconds: float = Field(default=300.0, gt=0)
    # Consecutive unusable inputs (undecodable audio, protocol errors, STT
    # failures, silence) tolerated before the call is closed.
    max_invalid_inputs: int = Field(default=8, ge=1)
    # Each ``reset`` replays a ~1.4 MB prompt, so it is bounded per session.
    max_resets: int = Field(default=10, ge=0)
    agent_timeout_seconds: float = Field(default=30.0, gt=0)
    send_transcripts: bool = True


class AgentSettings(BaseModel):
    """Configuration for the pluggable conversation agent (hybrid RAG seam)."""

    backend: AgentBackend = AgentBackend.ECHO
    # Import path to a factory/class implementing ``ConversationAgent``, used
    # when ``backend`` is ``hybrid_rag``. Example:
    # ``AGENT__IMPORT_PATH=rag.graph:build_agent``
    import_path: str | None = None
    top_k: int = 5
    max_context_chars: int = 8_000
    stream_responses: bool = False


class TTSSettings(BaseModel):
    backend: TTSBackend = TTSBackend.NONE
    voice_si: str = "si-LK-ThiliniNeural"
    voice_en: str = "en-US-JennyNeural"
    timeout_seconds: float = 20.0


class SecuritySettings(BaseModel):
    """Abuse controls for the public WebSocket.

    Every new connection costs the server several megabytes of prompt audio,
    so unauthenticated callers are limited before the conversation starts.
    """

    # ``None`` means: enforce the Origin check everywhere except ``local``.
    # Non-browser clients (scripts, Streamlit, telephony) send no Origin
    # header and are always allowed; browsers must match ``allowed_origins``.
    enforce_origin: bool | None = None
    # ``None`` falls back to ``Settings.cors_origins``.
    allowed_origins: Annotated[list[str] | None, NoDecode] = None
    # 0 disables a limit.
    max_connections_per_ip: int = Field(default=5, ge=0)
    max_connects_per_minute_per_ip: int = Field(default=30, ge=0)
    # Only enable behind a proxy you control; the header is client-supplied.
    trust_forwarded_for: bool = False

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "C-Clark Campus AI Voice Assistant"
    version: str = "2.1.0"
    environment: Environment = Environment.LOCAL
    api_prefix: str = "/api/v1"
    # NoDecode: accept "a,b" from the environment instead of requiring JSON.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:4200"]
    )

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    metrics_enabled: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _production_guard_rails(self) -> Settings:
        """Refuse configurations that would silently weaken a live greeting."""

        if self.environment is Environment.PRODUCTION:
            if not self.audio.strict_prompt_loading:
                raise ValueError("AUDIO__STRICT_PROMPT_LOADING must stay true in production: ""a half-loaded prompt library is a broken greeting flow")
            if self.conversation.language_selection_timeout_seconds > self.conversation.language_selection_deadline_seconds:
                raise ValueError("CONVERSATION__LANGUAGE_SELECTION_TIMEOUT_SECONDS cannot exceed CONVERSATION__LANGUAGE_SELECTION_DEADLINE_SECONDS")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    @property
    def origin_enforced(self) -> bool:
        flag = self.security.enforce_origin
        return flag if flag is not None else self.environment is not Environment.LOCAL

    @property
    def websocket_origins(self) -> list[str]:
        configured = self.security.allowed_origins
        return list(configured if configured is not None else self.cors_origins)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()