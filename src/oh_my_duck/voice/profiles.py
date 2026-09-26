import hashlib
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import soundfile as sf

from oh_my_duck.core.contracts.sensors import PayloadRef
from oh_my_duck.voice.base import VoiceCandidate, VoiceProfile


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WAV_SUBTYPES = frozenset({"PCM_16", "PCM_24", "PCM_32", "FLOAT"})
_PROFILE_COLUMNS = (
    "persona_id, revision, voice_id, audio_sha256, reference_text, "
    "design_model_revision, synthesis_model_revision, confirmed_at"
)


def _required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")


def _validate_profile(profile: VoiceProfile) -> None:
    for name in (
        "persona_id", "voice_id", "reference_text", "design_model_revision",
        "synthesis_model_revision", "confirmed_at",
    ):
        _required(getattr(profile, name), name)
    if not isinstance(profile.revision, int) or isinstance(profile.revision, bool) or profile.revision < 1:
        raise ValueError("revision 必须是正整数")
    confirmed_at = datetime.fromisoformat(profile.confirmed_at)
    if confirmed_at.tzinfo is None or confirmed_at.utcoffset() is None:
        raise ValueError("confirmed_at 必须包含时区")
    if profile.reference_audio.media_type not in {"audio/wav", "audio/x-wav"}:
        raise ValueError("reference_audio.media_type 必须是 WAV")
    if profile.reference_audio.sha256 is None or _SHA256.fullmatch(profile.reference_audio.sha256) is None:
        raise ValueError("reference_audio.sha256 必须是小写 SHA256")


def _source_path(uri: str) -> Path:
    _required(uri, "reference_audio.uri")
    if uri.startswith("file:"):
        parsed = urlsplit(uri)
        if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("仅支持本地 file URI")
        path = Path(unquote(parsed.path))
    else:
        path = Path(uri)
    if not path.is_absolute():
        raise ValueError("参考音频必须使用本地绝对路径")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_wav(path: Path) -> None:
    info = sf.info(path)
    if info.format != "WAV" or info.subtype not in _WAV_SUBTYPES:
        raise ValueError("参考音频必须是受支持的 WAV 编码")
    if info.channels not in {1, 2} or info.samplerate <= 0 or info.frames <= 0:
        raise ValueError("参考音频必须包含单声道或双声道音频帧")
    with sf.SoundFile(path) as audio:
        frames = 0
        while True:
            block = audio.read(65536, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            if not np.isfinite(block).all():
                raise ValueError("参考音频包含非有限采样值")
            frames += len(block)
        if frames != info.frames:
            raise ValueError("参考音频帧不完整")


class SQLiteVoiceProfileStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.audio_dir = self.root / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "voice_profiles.sqlite3"
        with closing(self._connection()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    persona_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    voice_id TEXT NOT NULL,
                    audio_sha256 TEXT NOT NULL,
                    reference_text TEXT NOT NULL,
                    design_model_revision TEXT NOT NULL,
                    synthesis_model_revision TEXT NOT NULL,
                    confirmed_at TEXT NOT NULL,
                    PRIMARY KEY (persona_id, revision)
                );
                CREATE TABLE IF NOT EXISTS active_profiles (
                    persona_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    FOREIGN KEY (persona_id, revision) REFERENCES profiles (persona_id, revision)
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    audio_sha256 TEXT NOT NULL,
                    reference_text TEXT NOT NULL,
                    model_revision TEXT NOT NULL
                );
                """
            )
        self.db_path.chmod(0o600)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _audio_path(self, sha256: str) -> Path:
        return self.audio_dir / f"{sha256}.wav"

    def _verified_audio(self, sha256: str) -> PayloadRef:
        if _SHA256.fullmatch(sha256) is None:
            raise ValueError("已保存的音频 SHA256 无效")
        path = self._audio_path(sha256)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != sha256:
            raise ValueError("已保存的参考音频 SHA256 不匹配")
        _validate_wav(path)
        return PayloadRef(uri=path.as_uri(), media_type="audio/wav", sha256=sha256)

    def _save_audio(self, source: Path, sha256: str) -> None:
        destination = self._audio_path(sha256)
        temporary = tempfile.NamedTemporaryFile(dir=self.audio_dir, suffix=".wav", delete=False)
        temporary_path = Path(temporary.name)
        try:
            with temporary:
                with source.open("rb") as original:
                    shutil.copyfileobj(original, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            if _sha256(temporary_path) != sha256:
                raise ValueError("参考音频 SHA256 不匹配")
            _validate_wav(temporary_path)
            if not destination.exists():
                os.link(temporary_path, destination)
            self._verified_audio(sha256)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _same_content(row: sqlite3.Row, profile: VoiceProfile) -> bool:
        return (
            row["voice_id"] == profile.voice_id
            and row["audio_sha256"] == profile.reference_audio.sha256
            and row["reference_text"] == profile.reference_text
            and row["design_model_revision"] == profile.design_model_revision
            and row["synthesis_model_revision"] == profile.synthesis_model_revision
            and row["confirmed_at"] == profile.confirmed_at
        )

    def save_confirmed(self, profile: VoiceProfile) -> None:
        _validate_profile(profile)
        source = _source_path(profile.reference_audio.uri)
        with closing(self._connection()) as connection:
            connection.row_factory = sqlite3.Row
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    f"SELECT {_PROFILE_COLUMNS} FROM profiles WHERE persona_id = ? AND revision = ?",
                    (profile.persona_id, profile.revision),
                ).fetchone()
                if existing is not None:
                    if not self._same_content(existing, profile):
                        raise ValueError("同一 revision 已保存不同内容")
                    if _sha256(source) != profile.reference_audio.sha256:
                        raise ValueError("参考音频 SHA256 不匹配")
                    _validate_wav(source)
                    self._verified_audio(existing["audio_sha256"])
                    return
                latest = connection.execute(
                    "SELECT MAX(revision) FROM profiles WHERE persona_id = ?", (profile.persona_id,)
                ).fetchone()[0]
                expected = 1 if latest is None else latest + 1
                if profile.revision != expected:
                    raise ValueError(f"revision 必须是 {expected}")
                self._save_audio(source, profile.reference_audio.sha256)
                connection.execute(
                    "INSERT INTO profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        profile.persona_id, profile.revision, profile.voice_id,
                        profile.reference_audio.sha256, profile.reference_text,
                        profile.design_model_revision, profile.synthesis_model_revision,
                        profile.confirmed_at,
                    ),
                )
                connection.execute(
                    "INSERT INTO active_profiles (persona_id, revision) VALUES (?, ?) "
                    "ON CONFLICT (persona_id) DO UPDATE SET revision = excluded.revision",
                    (profile.persona_id, profile.revision),
                )

    def _from_row(self, row: sqlite3.Row) -> VoiceProfile:
        return VoiceProfile(
            persona_id=row["persona_id"],
            voice_id=row["voice_id"],
            revision=row["revision"],
            reference_audio=self._verified_audio(row["audio_sha256"]),
            reference_text=row["reference_text"],
            design_model_revision=row["design_model_revision"],
            synthesis_model_revision=row["synthesis_model_revision"],
            confirmed_at=row["confirmed_at"],
        )

    def save_candidate(self, candidate: VoiceCandidate) -> None:
        for name in ("candidate_id", "description", "reference_text", "model_revision"):
            _required(getattr(candidate, name), name)
        if candidate.reference_audio.media_type not in {"audio/wav", "audio/x-wav"}:
            raise ValueError("候选音频必须是 WAV")
        sha256 = candidate.reference_audio.sha256
        if sha256 is None or _SHA256.fullmatch(sha256) is None:
            raise ValueError("候选音频 SHA256 无效")
        source = _source_path(candidate.reference_audio.uri)
        with closing(self._connection()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM candidates WHERE candidate_id = ?", (candidate.candidate_id,)
                ).fetchone() is not None:
                    raise ValueError("candidate_id 已存在")
                self._save_audio(source, sha256)
                connection.execute(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
                    (candidate.candidate_id, candidate.description, sha256,
                     candidate.reference_text, candidate.model_revision),
                )

    def candidate(self, candidate_id: str) -> VoiceCandidate | None:
        _required(candidate_id, "candidate_id")
        with closing(self._connection()) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT candidate_id, description, audio_sha256, reference_text, model_revision "
                "FROM candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            return None
        return VoiceCandidate(
            candidate_id=row["candidate_id"], description=row["description"],
            reference_audio=self._verified_audio(row["audio_sha256"]),
            reference_text=row["reference_text"], model_revision=row["model_revision"],
        )

    def confirm_candidate(
        self, candidate_id: str, persona_id: str, synthesis_model_revision: str
    ) -> VoiceProfile:
        candidate = self.candidate(candidate_id)
        if candidate is None:
            raise ValueError("candidate_id 不存在")
        _required(persona_id, "persona_id")
        _required(synthesis_model_revision, "synthesis_model_revision")
        latest = self.history(persona_id)
        profile = VoiceProfile(
            persona_id=persona_id, voice_id=candidate.candidate_id,
            revision=1 if not latest else latest[-1].revision + 1,
            reference_audio=candidate.reference_audio,
            reference_text=candidate.reference_text,
            design_model_revision=candidate.model_revision,
            synthesis_model_revision=synthesis_model_revision,
            confirmed_at=datetime.now().astimezone().isoformat(),
        )
        self.save_confirmed(profile)
        return self.get(persona_id, profile.revision)

    def active(self, persona_id: str) -> VoiceProfile | None:
        _required(persona_id, "persona_id")
        with closing(self._connection()) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                f"SELECT {_PROFILE_COLUMNS} FROM profiles JOIN active_profiles USING (persona_id, revision) "
                "WHERE persona_id = ?",
                (persona_id,),
            ).fetchone()
        return None if row is None else self._from_row(row)

    def get(self, persona_id: str, revision: int) -> VoiceProfile | None:
        _required(persona_id, "persona_id")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise ValueError("revision 必须是正整数")
        with closing(self._connection()) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                f"SELECT {_PROFILE_COLUMNS} FROM profiles WHERE persona_id = ? AND revision = ?",
                (persona_id, revision),
            ).fetchone()
        return None if row is None else self._from_row(row)

    def history(self, persona_id: str) -> tuple[VoiceProfile, ...]:
        _required(persona_id, "persona_id")
        with closing(self._connection()) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"SELECT {_PROFILE_COLUMNS} FROM profiles WHERE persona_id = ? ORDER BY revision",
                (persona_id,),
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)
