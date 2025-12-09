import json
import logging
import os
from typing import Any, Dict, Optional

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    redis = None

LOGGER = logging.getLogger(__name__)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CONTEXT_KEY = os.getenv("ESG_CONTEXT_KEY", "esg_ai_agent_context")


class RedisKVStore:
    """간단한 키-값 스토어 래퍼 (Redis 없으면 비활성)."""

    def __init__(self) -> None:
        self._client: Optional["redis.Redis"] = None
        if redis is None:
            LOGGER.warning("redis 패키지가 설치되지 않아 KV 스토어 기능이 비활성화됩니다.")
            return
        try:
            self._client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self._client.ping()
            LOGGER.info("Redis KV 스토어 연결 성공: %s", REDIS_URL)
        except Exception as exc:  # pragma: no cover - 네트워크 예외
            LOGGER.warning("Redis 연결 실패(%s): %s", REDIS_URL, exc)
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def load_context(self) -> Optional[Dict[str, Any]]:
        if not self._client:
            return None
        data = self._client.get(CONTEXT_KEY)
        if not data:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            LOGGER.error("Redis에 저장된 컨텍스트 JSON 파싱 실패")
            return None

    def save_context(self, context: Dict[str, Any]) -> bool:
        if not self._client:
            return False
        try:
            payload = json.dumps(context, ensure_ascii=False)
            self._client.set(CONTEXT_KEY, payload)
            return True
        except Exception as exc:  # pragma: no cover - 네트워크 예외
            LOGGER.error("Redis 컨텍스트 저장 실패: %s", exc)
            return False


kv_store = RedisKVStore()
