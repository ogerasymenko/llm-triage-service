"""Idempotency with a result cache. STUB — stage 5.

Your job: make the same idempotency key return the same answer, without calling
upstream twice.

The stubs below are PASS-THROUGH: they run, and do nothing useful. Stage 1
drives requests through the whole service, so a stub that raised would block
stage 1 on this one. Each `# stage 5:` comment says what replaces it.

The point of the exercise
-------------------------

Cache the result, not merely the fact that a call happened.

Deduplicating the side effect is not enough here. Ask a live model the same
question twice at `TEMPERATURE=0.7` and you get two different answers, both
valid, both well-formed. A caller that retries a request whose response it never
received -- the classic "did my POST land?" -- must get back *the answer it
already paid for*, byte for byte, not a fresh sample from the model.

So all three of these must hold:

  * the same key returns the identical object, serialised identically
  * the second call does not reach upstream at all -- not "reached it and
    discarded the answer", not "reached it with a dedupe header". At all.
  * two concurrent calls with the same key produce ONE upstream call, and both
    callers receive the same stored result

That last one is the one a bare check-then-set fails. N simultaneous retries all
miss the empty cache at the same instant and all call upstream -- and this is
the realistic case, because duplicate requests arrive together precisely when
something upstream just retried. You need a per-key lock and a second check
inside it.

Storage
-------

`build_redis()` is written for you and returns Redis or fakeredis depending on
`USE_FAKEREDIS`. Store the serialised JSON, not the model object: an in-process
dict would pass the single-process test and fail the moment there are two
workers or a restart.

Serialise once, on write. Every later read then returns exactly those bytes,
which is what makes "byte-identical" true rather than merely "equal".
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Awaitable, Callable

from app.config import Settings, get_settings
from app.schema import TriageResult

# How long a cached result stays valid. Long enough to cover any realistic
# client retry window, short enough that the cache is not a database.
DEFAULT_TTL_S = 24 * 60 * 60

KEY_PREFIX = "triage:idem:"


def make_key(text: str, *, idem_key: str | None = None, salt: str = "") -> str:
    """The cache key for one logical request.

    An explicit `idem_key` from the caller wins: it is the caller telling us
    "these two requests are the same one", which is a stronger statement than
    identical bytes. Absent one, identical input text is the fallback identity.
    """
    # stage 5: hash idem_key when given, else the request text; prefix with
    # KEY_PREFIX. Include `salt` so a run at a different model or temperature
    # does not serve the previous run's cached answers.
    return KEY_PREFIX


class IdempotencyStore:
    """Redis-backed store of completed results, keyed by idempotency key."""

    def __init__(self, redis: Any, *, ttl_s: int = DEFAULT_TTL_S) -> None:
        self.redis = redis
        self.ttl_s = ttl_s

    async def get(self, key: str) -> TriageResult | None:
        """The stored result for `key`, or None."""
        # stage 5: read the key and validate it back into a TriageResult.
        return None

    async def put(self, key: str, result: TriageResult) -> None:
        """Store `result` under `key` with a TTL."""
        # stage 5: store result.model_dump_json() with an expiry of self.ttl_s.
        # Serialise once, here -- that is what makes replays byte-identical.

    async def get_or_call(
        self, key: str, operation: Callable[[], Awaitable[TriageResult]]
    ) -> tuple[TriageResult, bool]:
        """Return the cached result for `key`, or run `operation` and cache it.

        Returns `(result, cached)`. Callers need the flag: the spend ledger must
        not charge twice for one upstream call, and the API reports it as a header.
        """
        # stage 5: on a hit return (result, True) without calling `operation`.
        # On a miss, take a PER-KEY lock, check again inside it, then call
        # `operation`, store the result and return (result, False). Without the
        # lock, concurrent duplicates all miss the empty cache together and all
        # hit upstream -- which is exactly when duplicates happen.
        # test_idempotency.py fires 8 at once.
        return await operation(), False

    async def clear(self) -> None:
        """Drop every cached result. Tests only."""
        # stage 5: delete every key under KEY_PREFIX.


def build_redis(settings: Settings | None = None) -> Any:
    """A Redis client, real or fake. GIVEN -- you do not need to change this.

    `USE_FAKEREDIS=true` is the no-Docker path and the default: the tests must
    run with nothing installed but pip packages.
    """
    settings = settings or get_settings()
    if settings.use_fakeredis:
        import fakeredis.aioredis

        return fakeredis.aioredis.FakeRedis(decode_responses=True)
    import redis.asyncio

    return redis.asyncio.from_url(settings.redis_url, decode_responses=True)
