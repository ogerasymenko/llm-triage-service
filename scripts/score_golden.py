"""Score the golden set across seeds, against the in-process mock.

    make golden                       # 20 seeds at TEMPERATURE=0.7
    make golden SEEDS=40 MODE=bad_content

Stage 4 asks for a threshold set by measuring. One run measures one seed; the
mock's readings of a hedged report vary with the seed the way reruns against a
live model vary, so the number to look at is the spread. The worst clean seed
and the best bad_content seed are the two edges a threshold has to sit between.
The suite itself runs one seed, SEED=1337, so that one is reported on its own:
a bar that clears the sweep but not the suite's seed fails in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.config as config_module  # noqa: E402
from app.service import TriageService  # noqa: E402
from tests.conftest import SAMPLES_DIR, MockServer, _free_port  # noqa: E402
from tests.stages.test_golden import CASES, grade  # noqa: E402

SUITE_SEED = 1337  # tests/conftest.py's default


async def score(seed: int, mode: str, temperature: str) -> tuple[float, list[str]]:
    os.environ.update(
        SEED=str(seed),
        TEMPERATURE=temperature,
        FAULT_RATE="1.0" if mode == "bad_content" else "0.0",
        FAULT_MODE="bad_content" if mode == "bad_content" else "all",
    )
    config_module.reset_settings()
    service = TriageService(settings=config_module.get_settings())
    failed: list[str] = []
    try:
        for case in CASES:
            result, _ = await service.triage((SAMPLES_DIR / case["file"]).read_text())
            ok, _ = grade(result, case["accept"])
            if not ok:
                failed.append(case["id"])
    finally:
        await service.aclose()
    return 1 - len(failed) / len(CASES), failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--mode", choices=["clean", "bad_content"], default="clean")
    parser.add_argument("--temperature", default="0.7")
    args = parser.parse_args()

    server = MockServer(_free_port())
    server.start()
    os.environ.update(
        PROVIDER="anthropic",
        PROVIDER_BASE_URL=server.base_url,
        ANTHROPIC_API_KEY="not-a-real-key",
        USE_FAKEREDIS="true",
        MAX_RETRIES="5",
    )
    try:
        runs = [asyncio.run(score(s, args.mode, args.temperature)) for s in range(1, args.seeds + 1)]
        suite_score, suite_missed = asyncio.run(score(SUITE_SEED, args.mode, args.temperature))
    finally:
        server.stop()

    spread = collections.Counter(round(value, 1) for value, _ in runs)
    misses = collections.Counter(case for _, failed in runs for case in failed)
    print(f"golden set, {args.mode}, TEMPERATURE={args.temperature}, seeds 1-{args.seeds}")
    print("  score -> seeds:", ", ".join(f"{k:.1f}: {v}" for k, v in sorted(spread.items())))
    print(f"  worst {min(v for v, _ in runs):.1f}, best {max(v for v, _ in runs):.1f}")
    print("  cases missed, and on how many seeds:", dict(sorted(misses.items())) or "none")
    print(f"  seed {SUITE_SEED} (the one the test suite runs): {suite_score:.1f}"
          + (f", missed {suite_missed}" if suite_missed else ""))


if __name__ == "__main__":
    main()
