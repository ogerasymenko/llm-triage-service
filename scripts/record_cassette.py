"""Record request/response cassettes for offline replay.

Default mode drives the local mock; --real drives the actual API and needs a key.

The distinction is not cosmetic. A synthetic cassette has a real wire format --
the same JSON the SDK parses -- so it is fine for exercising a client or a replay
harness. Its answers come from keyword rules in mock/scenarios.py, so nothing
about model behaviour can be learned from one. Every cassette records which mode
produced it, so a consumer never has to guess.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import httpx

from app.schema import TriageResult, triage_result_json_schema

SAMPLES_DIR = REPO / "samples"
DEFAULT_OUT = REPO / "cassettes"
MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT = (
    "You are a bug triage assistant. Read the report and return a single JSON "
    "object matching the provided schema. Judge severity by user impact, name "
    "the one component most likely at fault, and give the repro steps you can "
    "actually see in the text. Set confidence to how sure you are; a vague or "
    "noisy report should score low. Return the JSON object and nothing else."
)


def output_config() -> dict:
    return {"format": {"type": "json_schema", "schema": triage_result_json_schema()}}


def request_body(text: str, *, temperature: float, max_tokens: int) -> dict:
    """The request we are recording. Kept in one place so both modes match."""
    return {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": text}],
        "output_config": output_config(),
    }


async def record_one(client: httpx.AsyncClient, path: Path, *, temperature: float,
                     max_tokens: int) -> dict:
    """One sample in, one cassette entry out."""
    text = path.read_text()
    body = request_body(text, temperature=temperature, max_tokens=max_tokens)

    started = time.monotonic()
    response = await client.post("/v1/messages", json=body)
    elapsed = time.monotonic() - started
    response.raise_for_status()
    payload = response.json()

    # Validate on the way in. A cassette holding output that does not satisfy
    # the contract is a trap for whoever replays it later.
    content = "".join(
        block.get("text", "") for block in payload.get("content", [])
        if block.get("type") == "text"
    )
    result = TriageResult.model_validate_json(content)

    return {
        "sample": path.name,
        "request": body,
        "response": payload,
        "parsed": json.loads(result.model_dump_json()),
        "latency_s": round(elapsed, 4),
    }


async def run(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = sorted(SAMPLES_DIR.glob("*.txt"))
    if not samples:
        print(f"no samples found in {SAMPLES_DIR}", file=sys.stderr)
        return 1

    if args.real:
        base_url = "https://api.anthropic.com"
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key or api_key == "not-a-real-key":
            print(
                "--real needs a genuine ANTHROPIC_API_KEY. Without one, run "
                "without --real to generate synthetic cassettes.",
                file=sys.stderr,
            )
            return 2
        if os.environ.get("ALLOW_REAL_API", "").lower() not in {"1", "true", "yes", "on"}:
            print(
                "--real sends real, billable traffic. Set ALLOW_REAL_API=true to "
                "confirm you meant it.",
                file=sys.stderr,
            )
            return 2
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        synthetic = False
    else:
        base_url = args.mock_url
        headers = {"x-api-key": "not-a-real-key", "anthropic-version": "2023-06-01"}
        synthetic = True

    label = "REAL API" if args.real else f"the mock at {base_url}"
    print(f"Recording {len(samples)} cassette(s) against {label}")
    if synthetic:
        print("  -> these will be SYNTHETIC: real wire format, invented answers")

    entries = []
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=60.0) as client:
        for path in samples:
            try:
                entry = await record_one(
                    client, path, temperature=args.temperature, max_tokens=args.max_tokens
                )
            except httpx.HTTPError as exc:
                print(f"  {path.name}: FAILED ({exc})", file=sys.stderr)
                return 1
            entries.append(entry)
            parsed = entry["parsed"]
            print(
                f"  {path.name}: {parsed['severity']:>8} / {parsed['component']:<14} "
                f"conf={parsed['confidence']}"
            )

    cassette = {
        "meta": {
            "synthetic": synthetic,
            "source": "anthropic-api" if args.real else "local-mock",
            "model": MODEL,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": os.environ.get("SEED", "1337") if synthetic else None,
            "git_commit": _git_commit(),
            "note": (
                "SYNTHETIC. Wire format is real; the answers are generated by "
                "keyword rules in mock/scenarios.py, not by a model. Do not draw "
                "conclusions about model behaviour from this file."
                if synthetic
                else "Recorded against the real Anthropic API."
            ),
        },
        "entries": entries,
    }

    name = "synthetic.json" if synthetic else "real.json"
    destination = out_dir / name
    if args.real and destination.exists():
        existing = json.loads(destination.read_text())
        if existing.get("meta", {}).get("synthetic"):
            print(f"refusing to overwrite synthetic {destination} from a real run",
                  file=sys.stderr)
            return 1
    destination.write_text(json.dumps(cassette, indent=2) + "\n")
    print(f"\nwrote {destination} ({len(entries)} entries, synthetic={synthetic})")
    return 0


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", action="store_true",
                        help="record against the real API (needs a key; costs money)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument("--mock-url", default="http://127.0.0.1:8081",
                        help="where the mock is listening (synthetic mode)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
