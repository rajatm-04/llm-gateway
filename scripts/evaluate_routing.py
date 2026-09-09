"""Bounded paired-model pilot. Dry-run by default; never calls the gateway/cache.

Run from the repository root with uv. Live mode requires --allow-paid-calls.
Results contain prompts and answers; use public/synthetic evaluation data only.
"""

import argparse
import asyncio
import csv
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from gateway.config import settings
from gateway.models.schemas import ChatCompletionRequest
from gateway.providers.registry import ProviderRegistry
from gateway.router.model_router import ModelRouter

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def load_cases(path: Path, split: str):
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset IDs must be unique")
    return [case for case in cases if split == "all" or case["split"] == split]


def select_cases(path: Path, split: str, limit: int, case_ids: list[str] | None = None):
    """Select within a split, preserving dataset order and rejecting silent omissions."""
    cases = load_cases(path, split)
    if case_ids is not None:
        requested = set(case_ids)
        if len(requested) != len(case_ids):
            raise ValueError("--case-ids must not contain duplicates")
        missing = requested - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"Case IDs not found in selected split: {', '.join(sorted(missing))}")
        if len(requested) > limit:
            raise ValueError("--limit must cover all explicitly selected case IDs")
        cases = [case for case in cases if case["id"] in requested]
    return cases[:limit]


async def run(args):
    cases = select_cases(args.dataset, args.split, args.limit, getattr(args, "case_ids", None))
    if not cases:
        raise ValueError("No evaluation cases selected")
    if args.max_output_tokens > min(settings.local_max_output_tokens, settings.premium_max_output_tokens):
        raise ValueError("Output limit must fit both configured providers")
    router = ModelRouter(settings)
    prepared = []
    for case in cases:
        messages = case.get("messages") or [{"role": "user", "content": case["prompt"]}]
        request = ChatCompletionRequest(messages=messages, temperature=0.0, max_tokens=args.max_output_tokens)
        started = time.perf_counter()
        decision = router.select(request)
        routing_ms = (time.perf_counter() - started) * 1000
        # Check both explicit routes before authorizing any generation.
        router.select(request, "local")
        router.select(request, "premium")
        prepared.append((case, request, decision, routing_ms))

    print(f"Selected {len(prepared)} cases. Live run: {len(prepared)} local + {len(prepared)} premium calls.")
    print(f"Requested maximum output tokens per call: {args.max_output_tokens}. Automatic retries: 0.")
    print("This is a request/token bound, NOT a dollar cap. Input tokens and provider billing also matter.")
    if not args.allow_paid_calls:
        for case, _, decision, _ in prepared:
            print(f"{case['id']}: {decision.tier} ({decision.reason})")
        print("DRY RUN: no network calls or result files. Add --allow-paid-calls only after review.")
        return
    if len(prepared) > args.max_premium_calls:
        raise ValueError("Selected cases exceed --max-premium-calls; reduce --limit")
    if not settings.openai_api_key:
        raise ValueError("Premium API key is not configured")

    output = args.output_dir or ROOT / "evaluation" / "results" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing run.
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "local_model": settings.ollama_model, "premium_model": settings.openai_model,
        "policy_version": router.policy_version, "profile": router.profile.model_dump(),
        "split": args.split, "case_ids": [case["id"] for case, *_ in prepared],
        "temperature": 0, "max_output_tokens": args.max_output_tokens,
        "max_premium_calls": args.max_premium_calls, "retries": 0,
        "deployment_notes": args.deployment_notes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = ProviderRegistry(settings)
    try:
        with (output / "results.jsonl").open("x", encoding="utf-8") as results, \
             (output / "grades.csv").open("x", newline="", encoding="utf-8") as grade_file:
            grades = csv.DictWriter(grade_file, fieldnames=["id", "local_acceptable", "premium_acceptable", "notes"])
            grades.writeheader()
            for case, request, decision, routing_ms in prepared:
                row = {"id": case["id"], "split": case["split"], "category": case["category"],
                       "messages": [m.model_dump() for m in request.messages], "rubric": case["rubric"],
                       "selected_tier": decision.tier, "reason": decision.reason,
                       "routing_ms": routing_ms, "responses": {}}
                failed = False
                # Local first: stop before spending on premium if local service is broken.
                for tier in ("local", "premium"):
                    model = settings.ollama_model if tier == "local" else settings.openai_model
                    started = time.perf_counter()
                    try:
                        response = await registry.get(tier).generate(
                            messages=request.messages, model=model, temperature=0.0,
                            max_tokens=args.max_output_tokens,
                        )
                        row["responses"][tier] = {
                            "ok": True, **asdict(response), "latency_ms": (time.perf_counter() - started) * 1000,
                        }
                    except Exception as error:  # noqa: BLE001 -- save partial results and stop on any provider failure.
                        # Raw errors can contain credentials/provider response bodies: only record type.
                        row["responses"][tier] = {
                            "ok": False, "error_type": type(error).__name__,
                            "latency_ms": (time.perf_counter() - started) * 1000,
                        }
                        failed = True
                        break
                results.write(json.dumps(row, ensure_ascii=False) + "\n")
                results.flush()
                grades.writerow({"id": case["id"], "local_acceptable": "", "premium_acceptable": "", "notes": ""})
                grade_file.flush()
                print(f"{case['id']}: {'ERROR; stopping' if failed else 'saved'}")
                if failed:
                    break
    finally:
        await registry.close()
    print(f"Results: {output}")
    print("Review using evaluation/grading_policy.md and case rubrics; fill grades.csv with yes/no/review. Blanks are ungraded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "requests.jsonl")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--limit", type=positive_int, default=10)
    parser.add_argument("--case-ids", nargs="+", help="Run only these IDs within the selected split, e.g. d13 d14")
    parser.add_argument("--max-output-tokens", type=positive_int, default=256)
    parser.add_argument("--max-premium-calls", type=positive_int, default=10)
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--deployment-notes", default="Unspecified: record hardware, quantization and model digest")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (ValueError, OSError) as error:
        # These are local validation/file errors; provider errors are sanitized inside run().
        parser.exit(2, f"Evaluation setup failed: {error}\n")


if __name__ == "__main__":
    main()
