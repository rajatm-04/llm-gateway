"""Validate the pilot files and dry-run behavior without provider calls."""

import argparse
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_splits_and_schema():
    from gateway.models.schemas import ChatCompletionRequest
    rows = [json.loads(line) for line in (ROOT / "evaluation/requests.jsonl").read_text().splitlines()]
    assert len(rows) == 30
    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["split"] for row in rows} == {"dev", "test"}
    for row in rows:
        messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
        ChatCompletionRequest(messages=messages)
        assert row["rubric"]


@pytest.mark.asyncio
async def test_evaluation_dry_run_no_results_or_providers(config, tmp_path, capsys):
    script = runpy.run_path(str(ROOT / "scripts/evaluate_routing.py"))
    run = script["run"]
    run.__globals__["settings"] = config

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not construct providers")

    run.__globals__["ProviderRegistry"] = forbidden
    output = tmp_path / "not-created"
    await run(argparse.Namespace(
        dataset=ROOT / "evaluation/requests.jsonl", split="dev", limit=2,
        max_output_tokens=32, allow_paid_calls=False, max_premium_calls=2,
        output_dir=output, deployment_notes="test",
    ))
    assert not output.exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_rewrite_development_dataset():
    from gateway.models.schemas import ChatCompletionRequest

    script = runpy.run_path(str(ROOT / "scripts/evaluate_routing.py"))
    path = ROOT / "evaluation/rewrite_dev.jsonl"
    rows = script["load_cases"](path, "dev")
    assert len(rows) == 10
    assert len({row["id"] for row in rows}) == 10
    assert not script["load_cases"](path, "test")
    original = script["load_cases"](ROOT / "evaluation/requests.jsonl", "all")
    assert not {row["id"] for row in rows} & {row["id"] for row in original}
    for row in rows:
        ChatCompletionRequest(messages=[{"role": "user", "content": row["prompt"]}])
        assert row["rubric"]


def test_targeted_case_selection():
    script = runpy.run_path(str(ROOT / "scripts/evaluate_routing.py"))
    select = script["select_cases"]
    dataset = ROOT / "evaluation/requests.jsonl"
    cases = select(dataset, "dev", 2, ["d14", "d13"])
    assert [case["id"] for case in cases] == ["d13", "d14"]
    assert [case["id"] for case in select(dataset, "dev", 1)] == ["d01"]


@pytest.mark.parametrize("ids,limit", [
    (["d13", "missing"], 2),
    (["t01"], 1),
    (["d13", "d13"], 2),
    (["d13", "d14"], 1),
])
def test_invalid_case_selection(ids, limit):
    script = runpy.run_path(str(ROOT / "scripts/evaluate_routing.py"))
    with pytest.raises(ValueError):
        script["select_cases"](ROOT / "evaluation/requests.jsonl", "dev", limit, ids)


def test_report_does_not_count_ungraded_as_pass(tmp_path, capsys):
    row = {"id": "one", "selected_tier": "local", "routing_ms": 1, "responses": {
        tier: {"ok": True, "latency_ms": 10, "prompt_tokens": 2, "completion_tokens": 3}
        for tier in ("local", "premium")
    }}
    (tmp_path / "results.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "grades.csv").write_text("id,local_acceptable,premium_acceptable,notes\none,,,\n")
    script = runpy.run_path(str(ROOT / "scripts/report_evaluation.py"))
    script["report"](tmp_path)
    text = capsys.readouterr().out
    assert "Fully graded successful pairs: 0" in text
    assert "acceptable=not graded" in text


def test_report_review_and_benchmark_counts(tmp_path, capsys):
    import csv

    labels = [
        ("both", "yes", "yes"), ("loss", "no", "yes"),
        ("local_only", "yes", "no"), ("neither", "no", "no"),
        ("pending", "review", "yes"), ("blank", "", "yes"),
    ]
    rows = [{"id": name, "selected_tier": "local", "routing_ms": 1, "responses": {
        tier: {"ok": True, "latency_ms": 10, "prompt_tokens": 2, "completion_tokens": 3}
        for tier in ("local", "premium")
    }} for name, _, _ in labels]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with (tmp_path / "grades.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "local_acceptable", "premium_acceptable", "notes"])
        for name, local, premium in labels:
            writer.writerow([name, local, premium, ""])
    report = runpy.run_path(str(ROOT / "scripts/report_evaluation.py"))["report"]
    report(tmp_path)
    text = capsys.readouterr().out
    assert "Fully graded successful pairs: 4" in text
    assert "needing review: 1; with missing grades: 1" in text
    assert "both acceptable=1; premium only=1; local only=1; neither acceptable=1" in text
    assert "benchmark: 1/4 resolved local-selected pairs" in text


def test_report_rejects_invalid_grade(tmp_path):
    (tmp_path / "results.jsonl").write_text("")
    (tmp_path / "grades.csv").write_text("id,local_acceptable,premium_acceptable,notes\none,maybe,yes,\n")
    report = runpy.run_path(str(ROOT / "scripts/report_evaluation.py"))["report"]
    with pytest.raises(ValueError, match="Invalid local grade"):
        report(tmp_path)
