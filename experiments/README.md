# Optional routing evaluation

The files in this directory are optional quality-evaluation tooling, not part of
the gateway runtime. The synthetic datasets and grading policy live under
`experiments/evaluation/`; the paired provider runner and report generator are
`experiments/evaluate_routing.py` and `experiments/report_evaluation.py`.

The runner is a dry run by default. To preview routing decisions:

```bash
uv run python experiments/evaluate_routing.py --split dev --limit 10
```

Live provider calls require the explicit `--allow-paid-calls` flag. Use only
public or synthetic prompts, review the grading policy before grading responses,
and keep evaluation outputs in the ignored `experiments/evaluation/results/`
directory. Evaluation does not affect gateway routing or production behavior.
