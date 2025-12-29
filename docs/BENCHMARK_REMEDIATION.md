# Benchmark Remediation Loop

This repo includes a deterministic benchmark harness that exercises QBM via the FastAPI app using the canonical proof contract (`schemas/proof_response_v2.py`). The benchmark is proof-first: scoring is driven by Truth Layer + provenance correctness, not narrative quality.

## Run The Benchmark

- Smoke (fast Tier-A style run; 2 questions per section):
  - `python scripts/run_qbm_benchmark.py --dataset data/benchmarks/qbm_legendary_200.v1.jsonl --smoke --proof-only --ci`
- Full (all 200 questions):
  - `python scripts/run_qbm_benchmark.py --dataset data/benchmarks/qbm_legendary_200.v1.jsonl --proof-only`
- Focused iteration:
  - One item: `python scripts/run_qbm_benchmark.py --dataset data/benchmarks/qbm_legendary_200.v1.jsonl --id A01 --proof-only`
  - One section: `python scripts/run_qbm_benchmark.py --dataset data/benchmarks/qbm_legendary_200.v1.jsonl --section A --proof-only`

## Outputs

Each run writes to `reports/benchmarks/`:

- Machine JSON: `reports/benchmarks/qbm_legendary_200_<timestamp>.json`
- Human Markdown: `reports/benchmarks/qbm_legendary_200_<timestamp>.md`
- Triage CSV: `reports/benchmarks/qbm_legendary_200_<timestamp>.csv`

For every `FAIL`, a per-item remediation artifact is also written:

- `reports/benchmarks/failures/<timestamp>/<id>.json`

Each failure artifact includes:
- The benchmark question + expected capability spec
- The exact request payload and response
- Schema validation issues (if any)
- Deterministic scoring failure reasons/tags/metrics
- Minimal repro snippets (curl + PowerShell)

## Remediation Loop (No Manual Output Patching)

1. Run `--smoke --proof-only --ci` to get a fast signal.
2. Open the Markdown report and identify the top deterministic failure reasons.
3. For a specific failed item, open its failure artifact JSON and use the included repro snippet.
4. Fix the methodology gap in code and/or Truth Layer artifacts (retrieval, entity resolution, provenance emission, graph evidence, source coverage).
5. Re-run the benchmark to verify the fix and track pass-rate improvement.

Do not manually edit benchmark outputs. Reports must remain pure, reproducible derivatives of the system.

