#!/usr/bin/env bash
set -euo pipefail

dry_run=0
if [[ ${1:-} == --dry-run ]]; then
  dry_run=1
  shift
fi
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--dry-run] [ADDITIONAL_STEPS]" >&2
  exit 2
fi
steps=${1:-1000000}
if [[ ! $steps =~ ^[1-9][0-9]*$ ]]; then
  echo "ADDITIONAL_STEPS must be a positive integer" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/.." && pwd)
results_dir="$(cd "$repo_dir/.." && pwd)/results/public-release"

if [[ $dry_run == 0 && ${LINEAR_GD_CAFFEINATED:-0} != 1 ]] &&
  command -v caffeinate >/dev/null 2>&1; then
  export LINEAR_GD_CAFFEINATED=1
  exec caffeinate -i bash "$script_dir/run_linear_gd_baselines.sh" "$steps"
fi

for N in 25 40 80; do
  name="N${N}_sweep_alpha40_linear_gd"
  config="$repo_dir/configs/linear-gd/$name.json"
  endpoints="$results_dir/finite-n/$name"
  comparison="$results_dir/finite-n/${name}_comparison"
  echo "Running linear GD for N=$N"
  if [[ $dry_run == 1 ]]; then
    printf 'bash %q %q %q %q %q %q\n' \
      "$script_dir/run_finite_n_workflow.sh" "$config" "$endpoints" "$comparison" "$steps" linear_gd
    continue
  fi

  python -m mirror_risk.cli "$config" --prepare-only
  python - "$config" \
    "$repo_dir/configs/mirror-risk/N${N}_sweep_alpha40_kappa1.26.json" \
    "$repo_dir/configs/mirror-risk/N${N}_sweep_alpha40_kappa2.56.json" <<'PY'
import sys
from pathlib import Path
from mirror_risk.experiment import ExperimentConfig, _problem_path
from mirror_risk.problem import LinearRegressionProblem, ProblemSpec

baseline, *sources = (ExperimentConfig.from_json(path) for path in sys.argv[1:])
for source in sources:
    if (source.N, source.alpha_values, source.sigma, source.r, source.seeds) != (
        baseline.N, baseline.alpha_values, baseline.sigma, baseline.r, baseline.seeds
    ):
        raise ValueError("Linear GD and physical configurations specify different datasets")

for alpha in baseline.alpha_values:
    for seed in baseline.seeds:
        spec = ProblemSpec(baseline.N, alpha, baseline.sigma, baseline.r, seed)
        paths = [_problem_path(Path(config.output_directory), spec)
                 for config in (baseline, *sources)]
        fingerprints = [LinearRegressionProblem.load(path).fingerprint() for path in paths]
        if len(set(fingerprints)) != 1:
            raise ValueError(f"Dataset fingerprint mismatch for P={spec.P}, seed={seed}")
print(f"Verified {len(baseline.alpha_values) * len(baseline.seeds)} paired datasets")
PY
  bash "$script_dir/run_finite_n_workflow.sh" \
    "$config" "$endpoints" "$comparison" "$steps" linear_gd
done
