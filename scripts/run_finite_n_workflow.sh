#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 CONFIG ENDPOINT_DIR COMPARISON_DIR [ADDITIONAL_STEPS] [mirror|gate|linear_gd|both]" >&2
  exit 2
fi

config=$1
endpoints=$2
comparison=$3
steps=${4:-100000}
method=${5:-mirror}
if [[ ! $steps =~ ^[1-9][0-9]*$ ]]; then
  echo "ADDITIONAL_STEPS must be a positive integer" >&2
  exit 2
fi
case $method in
  mirror|gate|linear_gd)
    runs_dir="runs/$method"
    ;;
  both)
    runs_dir="runs"
    ;;
  *)
    echo "METHOD must be mirror, gate, linear_gd, or both" >&2
    exit 2
    ;;
esac

simulation=$(python - "$config" <<'PY'
import sys
from mirror_risk.experiment import ExperimentConfig

print(ExperimentConfig.from_json(sys.argv[1]).output_directory)
PY
)

if python - "$config" <<'PY'
import sys
from pathlib import Path
from mirror_risk.experiment import ExperimentConfig, _problem_path
from mirror_risk.problem import ProblemSpec

config = ExperimentConfig.from_json(sys.argv[1])
root = Path(config.output_directory)
complete = (root / "experiment_config.json").is_file() and all(
    _problem_path(root, ProblemSpec(config.N, alpha, config.sigma, config.r, seed)).is_file()
    for alpha in config.alpha_values for seed in config.seeds
)
sys.exit(0 if complete else 1)
PY
then
  echo "Datasets already present; skipping preparation."
else
  echo "Preparing datasets..."
  python -m mirror_risk.cli "$config" --prepare-only
fi

if [[ -e $endpoints ]]; then
  if ! python - "$config" "$endpoints" <<'PY'
import sys
from pathlib import Path
from mirror_risk.experiment import ExperimentConfig

config = ExperimentConfig.from_json(sys.argv[1])
root = Path(sys.argv[2])
methods = {item.method for item in config.dynamics}
required = []
if methods & {"mirror", "gate"}:
    required += ["endpoints.csv", "summary_by_alpha.csv"]
if "linear_gd" in methods:
    required += ["linear_endpoints.csv", "linear_summary_by_alpha.csv"]
sys.exit(0 if all((root / name).is_file() for name in required) else 1)
PY
  then
    echo "Endpoint directory exists but is incomplete: $endpoints" >&2
    exit 1
  fi
  echo "Endpoint results already present; skipping solve: $endpoints"
else
  echo "Solving finite-N endpoints..."
  python -m mirror_risk.finite_n solve "$config" "$endpoints"
fi

echo "Running or resuming $method dynamics (up to $steps additional steps per incomplete run)..."
if [[ $method == both ]]; then
  python -m mirror_risk.cli "$config" --resume --additional-steps "$steps"
else
  python -m mirror_risk.cli "$config" --resume --additional-steps "$steps" --method "$method"
fi

if [[ -e $comparison ]]; then
  if [[ ! -f $comparison/comparison_runs.csv || ! -f $comparison/comparison_by_alpha.csv ]]; then
    echo "Comparison directory exists but is incomplete: $comparison" >&2
    exit 1
  fi
  if [[ -d $simulation/$runs_dir ]] &&
    [[ -n $(find "$simulation/$runs_dir" -name '*.npz' -newer "$comparison/comparison_runs.csv" -print -quit) ]]; then
    echo "Comparison exists but predates some checkpoints; use a new comparison directory: $comparison"
  else
    echo "Comparison results already present; skipping compare: $comparison"
  fi
else
  echo "Comparing $method checkpoints with paired endpoints..."
  if [[ $method == both ]]; then
    python -m mirror_risk.finite_n compare "$config" "$endpoints" "$comparison"
  else
    python -m mirror_risk.finite_n compare "$config" "$endpoints" "$comparison" --method "$method"
  fi
fi
