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

if [[ $dry_run == 0 && ${FINE_ALPHA_CAFFEINATED:-0} != 1 ]] &&
  command -v caffeinate >/dev/null 2>&1; then
  export FINE_ALPHA_CAFFEINATED=1
  exec caffeinate -i bash "$script_dir/run_fine_alpha_sweeps.sh" "$steps"
fi

for N in 25 40 80; do
  for kappa in 1.26 2.56; do
    name="N${N}_sweep_alpha40_kappa${kappa}"
    config="$repo_dir/configs/mirror-risk/$name.json"
    endpoints="$results_dir/finite-n/$name"
    comparison="$results_dir/finite-n/${name}_comparison"
    echo "Running N=$N, kappa=$kappa"
    if [[ $dry_run == 1 ]]; then
      printf 'bash %q %q %q %q %q %q\n' \
        "$script_dir/run_finite_n_workflow.sh" "$config" "$endpoints" "$comparison" "$steps" both
    else
      bash "$script_dir/run_finite_n_workflow.sh" \
        "$config" "$endpoints" "$comparison" "$steps" both
    fi
  done
done
