# Resistor linear perceptron

This repository contains the numerical calculations for a resistor implementation
of a linear perceptron.  We study the resulting fixed
points and learning dynamics in a Gaussian teacher–student setting, and compare
finite-size realizations with the predictions in the high-dimensional limit.

The code includes the fixed-point minimizations, mirror descent, gradient descent
in the gate variables, and unconstrained linear regression. All methods use the
same datasets, so their solutions can be compared realization by realization.

## Installation

Use Python 3.10 or newer. From the repository directory, run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook]"
```

## Numerical results

The main finite-size calculations use $N=80$, $r=\sigma=1$, and 20 realizations
at each of 40 values of $\alpha=N/P$. We compare capacities $\kappa=1.26,2.56$
with the unconstrained limit $\kappa=\infty$. The configurations also include
$N=25,40$ for studying finite-size effects.

[REPRODUCING.txt](REPRODUCING.txt) gives the commands to generate these datasets,
solve their fixed points, and run the learning dynamics. Numerical outputs are
written to `../results/public-release/`, outside the repository.

After generating the three $N=80$ endpoint ensembles, open

```bash
python -m jupyterlab notebooks/results.ipynb
```

Run the cells in order. The notebook makes one figure showing training loss,
risk, self-overlap, and teacher overlap for the three capacities. Points are
individual realizations, colored by their fixed-point regime; black curves are
the thermodynamic predictions.

For a short run of the simulation routines, use

```bash
bash scripts/run_finite_n_workflow.sh \
  configs/mirror-risk/smoke_test.json \
  ../results/public-release/finite-n/smoke-endpoints \
  ../results/public-release/finite-n/smoke-comparison-round-1 \
  500 both
```

This runs mirror and gate dynamics on four small datasets. 

## Author and license

Marcelo Guzman. Released under the [MIT license](LICENSE).
