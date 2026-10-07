# Learning from Cancelled Requests

A reproducible implementation of cancellation-aware residual estimation (CARE), with seeded simulation results and 480,000 request-level HTTP measurements. The experiments evaluate how cancellation bounds and randomized completion audits support decisions about delayed replica requests.

## Quick start

Use Python 3.12 and a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --mode verify
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Verification checks the archive's SHA-256 manifest, runs ten mathematical and queueing tests, reconstructs every HTTP run summary from its request-level measurements, and writes `verification.json` with the main numerical comparisons. The recorded measurements and simulation summaries reproduce the reported values without running new network experiments.

## Repeat the experiments

```sh
python reproduce.py --mode simulation --workers 8
python reproduce.py --mode http
```

The simulation command regenerates all estimation, adaptive-policy, sensitivity, and queueing experiments into `recomputed/`, then compares every table entry with the reference CSV files using absolute and relative tolerances of 1e-9. The HTTP command starts two local replica processes, collects 120 new runs, and writes their measurements into `recomputed/http/`. New HTTP response times vary with scheduling, transport, and host load; the included recorded data support exact reanalysis. Each HTTP run uses 4,000 requests and the complete collection takes approximately 18 minutes in the recorded environment. The simulator uses eight independent worker processes by default. Reducing `--workers` reduces concurrent memory use. The completed experiments used 12 available CPU cores and 18 GB of system memory.

For a short transport check:

```sh
python src/http_experiment.py --quick --out recomputed/http-smoke
```

Optional accelerator agreement check:

```sh
python -m pip install -r requirements-gpu.txt
python reproduce.py --mode accelerator
```

The accelerator check evaluates 40 million candidate losses with float32 tensors and compares their means with float64 CPU results. It chooses an available MPS or CUDA backend and otherwise runs on the CPU. Primary statistical estimates use float64 CPU arrays.

## Files and outputs

| File | Purpose |
| --- | --- |
| `reproduce.py` | Verification, numerical reanalysis, experiment orchestration, and output comparisons. |
| `config.json` | Reference design, seeds, units, candidate delays, prices, probabilities, and sample sizes. |
| `src/core.py` | Duration generators, observable cancellation rectangles, loss bounds, CARE, audit probabilities, and Kaplan–Meier comparator. |
| `src/run_experiments.py` | Estimation, adaptive control, and sensitivity trials across independent seeds. |
| `src/queue_stress.py` | Event-driven two-server FCFS queue model with delayed cancellation and completion audits. |
| `src/http_experiment.py` | Two local asynchronous HTTP replicas, controlled delays, read/digest payloads, and measured cancellation. |
| `src/accelerator_check.py` | Optional CPU/accelerator numerical agreement test. |
| `tests/test_core.py` | Loss intervals, cap ties, accounting, hidden-outcome isolation, unbiasedness, and non-identification checks. |
| `tests/test_queue.py` | Hand-calculated cancellation/audit cases and an M/M/1 analytical reference. |
| `results/estimation.csv` | Estimator error for every family, sample size, seed, and method. |
| `results/policies.csv` | Per-seed adaptive-policy outcomes. |
| `results/curves.csv` | Block-level controller trajectories. |
| `results/sensitivity.csv` | Policy outcomes across audit multipliers and work prices. |
| `results/queue.csv` | Queueing outcomes across load, cancellation lag, and policy. |
| `results/http/*.csv.gz` | Request-level logs for all 120 recorded HTTP runs. |
| `results/http/summary.csv` | Per-run measured latency, occupancy, and audit summaries. |
| `results/*manifest.json` | Execution counts, timings, and software versions. |
| `results/accelerator.json` | Recorded tensor agreement errors and execution backend. |
| `checksums.sha256` | Integrity hashes of the distributed files. |
| `CITATION.cff` | Machine-readable citation metadata. |
| `LICENSE` | MIT terms for the code and generated data. |

The parameters in `config.json` document the reference design implemented in the experiment modules. All seeds are public. Main seeds are disjoint from feasibility-pilot seeds. Every policy sees the same latent request pairs within a simulation replicate. Estimator comparisons share the same audit draws. A request paired with eight candidate policies contributes eight policy-request evaluations, not eight physical network requests.

## Data dictionary

All duration, latency, occupancy, and cancellation-lag columns use milliseconds. A delay of 300 disables hedging in the capped simulation. `miss`, `audit`, and `launch` are fractions in summary CSV files; request-level `miss`, `audit`, and `launched` are binary. `work` is cumulative replica occupancy including audits. `backup` is ordinary duplicate occupancy. `audit_work` is additional occupancy retained or launched to reveal a missing completion. `operational_loss` adds the audit-work charge to deadline-and-backup loss.

For HTTP logs, `service` sums actual server attempt occupancy, `secondary` records secondary occupancy, `probability` is the logged audit propensity, `dispatch_lag` measures lateness relative to a scheduled arrival, and `cancel_lag` measures cancellation acknowledgement delay after the first response. The columns `z0` through `z7` store the loss estimates in candidate-delay order. `primary_latency` is present when the primary completes; cancelled primary attempts can have an empty value. `primary_completed` identifies a primary winner. Request identifiers are generated integers; server addresses are loopback only. Both workloads are generated, and the injected waits are specified by the duration constructors.

The six duration families are independent stragglers, correlated lognormal pairs, shared shocks, asymmetric replicas, Pareto tails, and a four-regime switching stream. They provide known counterfactual ground truth. The queue and HTTP experiments separately test operational effects beyond the exogenous-pair model. Queue delays are fitted on independent calibration data and then frozen; a uniform 2% audit variant stresses measurement overhead. The HTTP client schedules Poisson arrivals within blocks and drains each block before updating.

## Interpretation

The estimator assumes that changing a backup's start time does not change its potential duration. It permits dependence between the two replica durations. The queueing study intentionally tests outside this assumption. Audit percentages and work overhead are different quantities: a small number of retained slow attempts can consume substantial capacity. The reference outputs report deadline failures, tail latency, work, and audit cost separately to preserve these trade-offs.

## Citation and license

Use the metadata in `CITATION.cff` to cite this release. The implementation and generated measurements are distributed under the MIT License.
