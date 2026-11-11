# ForgeCC CPU benchmarks

Build the benchmark target with:

```sh
cmake -S . -B build -DFORGECC_BUILD_BENCHMARKS=ON
cmake --build build --target forgecc_benchmarks
```

Run it from the repository root:

```sh
./build/benchmarks/forgecc_benchmarks --warmup 5 --iterations 30 \
  --output benchmarks/results/phase10_3_cpu_benchmarks.csv
```

Generate a Markdown report from saved results without rerunning benchmarks:

```sh
cmake --build build --target forgecc_report
./build/benchmarks/forgecc_report \
  --input benchmarks/results/phase10_3_cpu_benchmarks.csv \
  --output benchmarks/results/phase10_3_performance_report.md
```

The benchmark reports CSV rows for a straightforward C++ baseline, the
existing Graph IR emitter after DCE/CSE, the baseline Loop IR-generated
implementation, and tiled Loop IR variants for matmul. Compilation time and
generated artifact sizes are reported separately from the timed kernel calls.
Each row retains every raw measured sample as well as aggregate statistics.
The default inputs are deterministic and the measured output is consumed
through a checksum. The Phase 10.3 comparison uses five warm-up iterations
and thirty measured iterations unless overridden on the command line.

The report recomputes medians from the raw samples and defines speedup as
`baseline median / compared median`; values above 1.0 are faster. It also
shows percentage differences, observed time-per-work-unit, compilation time,
source size, and executable size. It does not collect hardware counters
automatically. On Linux systems where permissions allow it, `perf stat` can be
run separately; counter availability depends on the host kernel and CPU.

## Baseline matrix multiplication

The current baseline computes `C = A × B` for `A[M, K]` and `B[K, N]` with
three nested loops:

```text
for i in [0, M):
  for j in [0, N):
    C[i, j] = 0
    for k in [0, K):
      C[i, j] += A[i, k] * B[k, j]
```

Tensors use contiguous row-major storage. `A[i, k]` advances contiguously as
`k` changes, and each output row of `C` is written across columns. In contrast,
`B[k, j]` advances by one row stride as `k` changes while `j` remains fixed,
so a column access is generally non-contiguous and can have poorer spatial
locality. Repeated accesses to those columns may increase cache pressure as
matrix sizes grow. These are expected access-pattern effects; this project
does not claim cache-miss measurements unless hardware counters are collected
separately.

The baseline benchmark includes square and rectangular cases, including
`2x3 × 3x2`, `3x2 × 2x4`, `1x3 × 3x1`, and `4x3 × 3x5`.

## Tiled matrix multiplication

For matrix-multiplication workloads, the benchmark also emits opt-in tiled
Loop IR variants with tile sizes 8, 16, and 32. Tiling partitions the row,
column, and shared dimensions into smaller blocks. Within each block, the
inner column loop advances contiguously through rows of `B` and `C`, while the
tile boundaries use clipped bounds so rectangular and non-divisible shapes
remain valid.

The tile size controls the working-block dimensions. Small matrices may not
benefit because loop and setup overhead dominates; excessively large tiles can
also reduce locality. These are expectations about the access pattern, not
claims of measured cache behavior. Tiled rows in the CSV are named
`tiled-8`, `tiled-16`, and `tiled-32`, and their compilation and execution
statistics are reported alongside the existing implementations.
