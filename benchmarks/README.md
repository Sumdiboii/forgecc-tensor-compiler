# ForgeCC CPU benchmarks

Build the benchmark target with:

```sh
cmake -S . -B build -DFORGECC_BUILD_BENCHMARKS=ON
cmake --build build --target forgecc_benchmarks
```

Run it from the repository root:

```sh
./build/benchmarks/forgecc_benchmarks --warmup 2 --iterations 7 \
  --output benchmarks/results/cpu_benchmarks.csv
```

Generate a Markdown report from saved results without rerunning benchmarks:

```sh
cmake --build build --target forgecc_report
./build/benchmarks/forgecc_report \
  --input benchmarks/results/cpu_benchmarks.csv \
  --output benchmarks/results/performance_report.md
```

The benchmark reports CSV rows for a straightforward C++ baseline, the
existing Graph IR emitter after DCE/CSE, and the Loop IR-generated
implementation. Compilation time and generated artifact sizes are reported
separately from the timed kernel calls. Each row retains one raw measured
sample as well as the aggregate statistics. The default inputs are
deterministic and the measured output is consumed through a checksum.

The report recomputes medians from the raw samples and defines speedup as
`baseline median / compared median`; values above 1.0 are faster. It also
shows percentage differences, observed time-per-work-unit, compilation time,
source size, and executable size. It does not collect hardware counters
automatically. On Linux systems where permissions allow it, `perf stat` can be
run separately; counter availability depends on the host kernel and CPU.
