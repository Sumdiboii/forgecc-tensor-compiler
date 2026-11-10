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

The benchmark reports CSV rows for a straightforward C++ baseline, the
existing Graph IR emitter after DCE/CSE, and the Loop IR-generated
implementation. Compilation time and generated artifact sizes are reported
separately from the timed kernel calls. Each row retains one raw measured
sample as well as the aggregate statistics. The default inputs are
deterministic and the measured output is consumed through a checksum.
