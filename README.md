# forgecc-tensor-compiler
A small ahead-of-time compiler for tensor expressions, written in C++20. It parses a simple language, infers shapes, builds an SSA graph IR, and runs passes like constant folding, operator fusion, and buffer reuse. It then generates C++ and CUDA code, with benchmarks comparing fused and unfused output.
