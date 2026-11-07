# brainfah: a fast brainfuck compiler for Apple Silicon

A single-file (`main.cpp`, C++23) optimizing brainfuck compiler that turns a program into native code and runs it
immediately. It is built for **minimum total time from `./a.out program.bf` to the last byte of output**: cold start
(first run, nothing cached) matters as much as peak speed.

```sh
clang++ main.cpp -O3 -std=c++23
./a.out tests/mandelbrot.bf
./a.out tests/awib-0.4.bf < tests/awib-0.4.bf
```

No flags are needed: the defaults are the tuned optimum.

## Results

Mac mini M4, Homebrew LLVM (clang 23), hyperfine, cache cleared before each cold run.

| Program | Cold (first run) | Warm (cached) |
|---|---|---|
| mandelbrot.bf (Erik Bosman) | **318 ms** | **155 ms** |
| hanoi.bf | 12 ms | 3 ms |
| awib-0.4.bf (self-compile) | 18 ms | 5 ms |
| factor.b (matslina) | 540 ms | 454 ms |
| long.b / bench.b / dbfi.b | 6–38 ms | 3–12 ms |

For reference, the first working version took 832–888 ms cold on mandelbrot. Warm mandelbrot runs at IPC ≈ 4.8 with
0.46% branch mispredictions and almost no L1D misses (measured with Instruments, CPU Counters), so it is limited by
instruction count and branch density rather than by memory or the predictor.

## How it works

```
source ─► optimizing IR ─► compile-time evaluation ─► C code ─► clang -O1/-O3 ─► native .o ─► run
              │                                                                      ▲
              └────────────── IR interpreter starts immediately on a cache miss ─────┘
```

1. **IR optimizer** (one pass): deferred pointer shifts, add/set folding, known-value tracking, loops → multiply-add /
   clear / `if`, closed forms for counted loops, scan loops (`[>]`, `[>>>]`) with unrolled multi-exit scalar scans and
   NEON for stride ±1, unbalanced loops, nested `while` → `if` collapsing.
2. **Partial evaluation**: the input-independent prefix of the program is executed at compile time.
3. **C code generation**: one function with register promotion of cell groups inside loops, strided loops that carry
   values between iterations in registers (split into segments for huge programs to keep clang time linear).
4. **Two tiers on a cache miss**: an IR interpreter starts at once while clang builds in the background. Short programs
   finish in the interpreter and never wait for clang. If the build finishes first, the program restarts natively
   (output is buffered, so the restart is invisible). The first native build uses `-O1`
   (about 90 ms cheaper to compile); a run longer than 0.1 s then triggers a detached `-O3` build that later runs use.
5. **Own Mach-O object loader** (macOS/arm64): clang only emits a relocatable `.o` (`-c`, no link, no code signing) and
   the program maps it itself (`MAP_JIT`, arm64 relocations, libc through `dlsym` stubs). The first `dlopen`/`exec`
   of *new* signed code costs about 180 ms on macOS (code-signature validation), which this avoids.
6. **Safety**: the tape is `mmap`ed with guard pages; out-of-bounds access is caught via `SIGSEGV` and reported as an
   error. Cache entries are written atomically and keyed by program, compiler, flags and codegen version.

If anything native fails (no compiler, unwritable cache directory, corrupt or unsupported cache entry), the program
falls back to the interpreter, rebuilds the entry, or switches to the dylib path.

## Requirements

* macOS on Apple Silicon (the loader and tuning target it; Linux works through the `.so` path, see below)
* Homebrew LLVM (clang 23, for `<print>`/C++23) to build the tool, and `clang` reachable at run time
  (override with `BF_CC`)

```sh
/opt/homebrew/opt/llvm/bin/clang++ main.cpp -O3 -std=c++23 -Wall -Wextra -Werror -Wshadow -Wconversion
```

## Usage and environment

```
./a.out [--emit-c] [--no-cache] [--pgo] [--tune] program.bf [< input]
```

| Option | Meaning |
|---|---|
| `--emit-c` | print the generated C |
| `--no-cache` | ignore cached builds |
| `--tune` | dev: auto-tune the code generator configuration (written to the cache) |
| `--pgo` | dev: profile-guided build (no measurable effect so far) |

| Variable | Meaning |
|---|---|
| `BF_CC`, `BF_CFLAGS` | compiler and flags (`BF_CFLAGS` also disables the -O1/-O3 two-tier mode) |
| `BF_CACHE_DIR` | cache directory (default under `$TMPDIR`) |
| `BF_NO_TIER` | always build before running (no interpreter race) |
| `BF_NO_OBJ` | use dylibs instead of the object loader |
| `BF_NO_FAST` | always compile with `-O3` |
| `BF_NO_UPGRADE` | no background `-O3` build; use this for cold-start benchmarks |
| `BF_LOADER_DEBUG` | print loader and fallback diagnostics |

Semantics: 8-bit wrapping cells; `,` at end of input sets the cell to 0; a few cells of zeroed slack exist to the
left of the tape start, while running far outside the tape is an error (exit code 1). Stray or unclosed brackets
are errors. If stdin is a regular file it is read into memory so both tiers see the same input.

## Testing

```sh
./reliability.sh ./a.out    # 32 checks: I/O edge cases, OOB, bad brackets, missing compiler, bad/corrupt cache,
                            # concurrent first runs, every build mode against a known md5
./bench_all.sh ./a.out      # downloads extra benchmarks, checks output across interpreter/-O1/-O3, prints timings
```

## Linux

The generator and the `.so` path build and run with `BF_CC=gcc` (tested on x86-64 with gcc), but the project is
tuned and benchmarked only on Apple Silicon.

## Project status and ideas

See the roadmap in the issues or the list below.

* Template JIT as tier 0 (emit arm64 straight from the IR in milliseconds, no clang): removes the interpreter and
  most of the cold-start cost; the loader infrastructure already exists.
* Record-layout (structure-of-arrays) pass for programs with fixed-stride records such as mandelbrot: expected 10–20%.
* Parallel compilation of code segments for very large programs.
* Standalone output (`-o program`) so a compiled program can be shipped without the tool.
* Optional cell width (16/32-bit), tape size and EOF behavior flags for compatibility with other implementations.
* CI on GitHub Actions (macOS arm64 runners) with a differential test: interpreter vs native vs reference
  implementation on a program corpus, plus fuzzing.
* Split `main.cpp` into modules (IR, optimizer, codegen, loader, runner) with unit tests.