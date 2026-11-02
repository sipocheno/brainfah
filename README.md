# brainfah
Brainfah interpreter in c++23. I am currently aiming at making fast

# Prerequisites
Basically, I only tested it on my Apple Silicon Mac, so it may not work on intel macOS or other platforms.
You need to download latest llvm:

`brew install llvm`

## Build
Use CMake if you want, but you can also use following command
```
clang++ -std=c++23 -O3 -mcpu=native -Wall -Wextra -Werror -Wpedantic -Wshadow -Wconversion main.cpp -o brainfah

# Run

./brainfah tests/mandelbrot.bf
./brainfah --emit-c tests/mandelbrot.bf
./brainfah --no-cache tests/mandelbrot.bf
./brainfah --tune tests/mandelbrot.bf  (benchmarks variants once, later runs use the best)
/brainfah --pgo tests/mandelbrot.bf   (1st run trains, later runs use the profile)
