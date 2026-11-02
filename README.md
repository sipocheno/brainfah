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
