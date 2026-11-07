#!/bin/bash
# usage: ./bench_all.sh [./a.out]    (run from the project root, tests/ must hold mandelbrot.bf hanoi.bf awib-0.4.bf)
BIN=${1:-./a.out}
D=tests/extra; mkdir -p "$D"
M=https://raw.githubusercontent.com/matslina/bfoptimization/master/progs
get() { [ -s "$D/$2" ] || curl -sfL "$1" -o "$D/$2" || echo "download failed: $2"; }
get $M/long.b long.b
get $M/factor.b factor.b
get $M/hanoi.b hanoi_big.b
get $M/dbfi.b dbfi.b
get https://raw.githubusercontent.com/kostya/benchmarks/master/brainfuck/bench.b kbench.b
: > "$D/empty.in"
printf '123456789012\n' > "$D/factor.in"
printf ',[.,]!hi\n' > "$D/dbfi.in"
md5f() { if command -v md5 >/dev/null; then md5 -q; else md5sum | cut -d' ' -f1; fi; }
ms() { LC_ALL=C awk -F, 'NR==1{for(i=1;i<=NF;i++){if($i=="time_wall_clock_mean"||$i=="mean")m=i; if($i=="time_wall_clock_unit")u=i}} NR==2{f=1000; if(u&&$u=="ms")f=1; else if(u&&$u=="us")f=0.001; printf "%.0f", $m*f}' "$1"; }
hf() { # csv, prepare-or-empty, input, then command words
  local csv=$1 prep=$2 inp=$3; shift 3
  if [ -n "$prep" ]; then hyperfine -N --runs 5 --prepare "$prep" --input "$inp" --export-csv "$csv" "$@" >/dev/null 2>&1
  else hyperfine -N --warmup 2 --runs 8 --input "$inp" --export-csv "$csv" "$@" >/dev/null 2>&1; fi
}
printf '%-16s %-6s %9s %9s %9s\n' program md5 cold_ms O1warm O3warm
for spec in tests/mandelbrot.bf:$D/empty.in tests/hanoi.bf:$D/empty.in tests/awib-0.4.bf:tests/awib-0.4.bf \
            $D/long.b:$D/empty.in $D/hanoi_big.b:$D/empty.in $D/kbench.b:$D/empty.in \
            $D/factor.b:$D/factor.in $D/dbfi.b:$D/dbfi.in; do
  prog=${spec%%:*}; inp=${spec##*:}; name=$(basename "$prog")
  [ -f "$prog" ] || { echo "missing $prog"; continue; }
  rm -rf /tmp/bfx_ref /tmp/bfx_c /tmp/bfx_t /tmp/bfx_o1 /tmp/bfx_o3
  ref=$(BF_NO_OBJ=1 BF_NO_FAST=1 BF_NO_TIER=1 BF_CACHE_DIR=/tmp/bfx_ref "$BIN" "$prog" < "$inp" | md5f)
  c1=$(BF_CACHE_DIR=/tmp/bfx_c "$BIN" "$prog" < "$inp" | md5f)       # cold: interpreter/-O1 + background upgrade
  sleep 3
  c2=$(BF_CACHE_DIR=/tmp/bfx_c "$BIN" "$prog" < "$inp" | md5f)       # warm: whatever the cache holds now
  ok=OK; [ "$ref" = "$c1" ] && [ "$ref" = "$c2" ] || ok=DIFF
  hf /tmp/bfx_cold.csv 'rm -rf /tmp/bfx_t' "$inp" "env BF_NO_UPGRADE=1 BF_CACHE_DIR=/tmp/bfx_t $BIN $prog"
  BF_NO_UPGRADE=1 BF_NO_TIER=1 BF_CACHE_DIR=/tmp/bfx_o1 "$BIN" "$prog" < "$inp" >/dev/null
  hf /tmp/bfx_o1.csv '' "$inp" "env BF_NO_UPGRADE=1 BF_CACHE_DIR=/tmp/bfx_o1 $BIN $prog"
  BF_NO_FAST=1 BF_NO_TIER=1 BF_CACHE_DIR=/tmp/bfx_o3 "$BIN" "$prog" < "$inp" >/dev/null
  hf /tmp/bfx_o3.csv '' "$inp" "env BF_NO_FAST=1 BF_CACHE_DIR=/tmp/bfx_o3 $BIN $prog"
  printf '%-16s %-6s %9s %9s %9s\n' "$name" "$ok" "$(ms /tmp/bfx_cold.csv)" "$(ms /tmp/bfx_o1.csv)" "$(ms /tmp/bfx_o3.csv)"
done