#!/bin/bash
# usage: ./reliability.sh [./a.out]   (run from the project root; needs tests/mandelbrot.bf)
BIN=${1:-./a.out}
W=$(mktemp -d /tmp/bfrel.XXXXXX); pass=0; failn=0
MD5=5024283fa65866ddd347b877798e84d8
md5f() { if command -v md5 >/dev/null; then md5 -q; else md5sum | cut -d' ' -f1; fi; }
ok() { pass=$((pass+1)); printf 'PASS  %s\n' "$1"; }
bad() { failn=$((failn+1)); printf 'FAIL  %s   (%s)\n' "$1" "$2"; }
eq() { [ "$2" = "$3" ] && ok "$1" || bad "$1" "got '$2' expected '$3'"; }
M=tests/mandelbrot.bf
echo ',[.,]' > $W/cat.bf; echo '++++++++[>++++++++<-]>+.' > $W/a.bf; echo '+[>+]' > $W/oob.bf
echo '[' > $W/open.bf; echo ']' > $W/close.bf; : > $W/empty.bf; echo ',.' > $W/eof.bf

# --- basic behaviour ---------------------------------------------------------------------------
eq "hello"            "$(BF_CACHE_DIR=$W/c1 $BIN $W/a.bf)" "A"
eq "empty program rc" "$($BIN $W/empty.bf >/dev/null 2>&1; echo $?)" 0
eq "stdin pipe"       "$(printf hello | BF_CACHE_DIR=$W/c1 $BIN $W/cat.bf)" hello
printf hello > $W/in.txt
eq "stdin file"       "$(BF_CACHE_DIR=$W/c1 $BIN $W/cat.bf < $W/in.txt)" hello
eq "stdin empty"      "$(BF_CACHE_DIR=$W/c1 $BIN $W/cat.bf < /dev/null | wc -c | tr -d ' ')" 0
eq "stdin closed rc"  "$(BF_CACHE_DIR=$W/c1 $BIN $W/eof.bf <&- >/dev/null 2>&1; echo $?)" 0
eq "oob rc"           "$(BF_CACHE_DIR=$W/c1 $BIN $W/oob.bf >/dev/null 2>&1; echo $?)" 1
eq "oob message"      "$(BF_CACHE_DIR=$W/c1 $BIN $W/oob.bf 2>&1 >/dev/null | grep -c 'out of bounds')" 1
eq "oob native rc"    "$(BF_NO_TIER=1 BF_CACHE_DIR=$W/c1b $BIN $W/oob.bf >/dev/null 2>&1; echo $?)" 1
eq "unclosed ["       "$($BIN $W/open.bf >/dev/null 2>&1; echo $?)" 1
eq "stray ]"          "$($BIN $W/close.bf >/dev/null 2>&1; echo $?)" 1
eq "missing file rc"  "$($BIN $W/none.bf >/dev/null 2>&1; echo $?)" 1
eq "no args rc"       "$($BIN >/dev/null 2>&1; echo $?)" 2

# --- mandelbrot in every mode ------------------------------------------------------------------
eq "mandelbrot default cold" "$(BF_CACHE_DIR=$W/c2 BF_NO_UPGRADE=1 $BIN $M | md5f)" $MD5
eq "mandelbrot default warm" "$(BF_CACHE_DIR=$W/c2 BF_NO_UPGRADE=1 $BIN $M | md5f)" $MD5
eq "mandelbrot native only"  "$(BF_CACHE_DIR=$W/c3 BF_NO_TIER=1 BF_NO_UPGRADE=1 $BIN $M | md5f)" $MD5
eq "mandelbrot no-obj"       "$(BF_CACHE_DIR=$W/c4 BF_NO_OBJ=1 BF_NO_TIER=1 $BIN $M | md5f)" $MD5
eq "mandelbrot -O3 direct"   "$(BF_CACHE_DIR=$W/c5 BF_NO_FAST=1 BF_NO_TIER=1 $BIN $M | md5f)" $MD5
eq "mandelbrot --no-cache"   "$(BF_CACHE_DIR=$W/c6 $BIN --no-cache $M | md5f)" $MD5

# --- environment problems ----------------------------------------------------------------------
eq "compiler missing (tier)"  "$(BF_CC=/nonexistent BF_CACHE_DIR=$W/d1 $BIN $M 2>/dev/null | md5f)" $MD5
eq "compiler missing (pipe)"  "$(printf hello | BF_CC=/nonexistent BF_CACHE_DIR=$W/d2 $BIN $W/cat.bf 2>/dev/null)" hello
eq "compiler missing (nat.)"  "$(BF_CC=/nonexistent BF_NO_TIER=1 BF_CACHE_DIR=$W/d3 $BIN $M 2>/dev/null | md5f)" $MD5
eq "bad cache dir"            "$(BF_CACHE_DIR=/dev/null/x $BIN $M 2>/dev/null | md5f)" $MD5
eq "bad cache dir + pipe"     "$(printf hello | BF_CACHE_DIR=/dev/null/x $BIN $W/cat.bf 2>/dev/null)" hello

# --- cache corruption: must self-heal and must NOT disable object mode ---------------------------
CC=$W/e1; BF_CACHE_DIR=$CC BF_NO_TIER=1 BF_NO_UPGRADE=1 $BIN $M >/dev/null
for f in $CC/bf_*.o $CC/bf_*.dylib $CC/bf_*.so; do [ -f "$f" ] || continue
  sz=$(wc -c < "$f"); cp "$f" $W/orig
  : > "$f"
  eq "corrupt: empty file"     "$(BF_CACHE_DIR=$CC BF_NO_TIER=1 BF_NO_UPGRADE=1 $BIN $M 2>/dev/null | md5f)" $MD5
  case "$f" in *.o) # truncated dylibs can only come from outside tampering (cache writes are atomic renames)
    head -c $((sz/2)) $W/orig > "$f"
    eq "corrupt: truncated half" "$(BF_CACHE_DIR=$CC BF_NO_TIER=1 BF_NO_UPGRADE=1 $BIN $M 2>/dev/null | md5f)" $MD5;;
  esac
  head -c 4096 /dev/urandom > "$f"
  eq "corrupt: random bytes"   "$(BF_CACHE_DIR=$CC BF_NO_TIER=1 BF_NO_UPGRADE=1 $BIN $M 2>/dev/null | md5f)" $MD5
  eq "corrupt: file rebuilt"   "$([ -s "$f" ] && echo yes || echo no)" yes
done
eq "no obj-unsupported marker after corruption" "$([ -e $CC/bf_obj_unsupported ] && echo present || echo absent)" absent
printf hey > $W/in2.txt; BF_CACHE_DIR=$W/e2 BF_NO_TIER=1 BF_NO_UPGRADE=1 $BIN $W/cat.bf < $W/in2.txt >/dev/null
for f in $W/e2/bf_*.o $W/e2/bf_*.dylib $W/e2/bf_*.so; do [ -f "$f" ] && : > "$f"; done
eq "corrupt + stdin file restarts with input" "$(BF_CACHE_DIR=$W/e2 BF_NO_TIER=1 BF_NO_UPGRADE=1 $BIN $W/cat.bf < $W/in2.txt 2>/dev/null)" hey

# --- concurrency ---------------------------------------------------------------------------------
rm -rf $W/g1; for i in 1 2 3 4 5 6; do (BF_CACHE_DIR=$W/g1 $BIN $M | md5f > $W/g.$i) & done; wait
eq "6 parallel cold runs" "$(cat $W/g.? | sort -u | tr '\n' ' ')" "$MD5 "
sleep 3
eq "after upgrade, still correct" "$(BF_CACHE_DIR=$W/g1 $BIN $M | md5f)" $MD5

echo; echo "passed: $pass  failed: $failn"; rm -rf "$W"
[ $failn -eq 0 ]