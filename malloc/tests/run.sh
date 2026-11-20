#!/bin/bash
# Builds libshardheap_malloc.so (nightly, initial-exec TLS) and runs the C API
# test and a smoke run of everyday programs with it preloaded. Fails on the
# first error.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUSTFLAGS='-Z tls-model=initial-exec' CARGO_TARGET_DIR=target-malloc \
    cargo +nightly build --release -p shardheap-malloc
LIB=$PWD/target-malloc/release/libshardheap_malloc.so
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

# Thread-locals are read without __tls_get_addr (initial-exec model): no
# function of shardheap or of this library calls it (std's own do, off the
# allocation paths).
n=$(objdump -d --no-show-raw-insn "$LIB" | awk '
    /^[0-9a-f]+ </ { f = ($0 ~ /9shardheap|16shardheap_malloc|<(malloc|free|calloc|realloc)>/) }
    f && /call.*__tls_get_addr/ { n++ }
    END { print n + 0 }')
if [ "$n" != 0 ]; then echo "shardheap code calls __tls_get_addr ($n times)"; exit 1; fi
echo "tls: initial-exec"

cc -O2 -pthread -o "$OUT/api" malloc/tests/api.c
LD_PRELOAD=$LIB "$OUT/api"

cc -O2 -pthread -o "$OUT/tsd" malloc/tests/tsd.c
LD_PRELOAD=$LIB "$OUT/tsd"

c++ -O2 -pthread -o "$OUT/smoke" malloc/tests/smoke.cpp
LD_PRELOAD=$LIB "$OUT/smoke"

LD_PRELOAD=$LIB ls -la /usr/lib > /dev/null
seq 1000000 | LD_PRELOAD=$LIB shuf | LD_PRELOAD=$LIB sort -n | tail -1 | grep -qx 1000000
LD_PRELOAD=$LIB python3 -c "import json; print(len(json.dumps(list(range(10**6)))))" > /dev/null
LD_PRELOAD=$LIB git --version > /dev/null
echo "smoke: ok"
