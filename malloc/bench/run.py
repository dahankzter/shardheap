#!/usr/bin/env python3
"""Runs mimalloc-bench's benchmarks with several allocators and prints a table.

    malloc/bench/run.py <mimalloc-bench dir> <name=lib.so|glibc>... [--runs N] [--only a,b]

Each allocator but glibc is loaded with LD_PRELOAD. Every benchmark runs
`--runs` times per allocator (rounds interleaved); the table shows the median
time and peak resident memory, and the ratio of each to mimalloc's ("mi" if
present, else the first allocator). Throughput benchmarks (larson, rptest,
xmalloc-test, glibc-thread) report a relative time computed from their own
output, as mimalloc-bench's bench.sh does. A run that fails is reported as
failed, never dropped. Command lines follow bench.sh, with procs = nproc.
"""

import os
import re
import statistics
import subprocess
import sys
import time

def tests(bench, procs):
    b = os.path.join(bench, "out", "bench")
    pdf = os.path.join(bench, "extern", "large.pdf")
    return [
        ("cfrac", "./cfrac 17545186520507317056371138836327483792789528", None, None),
        ("espresso", "./espresso ../../bench/espresso/largest.espresso", None, None),
        ("barnes", "./barnes", os.path.join(bench, "bench", "barnes", "input"), None),
        ("gs", f"gs -dBATCH -dNODISPLAY {pdf}", None, None),
        ("alloc-test1", "./alloc-test 1", None, None),
        ("alloc-testN", f"./alloc-test {min(procs, 16)}", None, None),
        ("larsonN", f"./larson 5 8 1000 5000 100 4141 {procs}", None,
         lambda out: float(re.search(r" time: ([0-9.]+)", out).group(1))),
        ("larsonN-sized", f"./larson-sized 5 8 1000 5000 100 4141 {procs}", None,
         lambda out: float(re.search(r" time: ([0-9.]+)", out).group(1))),
        ("sh6benchN", f"./sh6bench {2 * procs}", None, None),
        ("sh8benchN", f"./sh8bench {2 * procs}", None, None),
        ("xmalloc-testN", f"./xmalloc-test -w {procs} -t 5 -s 64", None,
         lambda out: float(re.search(r"rtime: ([0-9.]+)", out).group(1))),
        ("cache-thrash1", f"./cache-thrash 1 1000 1 2000000 {procs}", None, None),
        ("cache-thrashN", f"./cache-thrash {procs} 1000 1 2000000 {procs}", None, None),
        ("cache-scratch1", f"./cache-scratch 1 1000 1 2000000 {procs}", None, None),
        ("cache-scratchN", f"./cache-scratch {procs} 1000 1 2000000 {procs}", None, None),
        ("malloc-large", "./malloc-large", None, None),
        ("mstressN", f"./mstress {procs} 50 25", None, None),
        ("mleak10", "./mleak 5", None, None),
        ("mleak100", "./mleak 50", None, None),
        ("rptestN", f"./rptest {procs} 0 1 2 500 1000 100 8 16000", None,
         lambda out: 2000000 / int(re.search(r"\.\.\.([0-9]+) memory ops", out).group(1))),
        ("glibc-simple", "./glibc-simple", None, None),
        ("glibc-thread", f"./glibc-thread {procs}", None,
         lambda out: 1e9 / float(re.match(r"\s*([0-9.]+)", out).group(1))),
    ], b


def run_once(cmd, cwd, stdin, lib, rel):
    """One run in this process's only child: (time, peak KiB, None) or
    (None, None, output tail). ru_maxrss of RUSAGE_CHILDREN is the peak of the
    benchmark (`sh -c` execs a single command)."""
    import resource
    env = dict(os.environ)
    if lib:
        env["LD_PRELOAD"] = lib
    t0 = time.monotonic()
    with open(stdin or os.devnull) as inp:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdin=inp,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.monotonic() - t0
    if p.returncode != 0:
        return None, None, f"exit {p.returncode}: " + p.stdout[-400:]
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (rel(p.stdout) if rel else wall), peak, None


def child(argv):
    # child <cmd> <cwd> <stdin|-> <lib|-> <test>: one run, prints "time peak_kib"
    cmd, cwd, stdin, lib, name = argv
    procs = os.cpu_count()
    rel = {t[0]: t[3] for t in tests(os.path.dirname(os.path.dirname(cwd)), procs)[0]}[name]
    t, peak, err = run_once(cmd, cwd, None if stdin == "-" else stdin,
                            None if lib == "-" else lib, rel)
    if t is None:
        print("FAIL " + err.replace("\n", " | "))
    else:
        print(f"{t} {peak}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--child":
        return child(args[1:])
    runs, only = 3, None
    if "--runs" in args:
        i = args.index("--runs"); runs = int(args[i + 1]); del args[i:i + 2]
    if "--only" in args:
        i = args.index("--only"); only = set(args[i + 1].split(",")); del args[i:i + 2]
    bench, allocs = os.path.abspath(args[0]), [a.split("=", 1) for a in args[1:]]
    procs = os.cpu_count()
    ts, cwd = tests(bench, procs)
    ts = [t for t in ts if only is None or t[0] in only]
    res = {}  # (test, alloc) -> list of (time, peak) or "FAIL ..."
    for r in range(runs):
        for name, cmd, stdin, _ in ts:
            for an, lib in allocs:
                lib = None if lib == "glibc" else os.path.abspath(lib)
                out = subprocess.run(
                    [sys.executable, __file__, "--child", cmd, cwd, stdin or "-", lib or "-", name],
                    capture_output=True, text=True).stdout.strip()
                print(f"round {r + 1} {name} {an}: {out[:120]}", file=sys.stderr, flush=True)
                res.setdefault((name, an), []).append(out)
    base = "mi" if any(a == "mi" for a, _ in allocs) else allocs[0][0]
    names = [a for a, _ in allocs]
    print("| benchmark | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for name, *_ in ts:
        meds = {}
        for an in names:
            rows = res[(name, an)]
            ok = [tuple(map(float, x.split())) for x in rows if not x.startswith("FAIL")]
            meds[an] = None if len(ok) < len(rows) else (
                statistics.median(t for t, _ in ok), statistics.median(p for _, p in ok))
        b = meds.get(base)
        cells = []
        for an in names:
            m = meds[an]
            if m is None:
                cells.append("**failed**")
            elif b is None or an == base:
                cells.append(f"{m[0]:.3f} s, {m[1] / 1024:.0f} MiB")
            else:
                cells.append(f"{m[0]:.3f} s ({m[0] / b[0]:.2f}), {m[1] / 1024:.0f} MiB ({m[1] / b[1]:.2f})")
        print(f"| {name} | " + " | ".join(cells) + " |")
    fails = [(k, x) for k, v in res.items() for x in v if x.startswith("FAIL")]
    for (name, an), x in fails[:20]:
        print(f"\n{name} with {an}: {x}")


if __name__ == "__main__":
    main()
