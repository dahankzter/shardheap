//! Transparent huge pages: does backing segments with 2 MiB pages pay off?
//!
//! Builds `nodes` 64-byte boxed nodes in a `LocalHeap`, links them into one
//! random cycle, and chases it: a working set far beyond the 4 KiB-page TLB
//! reach (Zen 5: about 3k L2 TLB entries, 12 MiB) but well inside 2 MiB reach.
//!
//! ```text
//! cargo run --release --example thp -- [auto|always|never|kernel] [nodes] [hops]
//! ```

use std::cell::Cell;
use std::hint::black_box;
use std::time::Instant;

use shardheap::{HugePages, LocalHeap};

struct Node<'a> {
    next: Cell<Option<&'a Node<'a>>>,
    _payload: [u64; 7],
}

fn rng(mut s: u64) -> impl FnMut() -> u64 {
    move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn usage() -> (i64, usize) {
    // SAFETY: plain syscall filling a zeroed struct.
    let minflt = unsafe {
        let mut u: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut u);
        u.ru_minflt
    };
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let rss: usize = statm
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (minflt, rss * 4 / 1024)
}

fn anon_huge_mib() -> usize {
    let s = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with("AnonHugePages:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
        / 1024
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("auto");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4 << 20);
    let hops: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000_000);
    let policy = match mode {
        "auto" => HugePages::Auto,
        "always" => HugePages::Always,
        "never" => HugePages::Never,
        "kernel" => HugePages::Kernel,
        other => panic!("unknown policy {other}"),
    };

    let heap = LocalHeap::with_huge_pages(policy);
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut r = rng(1);
    for i in (1..n).rev() {
        order.swap(i, (r() % (i as u64 + 1)) as usize);
    }

    let (f0, _) = usage();
    let t = Instant::now();
    let nodes: Vec<Box<Node<'_>, &LocalHeap>> = (0..n)
        .map(|_| {
            Box::new_in(
                Node {
                    next: Cell::new(None),
                    _payload: [1; 7],
                },
                &heap,
            )
        })
        .collect();
    let build = t.elapsed();
    let (f1, rss) = usage();
    let huge = anon_huge_mib();

    // Link in random order: one cycle through every node.
    let refs: Vec<&Node<'_>> = nodes.iter().map(|b| &**b).collect();
    for w in order.windows(2) {
        refs[w[0] as usize].next.set(Some(refs[w[1] as usize]));
    }
    refs[order[n - 1] as usize]
        .next
        .set(Some(refs[order[0] as usize]));

    let t = Instant::now();
    let mut p = refs[order[0] as usize];
    for _ in 0..hops {
        p = p.next.get().unwrap();
    }
    black_box(p);
    let chase = t.elapsed();

    println!(
        "{mode:>8}  {} MiB of nodes  build {:>6.1} ms ({:>6} faults)  chase {:>5.1} ns/hop  \
         rss {:>5} MiB  AnonHugePages {:>5} MiB",
        (n * 64) >> 20,
        build.as_secs_f64() * 1e3,
        f1 - f0,
        chase.as_nanos() as f64 / hops as f64,
        rss,
        huge,
    );
}
