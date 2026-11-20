//! The concurrency core of frees across threads.
//!
//! Every thread heap embeds a [`Remote`]: an *inbox* that other threads push
//! whole batches of freed blocks onto (one compare-and-swap per batch), and a
//! state, active or parked. Exited threads' heaps are *parked* in a [`Pool`]
//! and adopted by threads that start later.
//!
//! The one subtle property is the store-load handshake between a sender and
//! an owner that parks: the sender pushes and then reads the state; the owner
//! sets the state and then takes the inbox. With sequentially consistent
//! ordering on both sides, either the owner's take sees the batch or the
//! sender sees `PARKED` (and drains the parked heap itself, under the pool
//! lock). The Loom tests at the bottom check exactly this.
//!
//! Messages are opaque here: the heap decides what a message is and where its
//! link word lives ([`Link`]). Under `--cfg loom`, atomics and the mutex come
//! from Loom.

use core::ptr;

#[cfg(loom)]
use loom::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicPtr, AtomicU8, Ordering::*},
};
#[cfg(not(loom))]
use {
    core::sync::atomic::{AtomicPtr, AtomicU8, Ordering::*},
    std::sync::{Mutex, MutexGuard},
};

/// An opaque message: one freed block, linked through a word the heap chooses.
pub(crate) type Msg = *mut u8;

/// Where a message keeps its link to the next message of a list.
pub(crate) trait Link {
    /// # Safety
    /// `m` is a live message.
    unsafe fn next(m: Msg) -> Msg;
    /// # Safety
    /// `m` is a live message owned by the caller.
    unsafe fn set_next(m: Msg, next: Msg);
}

/// Owned by a thread.
pub(crate) const ACTIVE: u8 = 0;
/// In the pool: its thread has exited.
pub(crate) const PARKED: u8 = 1;

/// The shared part of a thread heap: its inbox and its state. On a cache line
/// of its own: senders write it, and the owner's fields around it are written
/// on every allocation.
#[repr(align(64))]
pub(crate) struct Remote {
    /// Head of the inbox: messages from other threads, newest batch first.
    inbox: AtomicPtr<u8>,
    state: AtomicU8,
    /// Next parked remote in the pool; read and written under the pool lock
    /// only (an atomic so the type stays `Sync` and Loom can see it).
    pool_next: AtomicPtr<Remote>,
}

impl Remote {
    #[cfg(not(loom))]
    pub(crate) const fn new() -> Remote {
        Remote {
            inbox: AtomicPtr::new(ptr::null_mut()),
            state: AtomicU8::new(ACTIVE),
            pool_next: AtomicPtr::new(ptr::null_mut()),
        }
    }

    #[cfg(loom)]
    pub(crate) fn new() -> Remote {
        Remote {
            inbox: AtomicPtr::new(ptr::null_mut()),
            state: AtomicU8::new(ACTIVE),
            pool_next: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Pushes the batch `head..=tail` onto the inbox and returns whether the
    /// remote was parked right after: then the caller locks the pool and, if
    /// the remote is still parked, drains it as its owner.
    ///
    /// # Safety
    /// `head` to `tail` is a list of live messages linked through `L`, owned by
    /// the caller and not in any other list.
    pub(crate) unsafe fn push<L: Link>(&self, head: Msg, tail: Msg) -> bool {
        let mut old = self.inbox.load(Relaxed);
        loop {
            // SAFETY: caller's contract; `tail` is not shared yet.
            unsafe { L::set_next(tail, old) };
            match self.inbox.compare_exchange_weak(old, head, SeqCst, Relaxed) {
                Ok(_) => break,
                Err(h) => old = h,
            }
        }
        self.state.load(SeqCst) == PARKED
    }

    /// Takes every message pushed so far, as one list (null if none).
    pub(crate) fn take_all(&self) -> Msg {
        self.inbox.swap(ptr::null_mut(), SeqCst)
    }

    /// Whether anything is waiting, without taking it.
    pub(crate) fn has_mail(&self) -> bool {
        !self.inbox.load(Relaxed).is_null()
    }

    #[cfg(test)]
    pub(crate) fn is_parked(&self) -> bool {
        self.state.load(SeqCst) == PARKED
    }
}

/// The parked remotes, linked through `pool_next`.
pub(crate) struct PoolList {
    head: *mut Remote,
}

// SAFETY: the list is only touched under the pool's mutex.
unsafe impl Send for PoolList {}

/// Exited threads' heaps, waiting for a thread to adopt them.
pub(crate) struct Pool {
    lock: Mutex<PoolList>,
}

impl Pool {
    #[cfg(not(loom))]
    pub(crate) const fn new() -> Pool {
        Pool {
            lock: Mutex::new(PoolList {
                head: ptr::null_mut(),
            }),
        }
    }

    #[cfg(loom)]
    pub(crate) fn new() -> Pool {
        Pool {
            lock: Mutex::new(PoolList {
                head: ptr::null_mut(),
            }),
        }
    }

    pub(crate) fn lock(&self) -> PoolGuard<'_> {
        PoolGuard(self.lock.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// The pool, locked.
pub(crate) struct PoolGuard<'a>(MutexGuard<'a, PoolList>);

impl PoolGuard<'_> {
    /// Marks `r` parked and adds it to the pool.
    ///
    /// # Safety
    /// `r` is a live remote, not in the pool, that outlives its time in it.
    pub(crate) unsafe fn park(&mut self, r: *mut Remote) {
        // SAFETY: caller's contract.
        unsafe {
            (*r).state.store(PARKED, SeqCst);
            (*r).pool_next.store(self.0.head, Relaxed);
        }
        self.0.head = r;
    }

    /// Takes the first parked remote for which `f` holds out of the pool and
    /// marks it active.
    pub(crate) fn pop_where(
        &mut self,
        mut f: impl FnMut(*mut Remote) -> bool,
    ) -> Option<*mut Remote> {
        let mut prev: *mut Remote = ptr::null_mut();
        let mut r = self.0.head;
        while !r.is_null() {
            // SAFETY: remotes in the pool are live (`park`'s contract).
            let next = unsafe { (*r).pool_next.load(Relaxed) };
            if f(r) {
                if prev.is_null() {
                    self.0.head = next;
                } else {
                    // SAFETY: as above.
                    unsafe { (*prev).pool_next.store(next, Relaxed) };
                }
                // SAFETY: as above.
                unsafe { (*r).state.store(ACTIVE, SeqCst) };
                return Some(r);
            }
            prev = r;
            r = next;
        }
        None
    }

    /// Some parked remote, left in the pool.
    pub(crate) fn first(&self) -> Option<*mut Remote> {
        (!self.0.head.is_null()).then_some(self.0.head)
    }

    /// Whether `r` is in the pool (still parked).
    pub(crate) fn contains(&self, r: *mut Remote) -> bool {
        let mut p = self.0.head;
        while !p.is_null() {
            if p == r {
                return true;
            }
            // SAFETY: remotes in the pool are live.
            p = unsafe { (*p).pool_next.load(Relaxed) };
        }
        false
    }
}

/// How batches travel between heaps: the default is the inbox above; a
/// runtime with its own channels between threads (glommio) can carry them
/// instead. Not public yet.
pub(crate) trait Transport {
    /// Sends a batch; returns whether `dest` was parked right after (see
    /// [`Remote::push`]).
    ///
    /// # Safety
    /// As [`Remote::push`].
    unsafe fn send<L: Link>(dest: &Remote, head: Msg, tail: Msg) -> bool;
    /// Takes everything sent to `dest` so far.
    fn receive(dest: &Remote) -> Msg;
}

/// The lock-free stack inbox of [`Remote`].
pub(crate) struct StackTransport;

impl Transport for StackTransport {
    unsafe fn send<L: Link>(dest: &Remote, head: Msg, tail: Msg) -> bool {
        // SAFETY: caller's contract.
        unsafe { dest.push::<L>(head, tail) }
    }
    fn receive(dest: &Remote) -> Msg {
        dest.take_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// A test message: a boxed node whose first field is the link word.
    #[repr(C)]
    struct Node {
        next: Msg,
        id: usize,
    }

    struct NodeLink;

    impl Link for NodeLink {
        unsafe fn next(m: Msg) -> Msg {
            // SAFETY: `m` is a live `Node`.
            unsafe { (*m.cast::<Node>()).next }
        }
        unsafe fn set_next(m: Msg, next: Msg) {
            // SAFETY: `m` is a live `Node`.
            unsafe { (*m.cast::<Node>()).next = next }
        }
    }

    fn node(id: usize) -> Msg {
        Box::into_raw(Box::new(Node {
            next: core::ptr::null_mut(),
            id,
        }))
        .cast()
    }

    /// Links `ids` into a batch; returns (head, tail).
    fn batch(ids: impl IntoIterator<Item = usize>) -> (Msg, Msg) {
        let nodes: Vec<Msg> = ids.into_iter().map(node).collect();
        for w in nodes.windows(2) {
            // SAFETY: fresh nodes.
            unsafe { NodeLink::set_next(w[0], w[1]) };
        }
        (nodes[0], *nodes.last().unwrap())
    }

    /// Frees every node of a list, returning their ids.
    fn collect(mut m: Msg, out: &mut Vec<usize>) {
        while !m.is_null() {
            // SAFETY: a node made by `node`, reached exactly once.
            let n = unsafe { Box::from_raw(m.cast::<Node>()) };
            out.push(n.id);
            m = n.next;
        }
    }

    struct Shared(Remote);
    // SAFETY: `Remote` is shared between threads by design; the raw pointers in
    // messages are owned by whoever holds them.
    unsafe impl Sync for Shared {}
    unsafe impl Send for Shared {}

    #[cfg(not(loom))]
    #[test]
    fn senders_batches_all_arrive_once() {
        let r = std::sync::Arc::new(Shared(Remote::new()));
        let n = if cfg!(miri) { 2 } else { 50 };
        let threads: Vec<_> = (0..3)
            .map(|t| {
                let r = r.clone();
                std::thread::spawn(move || {
                    for b in 0..n {
                        let base = (t * n + b) * 3;
                        let (h, tl) = batch(base..base + 3);
                        // SAFETY: a well-formed batch of fresh nodes.
                        assert!(!unsafe { r.0.push::<NodeLink>(h, tl) });
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let mut ids = Vec::new();
        collect(r.0.take_all(), &mut ids);
        assert_eq!(ids.len(), 3 * n * 3);
        assert_eq!(
            ids.iter().copied().collect::<BTreeSet<_>>(),
            (0..3 * n * 3).collect()
        );
        assert!(r.0.take_all().is_null());
    }

    #[cfg(not(loom))]
    #[test]
    fn push_reports_parked_and_adoption_activates() {
        let r = Remote::new();
        let rp = (&raw const r).cast_mut();
        let (h, t) = batch([1]);
        // SAFETY: a fresh batch.
        assert!(!unsafe { r.push::<NodeLink>(h, t) });
        let pool = Pool::new();
        // SAFETY: `r` outlives the pool's use of it.
        unsafe { pool.lock().park(rp) };
        assert!(r.is_parked());
        assert!(pool.lock().contains(rp));
        let (h, t) = batch([2]);
        // SAFETY: a fresh batch.
        assert!(
            unsafe { r.push::<NodeLink>(h, t) },
            "push into a parked remote"
        );
        assert_eq!(pool.lock().pop_where(|_| true), Some(rp));
        assert!(!r.is_parked());
        assert!(!pool.lock().contains(rp));
        let mut ids = Vec::new();
        collect(r.take_all(), &mut ids);
        assert_eq!(ids, [2, 1]);
    }

    #[cfg(not(loom))]
    #[test]
    fn pop_where_picks_the_matching_remote() {
        let (a, b) = (Remote::new(), Remote::new());
        let (ap, bp) = ((&raw const a).cast_mut(), (&raw const b).cast_mut());
        let pool = Pool::new();
        // SAFETY: both outlive the pool's use of them.
        unsafe {
            let mut g = pool.lock();
            g.park(ap);
            g.park(bp);
        }
        assert_eq!(pool.lock().pop_where(|r| r == ap), Some(ap));
        assert!(pool.lock().contains(bp) && !pool.lock().contains(ap));
        assert_eq!(pool.lock().pop_where(|r| r == ap), None);
        assert_eq!(pool.lock().pop_where(|_| true), Some(bp));
    }

    // ---- Loom: every interleaving of the handshakes -------------------------

    #[cfg(loom)]
    fn drained(m: Msg) -> Vec<usize> {
        let mut v = Vec::new();
        collect(m, &mut v);
        v
    }

    #[cfg(loom)]
    fn assert_each_once(mut all: Vec<usize>, pushed: &[usize]) {
        all.sort_unstable();
        let mut want = pushed.to_vec();
        want.sort_unstable();
        assert_eq!(all, want, "every message drained exactly once");
    }

    #[cfg(loom)]
    #[test]
    fn loom_two_senders_one_drain() {
        loom::model(|| {
            let r = loom::sync::Arc::new(Shared(Remote::new()));
            let senders: Vec<_> = [[1, 2], [3, 4]]
                .into_iter()
                .map(|ids| {
                    let r = r.clone();
                    loom::thread::spawn(move || {
                        let (h, t) = batch(ids);
                        // SAFETY: a fresh batch.
                        unsafe { r.0.push::<NodeLink>(h, t) };
                    })
                })
                .collect();
            let mut all = drained(r.0.take_all());
            for s in senders {
                s.join().unwrap();
            }
            all.extend(drained(r.0.take_all()));
            assert_each_once(all, &[1, 2, 3, 4]);
        });
    }

    #[cfg(loom)]
    #[test]
    fn loom_park_versus_push() {
        loom::model(|| {
            let r = loom::sync::Arc::new(Shared(Remote::new()));
            let pool = loom::sync::Arc::new(Pool::new());
            let rp = (&raw const r.0).cast_mut();
            let owner = {
                let (r, pool) = (r.clone(), pool.clone());
                let rp = rp as usize;
                loom::thread::spawn(move || {
                    let mut g = pool.lock();
                    // SAFETY: `r` outlives the model run.
                    unsafe { g.park(rp as *mut Remote) };
                    drained(r.0.take_all())
                })
            };
            let (h, t) = batch([7, 8]);
            // SAFETY: a fresh batch.
            let mut sender = Vec::new();
            if unsafe { r.0.push::<NodeLink>(h, t) } {
                let g = pool.lock();
                if g.contains(rp) {
                    sender = drained(r.0.take_all());
                }
            }
            let owner = owner.join().unwrap();
            assert!(
                r.0.take_all().is_null(),
                "a batch was lost: nobody drained it"
            );
            assert_each_once(owner.into_iter().chain(sender).collect(), &[7, 8]);
        });
    }

    #[cfg(loom)]
    #[test]
    fn loom_adopt_versus_push() {
        loom::model(|| {
            let r = loom::sync::Arc::new(Shared(Remote::new()));
            let pool = loom::sync::Arc::new(Pool::new());
            let rp = (&raw const r.0).cast_mut();
            // SAFETY: `r` outlives the model run.
            unsafe { pool.lock().park(rp) };
            let adopter = {
                let (r, pool) = (r.clone(), pool.clone());
                let rp = rp as usize;
                loom::thread::spawn(move || {
                    let got = pool.lock().pop_where(|x| x == rp as *mut Remote);
                    assert!(got.is_some());
                    drained(r.0.take_all())
                })
            };
            let (h, t) = batch([5]);
            let mut sender = Vec::new();
            // SAFETY: a fresh batch.
            if unsafe { r.0.push::<NodeLink>(h, t) } {
                let g = pool.lock();
                if g.contains(rp) {
                    sender = drained(r.0.take_all());
                }
            }
            let adopter = adopter.join().unwrap();
            // The adopter owns the heap now; its next drain takes what is left.
            let later = drained(r.0.take_all());
            assert!(!r.0.is_parked());
            assert_each_once(
                adopter.into_iter().chain(sender).chain(later).collect(),
                &[5],
            );
        });
    }
}
