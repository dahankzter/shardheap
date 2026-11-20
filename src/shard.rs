//! [`ShardAlloc`]: an allocator handle for the current thread's own heap.
//!
//! Every thread that uses `ShardAlloc` gets a *record*: heap state at a stable
//! address, mapped from the OS (not from the global allocator) and never freed,
//! so a block's owner stamp always names a valid record. Records are linked into
//! a static registry, and a const thread-local points at the current thread's.
//!
//! A block freed by another thread is a message to its owner: the freeing
//! thread batches messages per destination (`BatchCache`) and pushes each batch
//! onto the owner's inbox with one compare-and-swap; the owner frees them as
//! its own on its slow path. When a thread exits, its record is parked in a
//! pool, still receiving messages, and a thread that starts later adopts it.

use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::cell::{Cell, UnsafeCell};
use core::ptr::{self, NonNull};
use core::sync::atomic::Ordering::Relaxed;
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use crate::heap::{self, HugePages, Inner, MsgLink, Stats};
use crate::os::{self, PAGE};
use crate::remote::{Msg, Pool, Remote, StackTransport, Transport};

/// A thread's heap, at an address that stays valid for the life of the process.
/// `repr(C)` with the heap first, so a record's address is its heap's address
/// (`UnsafeCell` is `repr(transparent)`): the thread-local value is the owner
/// stamp itself.
#[repr(C)]
pub(crate) struct Record {
    inner: UnsafeCell<Inner>,
    /// Inbox and state, shared with other threads.
    remote: Remote,
    /// Frees this thread sends to other heaps, batched by destination.
    /// Owner-thread only.
    batch: UnsafeCell<BatchCache>,
    /// Messages taken from the inbox but not freed yet (bounded drains).
    /// Owner-thread only.
    remainder: Cell<Msg>,
    /// Next record in `RECORDS`.
    next: *mut Record,
    /// Parked, empty, and counted in `WARM`: it kept memory for the thread
    /// that adopts it. Under the pool lock only.
    warm: Cell<bool>,
}

impl Record {
    const fn new(inner: Inner) -> Record {
        Record {
            inner: UnsafeCell::new(inner),
            remote: Remote::new(),
            batch: UnsafeCell::new(BatchCache::new()),
            remainder: Cell::new(ptr::null_mut()),
            next: ptr::null_mut(),
            warm: Cell::new(false),
        }
    }
}

/// Destinations a thread batches remote frees for at once.
const BATCH_SLOTS: usize = 8;
/// Bytes a thread holds in remote-free batches before it sends them all.
const BATCH_BUDGET: usize = 16 << 10;
/// Blocks of at least this many bytes travel listed in a carrier block
/// (`heap::carrier_msg`); smaller ones in group messages.
const CARRIER_MIN: usize = 256;
/// Messages a slow-path drain frees at most; the rest waits for the next one.
const DRAIN_LIMIT: usize = 256;

#[derive(Clone, Copy)]
struct Slot {
    /// Owner of the batched blocks; null for a free slot.
    dest: *mut Record,
    /// Messages ready to send, newest first.
    head: Msg,
    tail: Msg,
    /// Last use, for least-recently-used eviction.
    used: u64,
    /// The open group: small blocks of one page (`group_key`), chained from
    /// `first` to `last`, `count` of them (0: none open).
    first: *mut u8,
    last: *mut u8,
    key: usize,
    count: usize,
    /// The open carrier (null: none) and how many blocks it lists, of `room`.
    carrier: *mut u8,
    listed: usize,
    room: usize,
}

const NO_SLOT: Slot = Slot {
    dest: ptr::null_mut(),
    head: ptr::null_mut(),
    tail: ptr::null_mut(),
    used: 0,
    first: ptr::null_mut(),
    last: ptr::null_mut(),
    key: 0,
    count: 0,
    carrier: ptr::null_mut(),
    listed: 0,
    room: 0,
};

impl Slot {
    /// Queues message `m` to send.
    ///
    /// # Safety
    /// `m` is a live message owned by the caller, in no list.
    unsafe fn push(&mut self, m: Msg) {
        // SAFETY: caller's contract.
        unsafe { <MsgLink as crate::remote::Link>::set_next(m, self.head) };
        if self.head.is_null() {
            self.tail = m;
        }
        self.head = m;
    }

    /// Turns the open group, if any, into one message and queues it.
    ///
    /// # Safety
    /// The group's blocks are owned by the caller.
    unsafe fn close(&mut self) {
        if self.count != 0 {
            // SAFETY: caller's contract; the chain is well formed by `add`.
            unsafe { self.push(heap::group_msg(self.first, self.last, self.count)) };
            self.count = 0;
        }
    }

    /// Turns the open carrier, if any, into a message and queues it.
    ///
    /// # Safety
    /// The carrier and the blocks it lists are owned by the caller.
    unsafe fn close_carrier(&mut self) {
        if !self.carrier.is_null() {
            // SAFETY: caller's contract; `listed` entries are written.
            unsafe { self.push(heap::carrier_msg(self.carrier, self.listed)) };
            self.carrier = ptr::null_mut();
        }
    }

    /// Lists small block `m` of `bytes` bytes in the open carrier, or makes it
    /// the carrier.
    ///
    /// # Safety
    /// `m` is a live small block of this slot's destination, owned by the
    /// caller, of at least `bytes` bytes.
    unsafe fn carry(&mut self, m: *mut u8, bytes: usize) {
        // SAFETY: caller's contract; the open carrier has room for entry
        // `listed`.
        unsafe {
            if self.carrier.is_null() {
                self.carrier = m;
                self.listed = 0;
                self.room = heap::carrier_capacity(bytes);
                // Written now, not when the carrier is sent (see `add`).
                *m.cast::<*mut u8>() = ptr::null_mut();
            } else {
                *heap::carrier_slot(self.carrier, self.listed) = m;
                self.listed += 1;
                if self.listed == self.room {
                    self.close_carrier();
                }
            }
        }
    }
}

/// A thread's outgoing remote frees: one batch per destination heap, sent as a
/// whole with one compare-and-swap. Consecutive frees of one page's small
/// blocks form one group message, which the owner frees in one splice; bigger
/// blocks travel listed in a carrier, so the sender writes one block per
/// carrier instead of a link in each.
struct BatchCache {
    slots: [Slot; BATCH_SLOTS],
    /// Bytes held across all slots.
    bytes: usize,
    clock: u64,
}

impl BatchCache {
    const fn new() -> BatchCache {
        BatchCache {
            slots: [NO_SLOT; BATCH_SLOTS],
            bytes: 0,
            clock: 0,
        }
    }

    /// Adds message `m` (of a `bytes`-byte block) for `dest`; returns whether
    /// the cache is over its budget now.
    ///
    /// # Safety
    /// `m` is a live message for a block of `dest`, owned by the caller.
    unsafe fn add(&mut self, dest: *mut Record, m: Msg, bytes: usize) -> bool {
        self.clock += 1;
        let i = match self.slots.iter().position(|s| s.dest == dest) {
            Some(i) => i,
            None => {
                let i = (0..BATCH_SLOTS)
                    .min_by_key(|&i| (!self.slots[i].dest.is_null(), self.slots[i].used))
                    .expect("slots");
                // SAFETY: the evicted slot's messages are this thread's to send.
                unsafe { self.flush_slot(i) };
                self.slots[i].dest = dest;
                i
            }
        };
        let s = &mut self.slots[i];
        // SAFETY: caller's contract; a small message is its block, whose first
        // word is free.
        unsafe {
            match heap::group_key(m) {
                Some(_) if bytes >= CARRIER_MIN => s.carry(m, bytes),
                Some(k) if s.count != 0 && s.key == k => {
                    // Appended: the owner reuses the blocks in the order
                    // they were freed, as when it freed them one by one.
                    *s.last.cast::<*mut u8>() = m;
                    s.last = m;
                    s.count += 1;
                }
                Some(k) => {
                    s.close();
                    // Written now, not when the group is sent: the block's
                    // line is still the owner's, and a store missing on it
                    // would stall the send's locked compare-and-swap.
                    *m.cast::<*mut u8>() = ptr::null_mut();
                    (s.first, s.last, s.key, s.count) = (m, m, k, 1);
                }
                None => s.push(m),
            }
        }
        s.used = self.clock;
        self.bytes += bytes;
        self.bytes > BATCH_BUDGET
    }

    /// Sends slot `i`'s batch to its destination and frees the slot.
    ///
    /// # Safety
    /// The slot's messages are owned by this thread.
    unsafe fn flush_slot(&mut self, i: usize) {
        // SAFETY: caller's contract.
        unsafe {
            self.slots[i].close();
            self.slots[i].close_carrier();
        }
        let s = self.slots[i];
        if !s.head.is_null() {
            // SAFETY: records are never freed; the batch is a well-formed list.
            let parked =
                unsafe { StackTransport::send::<MsgLink>(&(*s.dest).remote, s.head, s.tail) };
            if parked {
                drain_parked(s.dest);
            }
        }
        self.slots[i] = NO_SLOT;
    }

    /// Sends every batch.
    ///
    /// # Safety
    /// As `flush_slot`.
    unsafe fn flush_all(&mut self) {
        for i in 0..BATCH_SLOTS {
            // SAFETY: caller's contract.
            unsafe { self.flush_slot(i) };
        }
        self.bytes = 0;
    }
}

/// Parked heaps that keep memory once empty: a thread that adopts one reuses
/// its segment instead of mapping and faulting in a fresh one. Each holds at
/// most one empty segment; beyond this many, an emptied parked heap gives all
/// its memory back.
const WARM_HEAPS: usize = 8;

/// Parked records with `warm` set. Changed under the pool lock only.
static WARM: AtomicUsize = AtomicUsize::new(0);

/// Whether the parked record `r` is warm. Under the pool lock.
fn warm(r: *mut Record) -> bool {
    // SAFETY: records are never freed; caller holds the pool lock.
    unsafe { (*r).warm.get() }
}

/// Uncounts `r` as warm: it was adopted, or holds blocks again. Under the pool
/// lock.
fn unwarm(r: *mut Record) {
    if warm(r) {
        // SAFETY: as `warm`.
        unsafe { (*r).warm.set(false) };
        WARM.fetch_sub(1, Relaxed);
    }
}

/// Settles what the parked record `r` keeps, under the pool lock: emptied, it
/// stays warm while fewer than `WARM_HEAPS` are, else gives all its memory
/// back. With live blocks it keeps its memory as it is: trimming it at exit
/// purged pages the blocks still freed into would fault in again (the next
/// thread to adopt it reuses them).
///
/// # Safety
/// `r` is parked and the caller holds the pool lock.
unsafe fn settle(r: *mut Record) {
    // SAFETY: caller's contract: a parked record is touched only under the lock.
    unsafe {
        let h = &mut *(*r).inner.get();
        if h.stats.allocated_bytes != 0 {
            unwarm(r);
        } else if !warm(r) {
            if WARM.load(Relaxed) < WARM_HEAPS {
                h.keep_warm();
                (*r).warm.set(true);
                WARM.fetch_add(1, Relaxed);
            } else {
                h.trim();
            }
        }
    }
}

/// A batch reached a parked heap: free it as the heap's owner, under the pool
/// lock, if the heap is still parked (else its adopter drains it). A parked heap
/// left with nothing allocated is settled (`settle`).
fn drain_parked(dest: *mut Record) {
    let g = POOL.lock();
    if g.contains(remote_of(dest)) {
        // SAFETY: a parked record is touched only under the pool lock.
        unsafe {
            drain_all(dest);
            (*(*dest).inner.get()).maintain();
            settle(dest);
        }
    }
    drop(g);
}

/// Frees up to `limit` messages sent to `r`: the remainder of the last drain
/// first, then the inbox. What is left waits in the remainder.
///
/// # Safety
/// `r` is the current thread's record, with no reference to its heap in use.
unsafe fn drain(r: *mut Record, limit: usize) {
    // SAFETY: caller's contract.
    unsafe { drain_into(r, &mut *(*r).inner.get(), limit) };
}

/// `drain` with the heap already borrowed: `h` is `r`'s heap.
///
/// # Safety
/// `r` is the current thread's record and `h` its heap.
unsafe fn drain_into(r: *mut Record, h: &mut Inner, limit: usize) -> bool {
    // SAFETY: caller's contract: owner-only fields; `h` frees its own blocks.
    unsafe {
        let mut m = (*r).remainder.get();
        if m.is_null() {
            m = StackTransport::receive(&(*r).remote);
        }
        if m.is_null() {
            return false;
        }
        (*r).remainder.set(h.free_msgs(m, limit));
        true
    }
}

/// Allocation after this thread's record was parked: from a parked record,
/// under the pool lock (a new one, parked at once, if the pool is empty).
#[cold]
#[inline(never)]
fn torn_allocate(layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let mut g = POOL.lock();
    let r = match g.first() {
        Some(rm) => record_of(rm),
        None => {
            let r = new_record();
            // SAFETY: a fresh record, never freed.
            unsafe { g.park(remote_of(r)) };
            r
        }
    };
    // SAFETY: a parked record is touched only under the pool lock, held here.
    let b = unsafe { (*(*r).inner.get()).alloc(layout) };
    // SAFETY: as above.
    unsafe { settle(r) };
    drop(g);
    b
}

/// A thread heap's mail hook (`Inner::mail`): sends this thread's batches and
/// drains its inbox, bounded. Returns whether a block was freed.
///
/// # Safety
/// `h` is a thread heap, on its own thread.
unsafe fn record_mail(h: &mut Inner) -> bool {
    // The owner stamp is the record, with the whole record's provenance.
    let r = h.owner.cast::<Record>();
    // SAFETY: caller's contract; owner-only fields of the record.
    unsafe {
        if (*(*r).batch.get()).bytes != 0 {
            (*(*r).batch.get()).flush_all();
        }
        has_mail(r) && drain_into(r, h, DRAIN_LIMIT)
    }
}

/// Frees everything sent to `r` so far.
///
/// # Safety
/// As `drain`.
unsafe fn drain_all(r: *mut Record) {
    // SAFETY: caller's contract.
    unsafe {
        while !(*r).remainder.get().is_null() || (*r).remote.has_mail() {
            drain(r, usize::MAX);
        }
    }
}

/// Whether `r` has messages waiting (inbox or remainder).
///
/// # Safety
/// `r` is the current thread's record.
#[inline]
unsafe fn has_mail(r: *mut Record) -> bool {
    // SAFETY: caller's contract.
    unsafe { !(*r).remainder.get().is_null() || (*r).remote.has_mail() }
}

/// A block of another heap was freed on this thread: batch it for its owner.
#[cold]
#[inline(never)]
fn remote_free(owner: *mut (), m: Msg, bytes: usize) {
    if owner.is_null() {
        heap::foreign_free(); // a `LocalHeap` block: misuse
    }
    let dest = owner.cast::<Record>();
    let cur = CURRENT.with(Cell::get);
    if cur == torn() {
        // After this thread's record was parked: no batch cache, send it alone.
        // SAFETY: `m` is owned here; records are never freed.
        unsafe {
            <MsgLink as crate::remote::Link>::set_next(m, ptr::null_mut());
            if StackTransport::send::<MsgLink>(&(*dest).remote, m, m) {
                drain_parked(dest);
            }
        }
        return;
    }
    let r = if cur == empty() { init_current() } else { cur };
    // SAFETY: the current thread's record; `owner` is a record (stamps are
    // record addresses), and records are never freed.
    unsafe {
        let batch = &mut *(*r).batch.get();
        if batch.add(owner.cast(), m, bytes) {
            batch.flush_all();
            if has_mail(r) {
                drain(r, DRAIN_LIMIT);
            }
        }
    }
}

/// Every record ever made, newest first. Push-only; keeps records reachable from
/// a static, and is the list stage B walks.
static RECORDS: AtomicPtr<Record> = AtomicPtr::new(ptr::null_mut());

/// The empty heap: every thread's `CURRENT` until its first allocation. Its
/// queues are all null, so allocation through it falls into the slow path,
/// which makes the thread's record; no block is stamped with its address, so a
/// free through it fails the owner check. Nothing ever writes it, nor takes a
/// `&mut` to it: it is only read, through raw pointers, from any thread. That
/// saves the null check on the allocation fast path (mimalloc's empty heap).
#[repr(transparent)]
struct Empty(Record);

// SAFETY: never written after const initialization; only read.
unsafe impl Sync for Empty {}

static EMPTY: Empty = Empty(Record::new(Inner::new(HugePages::Auto)));

/// A second empty heap: `CURRENT` after the thread's record was parked at exit.
/// Allocation and free through it take the pool lock (`torn_allocate`,
/// `remote_free`): other thread-locals' destructors may still use the handle.
static TORN: Empty = Empty(Record::new(Inner::new(HugePages::Auto)));

#[inline(always)]
fn empty() -> *mut Record {
    (&raw const EMPTY.0).cast_mut()
}

#[inline(always)]
fn torn() -> *mut Record {
    (&raw const TORN.0).cast_mut()
}

/// Exited threads' records, waiting to be adopted.
static POOL: Pool = Pool::new();

/// The record a pooled `Remote` belongs to.
fn record_of(rm: *mut Remote) -> *mut Record {
    // SAFETY: pooled remotes are fields of records, reached with the record's
    // provenance (`remote_of`).
    unsafe { rm.byte_sub(core::mem::offset_of!(Record, remote)).cast() }
}

fn remote_of(r: *mut Record) -> *mut Remote {
    // SAFETY: a field of a live record; no reference is made.
    unsafe { &raw mut (*r).remote }
}

/// Whether the parked record `r` holds no live blocks. Under the pool lock:
/// parked records change only under it.
fn holds_nothing(r: *mut Record) -> bool {
    // SAFETY: records are never freed; caller holds the pool lock.
    unsafe { (*(*r).inner.get()).stats.allocated_bytes == 0 }
}

thread_local! {
    /// The current thread's record, or the empty heap before its first use.
    /// Const and without a destructor, so reading it is a plain thread-local
    /// load.
    static CURRENT: Cell<*mut Record> = const { Cell::new((&raw const EMPTY.0).cast_mut()) };

    /// Destroyed at thread exit; registered on the thread's first use.
    static EXIT: ExitGuard = const { ExitGuard };
}

/// Returns the exiting thread's free memory to the OS. A heap with no live
/// blocks ends with nothing mapped; one with live blocks keeps them, until a
/// later thread-local destructor frees the last one, which trims it. `CURRENT`
/// stays set: other thread-locals' destructors may still allocate and free on
/// this thread after this one ran, and they keep using the same record.
struct ExitGuard;

impl Drop for ExitGuard {
    fn drop(&mut self) {
        exit_thread();
    }
}

/// Parks the current thread's record, if it has one: its batches sent, its
/// inbox drained, then `CURRENT` torn. Runs from `ExitGuard`'s destructor or
/// from `exit_key`'s, whichever comes first; the second finds nothing to do.
fn exit_thread() {
    if let Some(h) = existing() {
        let r = h.cast::<Record>();
        // SAFETY: the current thread's record; nothing else references it now.
        unsafe {
            // Frees batched for other heaps must not die with the thread.
            // Before the pool lock: a flush may lock it to drain a parked heap.
            (*(*r).batch.get()).flush_all();
            let mut g = POOL.lock();
            let rest = (*r).remainder.get();
            (*r).remainder
                .set((*(*r).inner.get()).free_msgs(rest, usize::MAX));
            // Parked first, then drained: a sender that pushes after the
            // drain sees `PARKED` and drains under the lock itself.
            g.park(remote_of(r));
            drain_all(r);
            settle(r);
            drop(g);
        }
        CURRENT.with(|c| c.set(torn()));
    }
}

/// A pthread key whose destructor also parks the record (`exit_thread`).
/// glibc runs Rust's thread-local destructors (`ExitGuard`) before pthread
/// keys' destructors; a thread that first uses the heap in a key's destructor
/// registers `ExitGuard` too late for it to run, and would strand its record.
/// A key set during another key's destructor is run again (up to
/// `PTHREAD_DESTRUCTOR_ITERATIONS` rounds), so this one still parks it.
#[cfg(all(unix, not(miri)))]
mod exit_key {
    use core::sync::atomic::AtomicUsize;
    use core::sync::atomic::Ordering::{AcqRel, Acquire};

    const NONE: usize = usize::MAX;
    static KEY: AtomicUsize = AtomicUsize::new(NONE);

    unsafe extern "C" fn dtor(_: *mut libc::c_void) {
        super::exit_thread();
    }

    fn key() -> libc::pthread_key_t {
        let k = KEY.load(Acquire);
        if k != NONE {
            return k as libc::pthread_key_t;
        }
        let mut new = 0;
        // SAFETY: a plain destructor with the expected signature.
        if unsafe { libc::pthread_key_create(&mut new, Some(dtor)) } != 0 {
            std::process::abort();
        }
        match KEY.compare_exchange(NONE, new as usize, AcqRel, Acquire) {
            Ok(_) => new,
            Err(won) => {
                // SAFETY: our own key, never used.
                unsafe { libc::pthread_key_delete(new) };
                won as libc::pthread_key_t
            }
        }
    }

    /// Arms the key on this thread: its destructor runs at thread exit.
    pub(super) fn arm(r: *mut super::Record) {
        // SAFETY: a live key; a non-null value makes its destructor run.
        unsafe { libc::pthread_setspecific(key(), r.cast()) };
    }
}

/// Offset of a record inside its mapping. The first block of every page is
/// 64 KiB-aligned, and a small class's hottest block sits in the first KiB of
/// a 4 KiB page; a page-aligned record put its queue heads at the same low 12
/// address bits, so every store to that block stalled the next load of the
/// heap (4K aliasing; `lifo_64B` 29 µs against 26.7 at this offset).
const RECORD_OFFSET: usize = 0x840;

const _: () = assert!(
    RECORD_OFFSET >= 1024
        && RECORD_OFFSET.is_multiple_of(64)
        && RECORD_OFFSET + size_of::<Record>() <= PAGE
);

/// Maps, initializes and registers a new record.
/// Registers the `fork` handlers, once: the pool lock is taken before a fork
/// and released after it in the parent and the child, so a child forked while
/// another thread held the lock does not inherit it held forever. In the
/// child, other threads' records stay active with nobody to drain them: their
/// memory is leaked.
fn fork_handlers() {
    #[cfg(all(unix, not(miri)))]
    {
        static DONE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
        unsafe extern "C" fn prepare() {
            // SAFETY: released by `release` on both sides of the fork.
            unsafe { POOL.lock_raw() }
        }
        unsafe extern "C" fn release() {
            // SAFETY: taken by `prepare` just before the fork.
            unsafe { POOL.unlock_raw() }
        }
        if !DONE.swap(true, Relaxed) {
            // SAFETY: plain function pointers with the expected signature.
            unsafe { libc::pthread_atfork(Some(prepare), Some(release), Some(release)) };
        }
    }
}

fn new_record() -> *mut Record {
    fork_handlers();
    // SAFETY: PAGE is a nonzero multiple of itself; PAGE alignment.
    let Some(base) = (unsafe { os::map_aligned(PAGE, PAGE) }) else {
        std::alloc::handle_alloc_error(Layout::new::<Record>())
    };
    // SAFETY: the record fits in the page after `RECORD_OFFSET` (checked above).
    let r = unsafe { base.as_ptr().add(RECORD_OFFSET) }.cast::<Record>();
    let mut inner = Inner::new(HugePages::Auto);
    inner.owner = r.cast(); // the heap's own address: `inner` is the first field
    inner.mail = Some(record_mail);
    // SAFETY: inside a fresh mapping, 64-byte aligned, with room for a record.
    unsafe { r.write(Record::new(inner)) };
    let mut head = RECORDS.load(Ordering::Relaxed);
    loop {
        // SAFETY: `r` is not yet shared; only this thread writes it.
        unsafe { (*r).next = head };
        match RECORDS.compare_exchange_weak(head, r, Ordering::Release, Ordering::Relaxed) {
            Ok(_) => return r,
            Err(h) => head = h,
        }
    }
}

/// First allocation on this thread: adopt a parked record (a warm one if any,
/// else an empty one, else any), else map a new one.
#[cold]
#[inline(never)]
fn init_current() -> *mut Record {
    // Initializing: an allocation made while the record is being found or
    // made (none is expected; this guards re-entry) takes the torn path, a
    // parked record under the pool lock, instead of re-entering setup. The
    // pool lock is not held across anything that allocates.
    CURRENT.with(|c| c.set(torn()));
    let adopted = {
        let mut g = POOL.lock();
        let r = g
            .pop_where(|rm| warm(record_of(rm)))
            .or_else(|| g.pop_where(|rm| holds_nothing(record_of(rm))))
            .or_else(|| g.pop_where(|_| true))
            .map(record_of);
        r.inspect(|&r| unwarm(r))
    };
    set_current(adopted.unwrap_or_else(new_record))
}

fn set_current(r: *mut Record) -> *mut Record {
    CURRENT.with(|c| c.set(r));
    // Registers the exit guard's destructor. Fails if this thread's Rust
    // thread-locals are already being destroyed (first use in a later
    // destructor); `exit_key` parks the record then.
    let _ = EXIT.try_with(|_| ());
    #[cfg(all(unix, not(miri)))]
    exit_key::arm(r);
    r
}

/// The current thread's heap, made on first use. Off the hot paths: those read
/// `CURRENT` themselves and leave the first-use branch as one cold call, so
/// they stay small enough to inline (`<&ShardAlloc as Allocator>` included).
fn current() -> *mut Inner {
    let mut r = CURRENT.with(Cell::get);
    if r == empty() {
        r = init_current();
    }
    debug_assert!(r != torn(), "callers handle the torn-down thread first");
    // A cast, not `UnsafeCell::get` on the field: the heap pointer keeps the
    // whole record's provenance, so `cast::<Record>()` back is in bounds.
    r.cast::<Inner>()
}

/// The current thread's heap, or the empty heap if it has none yet (no live
/// block's owner stamp matches the empty heap).
#[inline(always)]
fn owner_this() -> *mut Inner {
    CURRENT.with(Cell::get).cast::<Inner>()
}

/// `allocate` after the fast path on `this` found no block: the heap's slow
/// path, with the heap made first if the thread is still on the empty heap.
#[inline(always)]
fn allocate_slow(mut this: *mut Inner, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    if this == empty().cast() {
        this = current();
    } else if this == torn().cast() {
        return torn_allocate(layout);
    }

    // SAFETY: the current thread's heap; no other reference is live. The fast
    // path returns a block for zero-sized layouts, so `layout` is nonzero.
    unsafe { (*this).alloc_miss(layout) }
}

/// The current thread's heap, if it has one.
fn existing() -> Option<*mut Inner> {
    let r = CURRENT.with(Cell::get);
    // As in `current`: the record's own provenance.
    (r != empty() && r != torn()).then(|| r.cast::<Inner>())
}

/// Allocates from the current thread's own heap.
///
/// A zero-sized handle: `Vec<T, ShardAlloc>` costs no space for its allocator.
/// Each thread that uses it gets its own heap, made on first use.
///
/// This is the shape of a global allocator, and it pays what one pays on a
/// single thread: a thread-local lookup to allocate and an owner check to free.
/// [`&LocalHeap`](crate::LocalHeap) pays neither, because the heap is a value and
/// the compiler proves the free happens on the owning thread.
///
/// `ShardAlloc` is `Send + Sync`, and so are its collections (when their
/// contents are): a block freed on another thread is batched there and sent
/// home to the heap that made it, one atomic operation per batch.
///
/// ```
/// fn assert_send_sync<T: Send + Sync>() {}
/// assert_send_sync::<Vec<u8, shardheap::ShardAlloc>>();
/// ```
///
/// ```
/// use shardheap::ShardAlloc;
///
/// let mut v: Vec<u64, ShardAlloc> = Vec::new_in(ShardAlloc::new());
/// v.extend(0..1000);
/// assert!(ShardAlloc::stats().allocated_bytes >= 8000);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct ShardAlloc {
    _private: (),
}

impl ShardAlloc {
    /// The handle. All handles on a thread share that thread's heap.
    pub const fn new() -> Self {
        ShardAlloc { _private: () }
    }

    /// Byte counters of the current thread's heap (all zero if it has none yet).
    pub fn stats() -> Stats {
        // SAFETY: the current thread's record; nothing else references it now.
        existing().map_or_else(Stats::default, |h| unsafe { (*h).stats })
    }

    /// Housekeeping for the current thread's heap: sends this thread's batched
    /// frees to their owners, frees everything other threads sent here, then
    /// [`LocalHeap::maintain`](crate::LocalHeap::maintain). Call it when the
    /// thread goes idle: frees from other threads otherwise wait until this
    /// thread allocates again.
    pub fn maintain() {
        if let Some(h) = existing() {
            let r = h.cast::<Record>();
            // SAFETY: as `stats`; owner-only fields of the current record.
            unsafe {
                (*(*r).batch.get()).flush_all();
                drain_all(r);
                (*h).maintain()
            }
        }
    }

    /// The usable size of the block at `ptr`: at least what was asked for, and
    /// exactly the length `allocate` returned. Found from the block's address
    /// alone, without its `Layout` (for C's `malloc_usable_size`).
    ///
    /// # Safety
    /// `ptr` is a live block from a nonzero-size allocation by `ShardAlloc`, on
    /// any thread.
    pub unsafe fn usable_size(ptr: NonNull<u8>) -> usize {
        // SAFETY: caller's contract.
        unsafe { heap::block_of(ptr.as_ptr()).size }
    }

    /// Frees the block at `ptr` without its `Layout`, as C's `free` does: the
    /// size and kind come from the block's metadata. A block of another
    /// thread's heap is sent home, as by `deallocate`.
    ///
    /// # Safety
    /// `ptr` is a live block from a nonzero-size allocation by `ShardAlloc`, on
    /// any thread, and is not used again.
    pub unsafe fn free_unsized(ptr: NonNull<u8>) {
        let this = owner_this();
        // SAFETY: caller's contract; `this` is the empty heap or the current
        // thread's heap, whose owner stamp is its address.
        unsafe {
            Inner::dealloc_unsized(this, ptr.as_ptr(), |owner, m, bytes| {
                remote_free(owner, m, bytes)
            })
        }
    }

    /// [`LocalHeap::trim`](crate::LocalHeap::trim) for the current thread's heap.
    pub fn trim() {
        if let Some(h) = existing() {
            // SAFETY: as `stats`.
            unsafe { (*h).trim() }
        }
    }
}

// SAFETY: forwards to the `Allocator` methods below, which uphold the
// allocator contract; failure is a null pointer.
unsafe impl GlobalAlloc for ShardAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match Allocator::allocate(self, layout) {
            Ok(b) => b.cast::<u8>().as_ptr(),
            Err(AllocError) => ptr::null_mut(),
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: caller's contract: `ptr` is a live block allocated with `layout`.
        unsafe { Allocator::deallocate(self, NonNull::new_unchecked(ptr), layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        match Allocator::allocate_zeroed(self, layout) {
            Ok(b) => b.cast::<u8>().as_ptr(),
            Err(AllocError) => ptr::null_mut(),
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: caller's contract: `ptr` is live with `layout`, and
        // `new_size` rounded to `layout.align()` does not overflow.
        unsafe {
            let p = NonNull::new_unchecked(ptr);
            let new = Layout::from_size_align_unchecked(new_size, layout.align());
            let r = if new_size >= layout.size() {
                Allocator::grow(self, p, layout, new)
            } else {
                Allocator::shrink(self, p, layout, new)
            };
            r.map_or(ptr::null_mut(), |b| b.cast::<u8>().as_ptr())
        }
    }
}

// SAFETY: forwards to the current thread's heap, which upholds the allocator
// contract. A block freed on its allocating thread goes back to that heap; one
// freed on another thread fails the owner check and is sent to the heap its
// owner stamp names, which frees it as its own. Records live for the process,
// so blocks stay valid however long the handle is kept.
unsafe impl Allocator for ShardAlloc {
    #[inline(always)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the current thread's heap with no other reference in use, or
        // the empty heap, whose queues are all null.
        let this = owner_this();
        match unsafe { Inner::alloc_fast(this, layout) } {
            Some(b) => Ok(b),
            None => allocate_slow(this, layout),
        }
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if CURRENT.with(Cell::get) == torn() {
            let b = self.allocate(layout)?;
            // SAFETY: a fresh block of `b.len()` bytes.
            unsafe { ptr::write_bytes(b.as_ptr().cast::<u8>(), 0, b.len()) };
            return Ok(b);
        }
        // SAFETY: as `allocate`.
        unsafe { (*current()).alloc_zeroed(layout) }
    }

    #[inline(always)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // The owner check compares the block's stamp with this value first, and
        // the empty heap (no record) never matches.
        let this = owner_this();
        // SAFETY: caller's contract; `this` is the empty heap or the current
        // thread's heap, whose owner stamp is its address.
        unsafe {
            Inner::dealloc_owned(this, ptr.as_ptr(), layout, |owner, m| {
                remote_free(owner, m, layout.size())
            })
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract.
        unsafe { resize(ptr, old, new) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract; the block holds `b.len() >= old.size()` bytes.
        unsafe {
            let b = resize(ptr, old, new)?;
            let tail = b.as_ptr().cast::<u8>().add(old.size());
            ptr::write_bytes(tail, 0, b.len() - old.size());
            Ok(b)
        }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: caller's contract.
        unsafe { resize(ptr, old, new) }
    }
}

/// `grow`/`shrink` on the current thread: in place when the block is this
/// thread's; a block of another heap is copied into this thread's heap, and the
/// old one is sent home. A thread never resizes another thread's heap.
///
/// # Safety
/// As `Allocator::grow`/`shrink`.
unsafe fn resize(ptr: NonNull<u8>, old: Layout, new: Layout) -> Result<NonNull<[u8]>, AllocError> {
    if CURRENT.with(Cell::get) == torn() {
        // After teardown: a new block and a free, both through the pool lock.
        let a = ShardAlloc::new();
        let b = a.allocate(new)?;
        // SAFETY: caller's contract; disjoint blocks.
        unsafe {
            ptr::copy_nonoverlapping(ptr.as_ptr(), b.as_ptr().cast(), old.size().min(new.size()));
            a.deallocate(ptr, old);
        }
        return Ok(b);
    }
    let this = current();
    // SAFETY: caller's contract; `this` is the current thread's heap.
    unsafe {
        if old.size() == 0 {
            return (*this).resize(ptr.as_ptr(), old, new);
        }
        let (owner, m) = Inner::owner_of(ptr.as_ptr(), old);
        if owner == this.cast() {
            return (*this).resize(ptr.as_ptr(), old, new);
        }
        let b = (*this).alloc(new)?;
        ptr::copy_nonoverlapping(ptr.as_ptr(), b.as_ptr().cast(), old.size().min(new.size()));
        remote_free(owner, m, old.size());
        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pointer handed back from a thread that has exited.
    struct Sent<T>(*const T);
    // SAFETY: only read after the thread that made it is joined.
    unsafe impl<T> Send for Sent<T> {}
    impl<T> Clone for Sent<T> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<T> Copy for Sent<T> {}

    fn record_stats(r: Sent<Record>) -> Stats {
        // SAFETY: records are never freed; the owning thread has exited.
        unsafe { (*(*r.0).inner.get()).stats }
    }

    fn current_record() -> Sent<Record> {
        Sent(CURRENT.with(Cell::get))
    }

    /// The shard tests spawn threads that park and adopt records: one at a
    /// time, so each test sees the pool it set up.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether `s` is what a warm heap holds: nothing allocated, at most one
    /// segment, no cached large mapping.
    fn is_warm(s: Stats) -> bool {
        s.allocated_bytes == 0
            && s.segments <= 1
            && s.mapped_bytes == s.segments * crate::segment::SEGMENT_SIZE
    }

    #[cfg(all(target_os = "linux", not(miri)))]
    #[test]
    fn fork_while_another_thread_holds_the_pool_lock() {
        let _g = serial();
        let b = std::thread::spawn(|| {
            let l = Layout::new::<[u8; 64]>();
            Sent(ShardAlloc::new().allocate(l).unwrap().cast::<u8>().as_ptr())
        })
        .join()
        .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _l = POOL.lock();
            tx.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(300));
        });
        rx.recv().unwrap();
        // SAFETY: fork; the child only allocates, frees and `_exit`s.
        match unsafe { libc::fork() } {
            0 => unsafe {
                let v: Vec<u8, ShardAlloc> = Vec::with_capacity_in(100, ShardAlloc::new());
                drop(v);
                // A block of the exited thread's parked heap: sent there,
                // which takes the pool lock.
                ShardAlloc::free_unsized(NonNull::new(b.0.cast_mut()).unwrap());
                let mut big: Vec<u8, ShardAlloc> = Vec::new_in(ShardAlloc::new());
                big.resize(1 << 20, 1);
                drop(big);
                libc::_exit(0)
            },
            pid => {
                assert!(pid > 0);
                let mut st = 0;
                // SAFETY: waiting for our own child.
                assert_eq!(unsafe { libc::waitpid(pid, &mut st, 0) }, pid);
                assert!(libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0);
            }
        }
        holder.join().unwrap();
        // The child freed its copy; the parent's block is still live.
        // SAFETY: a live block of the exited thread's (parked) heap.
        unsafe { ShardAlloc::free_unsized(NonNull::new(b.0.cast_mut()).unwrap()) };
    }

    fn record_count() -> usize {
        let mut n = 0;
        let mut r = RECORDS.load(Ordering::Acquire);
        while !r.is_null() {
            n += 1;
            // SAFETY: records are never freed; `next` is set before publishing.
            r = unsafe { (*r).next };
        }
        n
    }

    #[test]
    fn threads_that_start_by_freeing_adopt_parked_heaps() {
        let _g = serial();
        let l = Layout::new::<[u8; 1024]>();
        let alloc = move || Sent(ShardAlloc::new().allocate(l).unwrap().cast::<u8>().as_ptr());
        let free = move |b: Sent<u8>| {
            // SAFETY: a live block, allocated with `l`.
            unsafe { ShardAlloc::new().deallocate(NonNull::new(b.0.cast_mut()).unwrap(), l) }
        };
        // larson's pattern: each thread's first operation frees a block an
        // earlier thread allocated, then it allocates one to hand on. Blocks
        // live for 64 generations, so parked heaps keep live blocks.
        let mut carry: std::collections::VecDeque<Sent<u8>> =
            std::thread::spawn(move || (0..64).map(|_| alloc()).collect())
                .join()
                .unwrap();
        let before = record_count();
        for _ in 0..50 {
            carry = std::thread::spawn(move || {
                let mut c = carry;
                free(c.pop_front().unwrap());
                c.push_back(alloc());
                c
            })
            .join()
            .unwrap();
        }
        let made = record_count() - before;
        assert!(made <= 4, "{made} new records for 50 short-lived threads");
        std::thread::spawn(move || carry.into_iter().for_each(free))
            .join()
            .unwrap();
    }

    #[cfg(all(target_os = "linux", not(miri)))]
    #[test]
    fn threads_first_using_the_heap_in_a_key_destructor_park_it() {
        let _g = serial();
        unsafe extern "C" fn dtor(p: *mut libc::c_void) {
            // This thread's first use of `ShardAlloc`, after the Rust
            // thread-locals' destructors have run.
            // SAFETY: a live block, set as this key's value below.
            unsafe { ShardAlloc::free_unsized(NonNull::new(p.cast()).unwrap()) }
        }
        let mut key = 0;
        // SAFETY: a fresh key with a plain destructor.
        assert_eq!(unsafe { libc::pthread_key_create(&mut key, Some(dtor)) }, 0);
        let l = Layout::new::<[u8; 64]>();
        let before = record_count();
        for _ in 0..50 {
            let b = Sent(ShardAlloc::new().allocate(l).unwrap().cast::<u8>().as_ptr());
            std::thread::spawn(move || {
                let b = b;
                // SAFETY: the key is live; the value is freed by its destructor.
                unsafe { libc::pthread_setspecific(key, b.0.cast()) };
            })
            .join()
            .unwrap();
        }
        let made = record_count() - before;
        // SAFETY: no thread uses the key any more.
        unsafe { libc::pthread_key_delete(key) };
        ShardAlloc::maintain();
        assert!(made <= 4, "{made} new records for 50 threads");
    }

    #[test]
    fn usable_size_is_the_block_size() {
        let _g = serial();
        let cases = [
            (1, 1),
            (24, 8),
            (1000, 16),
            (100_000, 64),
            (300_000, 8),
            (300_000, 1 << 20),
        ];
        for (size, align) in cases {
            let l = Layout::from_size_align(size, align).unwrap();
            let b = ShardAlloc::new().allocate(l).unwrap();
            let p = b.cast::<u8>();
            // SAFETY: a live, nonzero-size block of `ShardAlloc`.
            unsafe {
                assert_eq!(ShardAlloc::usable_size(p), b.len(), "{size}/{align}");
                ShardAlloc::free_unsized(p);
            }
        }
        assert_eq!(ShardAlloc::stats().allocated_bytes, 0);
    }

    #[test]
    fn free_unsized_on_another_thread_sends_the_block_home() {
        let _g = serial();
        let blocks: Vec<Sent<u8>> = [64usize, 1024, 4096, 300_000]
            .iter()
            .map(|&s| {
                let l = Layout::from_size_align(s, 8).unwrap();
                Sent(ShardAlloc::new().allocate(l).unwrap().cast::<u8>().as_ptr())
            })
            .collect();
        std::thread::spawn(move || {
            for b in blocks {
                // SAFETY: live blocks of the spawning thread's heap.
                unsafe { ShardAlloc::free_unsized(NonNull::new(b.0.cast_mut()).unwrap()) }
            }
        })
        .join()
        .unwrap();
        ShardAlloc::maintain();
        assert_eq!(ShardAlloc::stats().allocated_bytes, 0);
    }

    #[test]
    fn free_unsized_after_the_owner_exited_frees_once() {
        let _g = serial();
        let (r, b) = std::thread::spawn(|| {
            let b = ShardAlloc::new()
                .allocate(Layout::new::<[u8; 1024]>())
                .unwrap();
            (current_record(), Sent(b.cast::<u8>().as_ptr()))
        })
        .join()
        .unwrap();
        assert!(parked(&r));
        // Freed on a third thread: its exit flushes the batch to the parked
        // heap, which the sender drains under the pool lock.
        std::thread::spawn(move || {
            let b = b; // the whole `Sent`, not its raw-pointer field
            // SAFETY: a live block of the exited thread's (parked) heap.
            unsafe { ShardAlloc::free_unsized(NonNull::new(b.0.cast_mut()).unwrap()) }
        })
        .join()
        .unwrap();
        assert_eq!(record_stats(r).allocated_bytes, 0);
    }

    #[test]
    fn big_frees_travel_listed_in_one_carrier_block() {
        let _g = serial();
        let heap = heap::LocalHeap::new();
        let l = Layout::new::<[u8; 1024]>();
        let keep = heap.allocate(l).unwrap(); // keeps the page in use
        let blocks: Vec<*mut u8> = (0..8)
            .map(|_| heap.allocate(l).unwrap().cast::<u8>().as_ptr())
            .collect();
        let dest = new_record(); // a mailbox only: nothing of its heap is freed
        let mut batch = BatchCache::new();
        // SAFETY: the blocks are live blocks of `heap`, owned here; `dest` is
        // never freed, and its inbox is taken here only.
        unsafe {
            for &b in &blocks {
                ptr::write_bytes(b, 0xAB, l.size());
                batch.add(dest, heap::small_msg(b), l.size());
            }
            batch.flush_all();
            let m = (*dest).remote.take_all();
            assert!(!m.is_null());
            assert!(
                <MsgLink as crate::remote::Link>::next(m).is_null(),
                "one message"
            );
            assert_ne!(m.addr() & heap::CARRIER_MSG, 0);
            let carrier = m.map_addr(|a| a & !heap::CARRIER_MSG);
            for &b in blocks.iter().filter(|&&b| b != carrier) {
                let line = core::slice::from_raw_parts(b, 64);
                assert!(
                    line.iter().all(|&x| x == 0xAB),
                    "a listed block was written"
                );
            }
            assert!(heap.inner().free_msgs(m, 1).is_null());
            assert_eq!(heap.stats().allocated_bytes, l.size());
            heap.deallocate(keep.cast(), l);
        }
        assert_eq!(heap.stats().allocated_bytes, 0);
    }

    #[test]
    fn frees_from_one_page_travel_as_one_message() {
        let _g = serial();
        let heap = heap::LocalHeap::new();
        let l = Layout::new::<[u8; 64]>(); // below CARRIER_MIN: a group
        let keep = heap.allocate(l).unwrap(); // keeps the page in use
        let blocks: Vec<*mut u8> = (0..8)
            .map(|_| heap.allocate(l).unwrap().cast::<u8>().as_ptr())
            .collect();
        let dest = new_record(); // a mailbox only: nothing of its heap is freed
        let mut batch = BatchCache::new();
        // SAFETY: the blocks are live blocks of `heap`, owned here; `dest` is
        // never freed, and its inbox is taken here only.
        unsafe {
            for &b in &blocks {
                batch.add(dest, heap::small_msg(b), l.size());
            }
            batch.flush_all();
            let m = (*dest).remote.take_all();
            assert!(!m.is_null());
            assert!(
                <MsgLink as crate::remote::Link>::next(m).is_null(),
                "one message"
            );
            assert_ne!(m.addr() & heap::GROUP_MSG, 0);
            assert!(heap.inner().free_msgs(m, 1).is_null());
            assert_eq!(heap.stats().allocated_bytes, l.size());
            heap.deallocate(keep.cast(), l);
        }
        assert_eq!(heap.stats().allocated_bytes, 0);
    }

    /// Trims every warm parked heap, so a test starts with none.
    fn cool_all() {
        let mut g = POOL.lock();
        while let Some(rm) = g.pop_where(|rm| warm(record_of(rm))) {
            // SAFETY: under the pool lock; parked again at once.
            unsafe {
                let r = record_of(rm);
                unwarm(r);
                (*(*r).inner.get()).trim();
                g.park(rm);
            }
        }
        assert_eq!(WARM.load(Relaxed), 0);
    }

    fn parked(r: &Sent<Record>) -> bool {
        // SAFETY: records are never freed.
        let rm = unsafe { &raw mut (*r.0.cast_mut()).remote };
        POOL.lock().contains(rm)
    }

    #[test]
    fn exit_keeps_an_empty_heap_warm() {
        let _s = serial();
        cool_all();
        let r = std::thread::spawn(|| {
            let mut small: Vec<u8, ShardAlloc> = Vec::new_in(ShardAlloc::new());
            small.extend_from_slice(&[1; 200]);
            let mut big: Vec<u8, ShardAlloc> = Vec::new_in(ShardAlloc::new());
            big.resize(2 << 20, 2);
            drop((small, big));
            current_record()
        })
        .join()
        .unwrap();
        let s = record_stats(r);
        assert!(parked(&r));
        assert!(
            is_warm(s) && s.segments == 1,
            "one segment kept for the next thread, the large mapping released: {s:?}"
        );
        assert_eq!(WARM.load(Relaxed), 1);
    }

    #[test]
    fn warm_heaps_are_capped_and_adoption_takes_one() {
        let _s = serial();
        cool_all();
        let n = WARM_HEAPS + 2;
        let go = std::sync::Arc::new(std::sync::Barrier::new(n));
        let rs: Vec<Sent<Record>> = (0..n)
            .map(|_| {
                let go = go.clone();
                std::thread::spawn(move || {
                    drop(Box::new_in([1u8; 100], ShardAlloc::new()));
                    go.wait(); // all alive at once: no adoption among them
                    current_record()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect();
        let stats: Vec<Stats> = rs.iter().map(|&r| record_stats(r)).collect();
        assert!(stats.iter().all(|&s| is_warm(s)));
        let kept = stats.iter().filter(|s| s.mapped_bytes > 0).count();
        assert_eq!(kept, WARM_HEAPS, "the rest gave their memory back");
        assert_eq!(WARM.load(Relaxed), WARM_HEAPS);
        std::thread::spawn(|| {
            drop(Box::new_in(1u8, ShardAlloc::new()));
            assert_eq!(WARM.load(Relaxed), WARM_HEAPS - 1, "adopted a warm heap");
            let s = ShardAlloc::stats();
            assert!(s.mapped_bytes > 0);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn exit_keeps_a_heap_with_live_blocks() {
        let _s = serial();
        let (r, p) = std::thread::spawn(|| {
            // The pointer from `allocate` itself, as collections keep it: it has
            // the provenance the allocator needs to find the block's page on
            // free (a pointer taken from a reference to the block would not).
            let l = Layout::new::<[u8; 1000]>();
            let b = ShardAlloc::new().allocate(l).unwrap().cast::<u8>();
            // SAFETY: a fresh block of 1000 bytes.
            unsafe { b.as_ptr().write_bytes(7, 1000) };
            (current_record(), Sent(b.as_ptr().cast_const()))
        })
        .join()
        .unwrap();
        let s = record_stats(r);
        assert!(s.allocated_bytes >= 1000 && s.mapped_bytes > 0);
        assert!(parked(&r), "an exited thread's heap waits in the pool");
        // SAFETY: the leaked block stays mapped with its record.
        let bytes = unsafe { core::slice::from_raw_parts(p.0, 1000) };
        assert!(bytes.iter().all(|&b| b == 7));
        // Freed here, on another thread, into the parked heap: it empties and
        // keeps only what a warm heap keeps.
        // SAFETY: allocated on the exited thread with this layout.
        unsafe {
            ShardAlloc::new().deallocate(
                NonNull::new(p.0.cast_mut()).unwrap(),
                Layout::new::<[u8; 1000]>(),
            )
        };
        ShardAlloc::maintain();
        assert!(is_warm(record_stats(r)), "{:?}", record_stats(r));
    }

    struct LateVec(core::cell::RefCell<Option<Vec<u8, ShardAlloc>>>);
    impl Drop for LateVec {
        fn drop(&mut self) {
            drop(self.0.borrow_mut().take());
        }
    }
    thread_local!(static LATE_SMALL: LateVec = const { LateVec(core::cell::RefCell::new(None)) });
    thread_local!(static LATE_LARGE: LateVec = const { LateVec(core::cell::RefCell::new(None)) });

    #[test]
    fn frees_after_the_exit_guard_still_empty_the_heap() {
        let _s = serial();
        let r = std::thread::spawn(|| {
            // Registered before the exit guard, so destroyed after it (Linux).
            LATE_SMALL.with(|_| ());
            LATE_LARGE.with(|_| ());
            let mut small = Vec::new_in(ShardAlloc::new());
            small.extend_from_slice(&[1u8; 200]);
            let mut large = Vec::new_in(ShardAlloc::new());
            large.resize(1 << 20, 2u8);
            LATE_SMALL.with(|l| *l.0.borrow_mut() = Some(small));
            LATE_LARGE.with(|l| *l.0.borrow_mut() = Some(large));
            current_record()
        })
        .join()
        .unwrap();
        assert!(is_warm(record_stats(r)), "{:?}", record_stats(r));
    }

    #[test]
    fn a_fresh_thread_starts_on_the_empty_heap() {
        let _s = serial();
        std::thread::spawn(|| {
            let empty = (&raw const EMPTY.0).cast_mut();
            assert_eq!(CURRENT.with(Cell::get), empty, "no record before first use");
            assert_eq!(ShardAlloc::stats(), Stats::default());
            drop(Box::new_in(7u32, ShardAlloc::new()));
            let r = CURRENT.with(Cell::get);
            assert_ne!(r, empty, "the first allocation made a record");
            assert_eq!(r.addr() % PAGE, RECORD_OFFSET);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn the_inbox_has_a_cache_line_to_itself() {
        // Other threads push to it; the owner's heap fields are written on
        // every allocation. Sharing a line makes every push a coherence miss.
        let at = std::mem::offset_of!(Record, remote);
        assert_eq!((RECORD_OFFSET + at) % 64, 0, "starts a line");
        assert!(
            std::mem::offset_of!(Record, batch) >= at + 64,
            "nothing owner-written after it on its line"
        );
    }

    #[test]
    fn record_heaps_sit_clear_of_page_start_blocks() {
        let _s = serial();
        // The first block of every page starts 64 KiB-aligned; small classes put
        // their hottest block in the first KiB of a 4 KiB page. The heap's queue
        // heads and counters must not share those low 12 address bits, or each
        // store to that block stalls the next load of the heap (4K aliasing).
        drop(Box::new_in(0u8, ShardAlloc::new()));
        let off = current_record().0.addr() % PAGE;
        assert!(off >= 1024, "record heap at page offset {off:#x}");
        assert!(
            off + size_of::<Record>() <= PAGE,
            "record heap crosses a page"
        );
    }

    #[test]
    fn exited_thread_heap_is_parked_and_adopted() {
        let _s = serial();
        let a = std::thread::spawn(|| {
            drop(Box::new_in([1u8; 100], ShardAlloc::new()));
            current_record()
        })
        .join()
        .unwrap();
        assert!(parked(&a));
        // SAFETY: records are never freed.
        assert!(unsafe { (*a.0).remote.is_parked() });
        let b = std::thread::spawn(|| {
            drop(Box::new_in(1u64, ShardAlloc::new()));
            let r = current_record();
            // SAFETY: as above.
            let active = !unsafe { (*r.0).remote.is_parked() };
            (r, active, CURRENT.with(Cell::get) != empty())
        })
        .join()
        .unwrap();
        assert!(
            b.1 && b.2,
            "the adopting thread owns the record while it runs"
        );
        assert_eq!(
            b.0.0, a.0,
            "the parked heap was adopted, not a new one mapped"
        );
    }

    #[test]
    fn remote_frees_into_a_parked_heap_empty_it() {
        let _s = serial();
        let (tx, rx) = std::sync::mpsc::channel::<Vec<u8, ShardAlloc>>();
        let r = std::thread::spawn(move || {
            let mut big = Vec::new_in(ShardAlloc::new());
            big.resize(1 << 20, 1u8);
            tx.send(big).unwrap();
            for i in 0..1000 {
                let mut v = Vec::new_in(ShardAlloc::new());
                v.resize(8 + i % 500, 2u8);
                tx.send(v).unwrap();
            }
            current_record()
        })
        .join()
        .unwrap();
        assert!(parked(&r));
        assert!(record_stats(Sent(r.0)).allocated_bytes > 1 << 20);
        let r2 = Sent(r.0);
        std::thread::spawn(move || {
            drop(rx.into_iter().collect::<Vec<_>>());
            ShardAlloc::maintain(); // sends the last batches
        })
        .join()
        .unwrap();
        assert!(is_warm(record_stats(r2)), "{:?}", record_stats(r2));
    }

    #[test]
    fn remote_free_racing_owner_exit_frees_once() {
        let _s = serial();
        for round in 0..if cfg!(miri) { 2 } else { 20 } {
            let n = if cfg!(miri) { 30 } else { 1000 };
            let (txs, rxs): (Vec<_>, Vec<_>) = (0..3)
                .map(|_| std::sync::mpsc::channel::<Box<[u8; 48], ShardAlloc>>())
                .unzip();
            let freers: Vec<_> = rxs
                .into_iter()
                .map(|rx| {
                    std::thread::spawn(move || {
                        for b in rx {
                            assert!(b.iter().all(|&x| x == round as u8));
                        }
                    })
                })
                .collect();
            let owner = std::thread::spawn(move || {
                for i in 0..n {
                    txs[i % 3]
                        .send(Box::new_in([round as u8; 48], ShardAlloc::new()))
                        .unwrap();
                }
                current_record() // exits while the freers are still freeing
            });
            let r = owner.join().unwrap();
            for f in freers {
                f.join().unwrap();
            }
            let s = record_stats(r);
            assert_eq!(
                s.allocated_bytes, 0,
                "round {round}: every block freed exactly once"
            );
        }
    }

    #[test]
    fn records_are_distinct_and_registered_under_contention() {
        let _s = serial();
        // Alive at the same time, so none can adopt another's record.
        let all = std::sync::Arc::new(std::sync::Barrier::new(8));
        let rs: Vec<usize> = (0..8)
            .map(|_| {
                let all = all.clone();
                std::thread::spawn(move || {
                    drop(Box::new_in(1u64, ShardAlloc::new()));
                    all.wait();
                    current_record().0.addr()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect();
        let set: std::collections::HashSet<_> = rs.iter().copied().collect();
        assert_eq!(set.len(), 8);
        let mut seen = 0;
        let mut r = RECORDS.load(Ordering::Acquire);
        while !r.is_null() {
            if set.contains(&r.addr()) {
                seen += 1;
            }
            // SAFETY: registered records are never freed.
            r = unsafe { (*r).next };
        }
        assert_eq!(seen, 8);
    }
}
