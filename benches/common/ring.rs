//! A bounded queue that waits by spinning, for the cross-thread benchmarks.
//!
//! A blocking channel puts a consumer that keeps up to sleep, and the
//! producers then pay a futex wake-up on almost every send: the timings measure
//! the channel's sleeping, and a faster allocator can lose. Here nobody sleeps
//! (a waiter yields after a while, for oversubscribed machines).
//!
//! Dmitry Vyukov's bounded MPMC queue, used with one consumer.

use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::thread::yield_now;

#[repr(align(128))]
struct Padded<T>(T);

struct Slot<T> {
    /// `pos`: free for the send at `pos`; `pos + 1`: full, for the receive at
    /// `pos`.
    seq: AtomicUsize,
    val: UnsafeCell<MaybeUninit<T>>,
}

pub struct Ring<T> {
    slots: Box<[Slot<T>]>,
    mask: usize,
    tail: Padded<AtomicUsize>,
    head: Padded<AtomicUsize>,
}

// SAFETY: a value moves from one sender to the receiver through a slot, handed
// over by the slot's sequence number (release/acquire).
unsafe impl<T: Send> Send for Ring<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for Ring<T> {}

fn wait(spins: &mut u32) {
    if *spins < 1 << 12 {
        spin_loop();
    } else {
        yield_now();
    }
    *spins += 1;
}

impl<T> Ring<T> {
    /// A ring of `cap` slots, a power of two.
    pub fn new(cap: usize) -> Ring<T> {
        assert!(cap.is_power_of_two());
        Ring {
            slots: (0..cap)
                .map(|i| Slot {
                    seq: AtomicUsize::new(i),
                    val: UnsafeCell::new(MaybeUninit::uninit()),
                })
                .collect(),
            mask: cap - 1,
            tail: Padded(AtomicUsize::new(0)),
            head: Padded(AtomicUsize::new(0)),
        }
    }

    /// Sends `v`, spinning while the ring is full.
    pub fn send(&self, v: T) {
        let mut spins = 0;
        let mut pos = self.tail.0.load(Relaxed);
        loop {
            let s = &self.slots[pos & self.mask];
            let seq = s.seq.load(Acquire);
            if seq == pos {
                match self
                    .tail
                    .0
                    .compare_exchange_weak(pos, pos + 1, Relaxed, Relaxed)
                {
                    Ok(_) => {
                        // SAFETY: the slot is free and claimed by this send.
                        unsafe { (*s.val.get()).write(v) };
                        s.seq.store(pos + 1, Release);
                        return;
                    }
                    Err(p) => pos = p,
                }
            } else {
                if seq < pos {
                    wait(&mut spins); // full
                }
                pos = self.tail.0.load(Relaxed);
            }
        }
    }

    /// Receives the next value, spinning while the ring is empty. One thread
    /// receives.
    pub fn recv(&self) -> T {
        let pos = self.head.0.load(Relaxed);
        let s = &self.slots[pos & self.mask];
        let mut spins = 0;
        while s.seq.load(Acquire) != pos + 1 {
            wait(&mut spins);
        }
        // SAFETY: the slot is full (its sequence says so) and only this
        // thread receives.
        let v = unsafe { (*s.val.get()).assume_init_read() };
        s.seq.store(pos + self.mask + 1, Release);
        self.head.0.store(pos + 1, Relaxed);
        v
    }
}

impl<T> Drop for Ring<T> {
    fn drop(&mut self) {
        let (head, tail) = (*self.head.0.get_mut(), *self.tail.0.get_mut());
        for pos in head..tail {
            // SAFETY: every slot between head and tail is full (no thread is
            // left to send or receive).
            unsafe { self.slots[pos & self.mask].val.get_mut().assume_init_drop() };
        }
    }
}
