//! `malloc`, `free` and the rest of the C allocation API over shardheap's
//! [`ShardAlloc`], built as a shared library for `LD_PRELOAD`:
//!
//! ```text
//! RUSTFLAGS='-Z tls-model=initial-exec' cargo +nightly build --release -p shardheap-malloc
//! LD_PRELOAD=target/release/libshardheap_malloc.so some-program
//! ```
//!
//! Built with the initial-exec TLS model, the library reads its thread-local
//! without a call to `__tls_get_addr`; it must then be loaded at startup
//! (`LD_PRELOAD` or linked), not with `dlopen`. A panic cannot unwind out of
//! these functions: `extern "C"` aborts instead.

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::{self, NonNull};

use libc::{c_int, c_void, size_t};
use shardheap::ShardAlloc;

/// `max_align_t` on the supported targets.
const MIN_ALIGN: usize = 16;
const PAGE: usize = 4096;

static A: ShardAlloc = ShardAlloc::new();

fn set_errno(e: c_int) {
    // SAFETY: the calling thread's errno.
    unsafe { *libc::__errno_location() = e };
}

/// `size` bytes aligned to `align` (at least 16), or null with `ENOMEM`.
fn alloc(size: usize, align: usize, zeroed: bool) -> *mut c_void {
    let Ok(l) = Layout::from_size_align(size.max(1), align.max(MIN_ALIGN)) else {
        set_errno(libc::ENOMEM);
        return ptr::null_mut();
    };
    // SAFETY: a nonzero layout.
    let p = unsafe {
        if zeroed {
            A.alloc_zeroed(l)
        } else {
            A.alloc(l)
        }
    };
    if p.is_null() {
        set_errno(libc::ENOMEM);
    }
    p.cast()
}

/// # Safety
/// The C contract of `malloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc(size: size_t) -> *mut c_void {
    alloc(size, MIN_ALIGN, false)
}

/// # Safety
/// The C contract of `free`: `p` is null or a live block from this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free(p: *mut c_void) {
    if let Some(p) = NonNull::new(p.cast::<u8>()) {
        // SAFETY: caller's contract.
        unsafe { ShardAlloc::free_unsized(p) }
    }
}

/// # Safety
/// The C contract of `calloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn calloc(n: size_t, size: size_t) -> *mut c_void {
    match n.checked_mul(size) {
        Some(total) => alloc(total, MIN_ALIGN, true),
        None => {
            set_errno(libc::ENOMEM);
            ptr::null_mut()
        }
    }
}

/// In place when `size` fits the block and uses at least half of it; else a
/// new block, the bytes copied, the old block freed. `realloc(p, 0)` frees `p`
/// and returns null, as glibc does.
///
/// # Safety
/// The C contract of `realloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn realloc(p: *mut c_void, size: size_t) -> *mut c_void {
    let Some(old) = NonNull::new(p.cast::<u8>()) else {
        return alloc(size, MIN_ALIGN, false);
    };
    // SAFETY: caller's contract: `p` is a live block from this library.
    unsafe {
        if size == 0 {
            ShardAlloc::free_unsized(old);
            return ptr::null_mut();
        }
        let usable = ShardAlloc::usable_size(old);
        if size <= usable && size >= usable / 2 {
            return p;
        }
        let new = alloc(size, MIN_ALIGN, false);
        if !new.is_null() {
            ptr::copy_nonoverlapping(old.as_ptr(), new.cast::<u8>(), usable.min(size));
            ShardAlloc::free_unsized(old);
        }
        new
    }
}

/// # Safety
/// The C contract of `reallocarray`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reallocarray(p: *mut c_void, n: size_t, size: size_t) -> *mut c_void {
    match n.checked_mul(size) {
        // SAFETY: caller's contract.
        Some(total) => unsafe { realloc(p, total) },
        None => {
            set_errno(libc::ENOMEM);
            ptr::null_mut()
        }
    }
}

/// # Safety
/// The C contract of `posix_memalign`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn posix_memalign(
    out: *mut *mut c_void,
    align: size_t,
    size: size_t,
) -> c_int {
    if !align.is_power_of_two() || !align.is_multiple_of(size_of::<*mut c_void>()) {
        return libc::EINVAL;
    }
    let p = alloc(size, align, false);
    if p.is_null() {
        return libc::ENOMEM;
    }
    // SAFETY: caller's contract: `out` is writable.
    unsafe { *out = p };
    0
}

/// # Safety
/// The C contract of `aligned_alloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aligned_alloc(align: size_t, size: size_t) -> *mut c_void {
    if !align.is_power_of_two() {
        set_errno(libc::EINVAL);
        return ptr::null_mut();
    }
    alloc(size, align, false)
}

/// Like glibc, rounds an alignment that is not a power of two up to one.
///
/// # Safety
/// The C contract of `memalign`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memalign(align: size_t, size: size_t) -> *mut c_void {
    match align.checked_next_power_of_two() {
        Some(a) => alloc(size, a, false),
        None => {
            set_errno(libc::EINVAL);
            ptr::null_mut()
        }
    }
}

/// # Safety
/// The C contract of `valloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn valloc(size: size_t) -> *mut c_void {
    alloc(size, PAGE, false)
}

/// # Safety
/// The C contract of `pvalloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pvalloc(size: size_t) -> *mut c_void {
    match size.max(1).checked_next_multiple_of(PAGE) {
        Some(s) => alloc(s, PAGE, false),
        None => {
            set_errno(libc::ENOMEM);
            ptr::null_mut()
        }
    }
}

/// # Safety
/// The C contract of `malloc_usable_size`: `p` is null or a live block from
/// this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc_usable_size(p: *mut c_void) -> size_t {
    match NonNull::new(p.cast::<u8>()) {
        // SAFETY: caller's contract.
        Some(p) => unsafe { ShardAlloc::usable_size(p) },
        None => 0,
    }
}

/// Returns this thread's free memory to the OS ([`ShardAlloc::trim`]).
///
/// # Safety
/// None beyond the C contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn malloc_trim(_pad: size_t) -> c_int {
    ShardAlloc::trim();
    1
}
