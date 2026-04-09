#![no_std]

use core::alloc::{GlobalAlloc, Layout};
use core::ptr;

mod slab;
pub use slab::{SlabAllocator, SlabCache, SIZE_CLASSES};

/// A [`SlabAllocator`] protected by a spinlock for use as a `#[global_allocator]`.
///
/// `GlobalAlloc::alloc` and `GlobalAlloc::dealloc` receive `&self` (shared
/// reference), so interior mutability is required to update the free-lists.
/// A `spin::Mutex` provides that without depending on the OS — safe for
/// `no_std` kernel code running before any scheduler exists.
///
/// # Registration
///
/// ```ignore
/// #[global_allocator]
/// static ALLOCATOR: LockedSlabAllocator = LockedSlabAllocator::new();
/// ```
///
/// Then call `ALLOCATOR.init(heap_start, heap_size)` once the heap is mapped.
pub struct LockedSlabAllocator(spin::Mutex<SlabAllocator>);

impl LockedSlabAllocator {
    /// Creates a new allocator with all caches empty.
    ///
    /// No memory is consumed at this point.  Must be followed by a call to
    /// [`LockedSlabAllocator::init`] before any allocation can succeed.
    ///
    /// # Examples
    ///
    /// ```
    /// use slab_allocator::LockedSlabAllocator;
    ///
    /// let alloc = LockedSlabAllocator::new();
    /// // Allocator exists but has no backing memory yet.
    /// ```
    pub const fn new() -> Self {
        LockedSlabAllocator(spin::Mutex::new(SlabAllocator::new()))
    }

    /// Partitions `heap_size` bytes starting at `heap_start` across all
    /// size-class caches, making the allocator ready to serve requests.
    ///
    /// Delegates directly to [`SlabAllocator::init`].
    ///
    /// # Safety
    ///
    /// Same contract as [`SlabAllocator::init`]: the memory region must be
    /// valid, exclusively owned, and writable for the lifetime of this
    /// allocator.
    pub unsafe fn init(&self, heap_start: usize, heap_size: usize) {
        self.0.lock().init(heap_start, heap_size);
    }
}

unsafe impl GlobalAlloc for LockedSlabAllocator {
    /// Allocates a block of memory satisfying `layout`.
    ///
    /// Dispatches to the smallest size class that is large enough to hold
    /// `ceil(layout.size(), layout.align())` bytes.  Returns `null` if no
    /// suitable class exists (request > 2048 bytes) or the matching cache
    /// is exhausted.
    ///
    /// # Safety
    ///
    /// Follows the contract of [`GlobalAlloc::alloc`]: the returned pointer
    /// is valid for `layout.size()` bytes and aligned to `layout.align()`.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Round the requested size up to the required alignment so we always
        // return a pointer with the correct alignment guarantee.
        let needed = align_up(layout.size(), layout.align());

        // Find the smallest size class that fits.
        let index = SIZE_CLASSES.iter().position(|&c| c >= needed);

        match index {
            None => ptr::null_mut(), // request too large — OOM
            Some(i) => self.0.lock().caches[i].allocate(),
        }
    }

    /// Returns a previously allocated block to its size-class cache.
    ///
    /// The correct cache is identified by rounding `layout.size()` up to
    /// `layout.align()` and finding the matching class — the same logic used
    /// in [`GlobalAlloc::alloc`].
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by `alloc` with the same `layout`.
    /// - `ptr` must not be used after this call.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let needed = align_up(layout.size(), layout.align());

        if let Some(i) = SIZE_CLASSES.iter().position(|&c| c >= needed) {
            self.0.lock().caches[i].deallocate(ptr);
        }
        // If no class matches the size was > 2048 and was never allocated
        // through us — silently ignore (mirrors null return in alloc).
    }
}

/// Aligns `addr` upward to the nearest multiple of `align`.
///
/// This is the foundational primitive for the slab allocator: every object
/// and every slab page must start at an address satisfying the required
/// alignment constraint.  The implementation relies on a standard bitmask
/// trick that is valid **only when `align` is a power of two** — a property
/// always guaranteed by [`core::alloc::Layout::align`].
///
/// # Algorithm
///
/// ```text
/// aligned = (addr + align - 1) & !(align - 1)
/// ```
///
/// `align - 1` produces a mask of the low bits that must be zero.
/// Inverting it (`!`) gives a mask that zeroes those bits.
/// Adding `align - 1` before masking ensures we round *up*, not down.
///
/// # Examples
///
/// ```
/// use slab_allocator::align_up;
///
/// // 13 is not a multiple of 8 — rounds up to 16.
/// assert_eq!(align_up(13, 8), 16);
///
/// // Already aligned — stays the same.
/// assert_eq!(align_up(16, 8), 16);
///
/// // Zero address always stays zero.
/// assert_eq!(align_up(0, 64), 0);
///
/// // Works for any power-of-two alignment.
/// assert_eq!(align_up(1, 4096), 4096);
/// ```
///
/// # Panics
///
/// Does not panic, but produces incorrect results if `align` is **not** a
/// power of two.  Callers are responsible for upholding this invariant.
pub fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}
