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

#[cfg(test)]
mod tests {
    use super::*;

    // Force 8-byte alignment so FreeNode writes inside the region are valid.
    #[repr(align(8))]
    struct AlignedHeap<const N: usize>([u8; N]);

    // 4096 bytes per cache × 9 caches = 36 864 bytes total.
    const HEAP_SIZE: usize = 4096 * 9;

    /// Allocating a single object returns a non-null, properly aligned pointer.
    #[test]
    fn simple_allocation() {
        static mut HEAP: AlignedHeap<HEAP_SIZE> = AlignedHeap([0u8; HEAP_SIZE]);
        let mut alloc = SlabAllocator::new();
        unsafe {
            alloc.init(HEAP.0.as_mut_ptr() as usize, HEAP_SIZE);
            // Request 4 bytes — served by the 8-byte size class.
            let _layout = Layout::from_size_align(4, 4).unwrap();
            let ptr = alloc.caches[0].allocate();
            assert!(!ptr.is_null(), "allocation must succeed");
            // Pointer must satisfy the minimum alignment (8 bytes on 64-bit).
            assert_eq!(ptr as usize % 8, 0, "pointer must be 8-byte aligned");
            alloc.caches[0].deallocate(ptr);
        }
    }

    /// Allocating and immediately freeing many objects must not exhaust the cache.
    #[test]
    fn alloc_dealloc_cycle() {
        static mut HEAP: AlignedHeap<HEAP_SIZE> = AlignedHeap([0u8; HEAP_SIZE]);
        let mut alloc = SlabAllocator::new();
        unsafe {
            alloc.init(HEAP.0.as_mut_ptr() as usize, HEAP_SIZE);
            let cache = &mut alloc.caches[1]; // 16-byte class
            for _ in 0..100 {
                let ptr = cache.allocate();
                assert!(!ptr.is_null());
                cache.deallocate(ptr);
            }
        }
    }

    /// Stress test: fill the 64-byte cache completely, then free everything.
    #[test]
    fn many_boxes() {
        static mut HEAP: AlignedHeap<HEAP_SIZE> = AlignedHeap([0u8; HEAP_SIZE]);
        let mut alloc = SlabAllocator::new();
        unsafe {
            alloc.init(HEAP.0.as_mut_ptr() as usize, HEAP_SIZE);
            let cache = &mut alloc.caches[3]; // 64-byte class
            // Each cache chunk = 4096 bytes → 4096 / 64 = 64 objects.
            let mut ptrs = [core::ptr::null_mut::<u8>(); 64];
            for p in ptrs.iter_mut() {
                *p = cache.allocate();
                assert!(!p.is_null(), "cache must have capacity for 64 objects");
            }
            // Cache is now empty — next allocation must return null.
            assert!(cache.allocate().is_null(), "exhausted cache must return null");
            // Free all objects — cache must accept them all back.
            for p in ptrs.iter_mut() {
                cache.deallocate(*p);
            }
            // After freeing, allocation must work again.
            let ptr = cache.allocate();
            assert!(!ptr.is_null(), "cache must be usable after full free");
            cache.deallocate(ptr);
        }
    }

    /// Requesting a size larger than 2048 bytes must return null (not panic).
    #[test]
    fn oversized_request_returns_null() {
        static mut HEAP: AlignedHeap<HEAP_SIZE> = AlignedHeap([0u8; HEAP_SIZE]);
        let alloc = LockedSlabAllocator::new();
        unsafe {
            alloc.init(HEAP.0.as_mut_ptr() as usize, HEAP_SIZE);
            let layout = Layout::from_size_align(4096, 8).unwrap();
            let ptr = alloc.alloc(layout);
            assert!(ptr.is_null(), "request > 2048 bytes must return null");
        }
    }

    /// align_up correctness — mirrors section 5 of the research report.
    #[test]
    fn align_up_cases() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(13, 8), 16);
        assert_eq!(align_up(16, 8), 16);
        assert_eq!(align_up(1, 4096), 4096);
    }
}
