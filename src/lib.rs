#![no_std]

mod slab;
pub use slab::{SlabAllocator, SlabCache, SIZE_CLASSES};

/// Global allocator instance.
///
/// Registered via `#[global_allocator]` in the kernel entry point.
/// All heap types (`Box`, `Vec`, etc.) will route through this.
pub struct LockedSlabAllocator(spin::Mutex<Option<SlabAllocator>>);

impl LockedSlabAllocator {
    /// Creates a new uninitialized allocator.
    pub const fn new() -> Self {
        LockedSlabAllocator(spin::Mutex::new(None))
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
