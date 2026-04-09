/// A node embedded inside a free (unallocated) object.
///
/// This is the direct Rust equivalent of the *freelist intra-objet* technique
/// used by the Linux SLUB allocator: instead of maintaining a separate array
/// of free-object pointers, the pointer to the *next* free object is written
/// **into the object's own memory**.  When the object is allocated, that
/// memory is handed to the caller and the pointer is no longer needed.
///
/// # Memory layout
///
/// ```text
/// free object bytes
/// ┌──────────────────────────────────┐
/// │ next: *mut FreeNode  (8 bytes)   │  ← FreeNode lives here
/// │ ... rest unused until allocated  │
/// └──────────────────────────────────┘
/// ```
///
/// This means every size class must be **at least `size_of::<FreeNode>()`
/// bytes** (i.e. 8 bytes on a 64-bit target), which is already satisfied by
/// the smallest class in our design (8 bytes).
///
/// # Safety
///
/// `FreeNode` is only ever constructed by writing into raw memory obtained
/// from a slab page.  It must never be created on the stack or inside a
/// `Box` — doing so would defeat the purpose and corrupt allocator state.
pub struct FreeNode {
    /// Pointer to the next free object in the same slab, or `null` if this
    /// is the last free slot.
    pub next: *mut FreeNode,
}

/// Core slab allocator — inspired by the Linux SLUB design.
///
/// Manages a fixed-size free-list of objects carved out of a contiguous
/// memory region.  Each instance handles exactly **one size class**.
pub struct SlabAllocator;
