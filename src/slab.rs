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

/// A cache managing free objects of a single fixed size.
///
/// Each `SlabCache` corresponds to one *size class* (e.g. 8, 16, 32 … 2048
/// bytes).  It maintains a singly-linked free-list whose nodes are stored
/// **inside the free objects themselves** — the SLUB *freelist intra-objet*
/// technique described in the research report (section 3).
///
/// # Structure layout
///
/// ```text
/// SlabCache
/// ├── object_size : usize          — bytes per object (e.g. 64)
/// └── free_list   : *mut FreeNode  — head of the free-object chain
///       │
///       ▼
///     [ FreeNode ] → [ FreeNode ] → [ FreeNode ] → null
///      (free obj)     (free obj)     (free obj)
/// ```
///
/// When `allocate` is called the head node is popped and its raw pointer
/// returned to the caller.  When `deallocate` is called the pointer is pushed
/// back as a new head — O(1) in both directions, no lock needed at this
/// level (locking is handled by the outer `LockedSlabAllocator`).
pub struct SlabCache {
    /// Size in bytes of every object managed by this cache.
    ///
    /// Must be at least `size_of::<FreeNode>()` (8 bytes on 64-bit) and
    /// must match the size class this cache was created for.
    pub object_size: usize,

    /// Head of the intrusive free-list.
    ///
    /// `null` means the cache is empty and `grow` must be called before the
    /// next allocation can succeed.
    pub free_list: *mut FreeNode,
}

/// Top-level slab allocator holding one [`SlabCache`] per size class.
///
/// Size classes: 8, 16, 32, 64, 128, 256, 512, 1024, 2048 bytes.
/// Allocation requests are dispatched to the smallest class that fits.
pub struct SlabAllocator {
    /// One cache per size class — index 0 → 8 B, index 8 → 2048 B.
    pub caches: [SlabCache; 9],
}
