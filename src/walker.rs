//! Generic page-table walker.
//!
//! [`PageTableWalker<Meta, Entry, Alloc>`] is the single walker reused across
//! every arch and every consumer (CPU paging in kraph, future IOMMU
//! contexts via vendor Entry impls). All semantic policy lives in the
//! trait impls — this module only implements the *mechanism* of
//! descending the tree, allocating intermediates, swapping leaves,
//! and reclaiming frames.
//!
//! # Correctness invariants
//!
//! * **Frame layout**: every page-table frame is sized by metadata and
//!   holds `Meta::ENTRIES_PER_TABLE` entries of type `Entry`. The walker
//!   treats each frame as a `Entry` array via `Alloc::phys_to_virt`.
//!   `Entry` must be `u64`-sized/aligned and the table must fit in
//!   `Meta::TABLE_FRAME_SIZE`.
//!
//! * **Leaf vs intermediate**: `Entry::entry_kind(level)` decodes whether
//!   a present entry is a table pointer or a terminal leaf/block at the
//!   current level. This is level-aware so CPU, IOMMU, and SMMU formats
//!   can reject encodings that are reserved at a particular level.
//!
//! * **Drop reclaim**: [`Drop`] walks the entire tree depth-first,
//!   deallocating every intermediate + the root via `Alloc::deallocate`.
//!   Leaf-pointed frames (user data) are *not*
//!   touched — those are the consumer's lifetime to manage. This
//!   preserves the semantics kraph today inherits from
//!   `page_table_multiarch::PageTable64::Drop`.

use core::marker::PhantomData;
use core::mem::{align_of, size_of};

use heapless::Vec;
use memory_addr::{AddrRange, MemoryAddr, PhysAddr, PhysAddrRange};

use crate::{
    AddrSpaceActivation, FrameAllocator, IntoMapBacking, MapBacking, Mapping, MappingContiguity,
    MappingFlags, PageSize, PageTableEntry, PageTableEntryKind, PagingError, PagingMetaData,
    PagingResult, TlbInvalidation,
};

/// Maximum stack depth for the walker's parent-tracking. Sized to cover
/// all currently-supported page-table heights (x86 4-level + 5-level,
/// aarch64 4-level, riscv Sv39/Sv48/Sv57, AMD-Vi up to 6 levels).
const MAX_LEVELS: usize = 6;

/// Page-table operations over a typed virtual address space.
///
/// This is the public trait form of the walker surface. Concrete CPU
/// page tables use `memory_addr::VirtAddr`; IOMMU consumers can bind
/// `PagingMetaData::VirtAddr` to their own typed IOVA address.
pub trait PageTable<V: MemoryAddr> {
    const INPUT_ADDR_BITS: u8;
    const OUTPUT_ADDR_BITS: u8;

    type Entry: PageTableEntry;

    fn root(&self) -> PhysAddr;

    /// Install this page table into an architecture-specific activation
    /// policy. The returned token can later be made active through
    /// [`AddrSpaceActivation::activate`].
    #[inline]
    fn install_with<A>(&self, activation: &A, controls: A::Controls) -> PagingResult<A::Token>
    where
        A: AddrSpaceActivation,
    {
        activation.install(self.root(), controls)
    }

    fn query(&self, vaddr: V) -> PagingResult<Mapping<Self::Entry, V>>;

    fn map<'a, B, F, Tlb>(
        &mut self,
        range: AddrRange<V>,
        backing: B,
        flags: F,
        tlb: &Tlb,
    ) -> PagingResult
    where
        B: IntoMapBacking<'a>,
        F: Into<MappingFlags<<Self::Entry as PageTableEntry>::Flags>>,
        Tlb: TlbInvalidation<V>;

    fn remap<Tlb>(
        &mut self,
        range: AddrRange<V>,
        paddr: PhysAddr,
        flags: <Self::Entry as PageTableEntry>::Flags,
        tlb: &Tlb,
    ) -> PagingResult<Mapping<Self::Entry, V>>
    where
        Tlb: TlbInvalidation<V>;

    fn protect<Tlb>(
        &mut self,
        range: AddrRange<V>,
        flags: <Self::Entry as PageTableEntry>::Flags,
        tlb: &Tlb,
    ) -> PagingResult<Mapping<Self::Entry, V>>
    where
        Tlb: TlbInvalidation<V>;

    fn unmap<Tlb>(
        &mut self,
        range: AddrRange<V>,
        tlb: &Tlb,
    ) -> PagingResult<Mapping<Self::Entry, V>>
    where
        Tlb: TlbInvalidation<V>;

    fn split_at<Tlb>(&mut self, range: AddrRange<V>, tlb: &Tlb) -> PagingResult<PageSize>
    where
        Tlb: TlbInvalidation<V>;

    fn merge_at<Tlb>(&mut self, range: AddrRange<V>, tlb: &Tlb) -> PagingResult<PageSize>
    where
        Tlb: TlbInvalidation<V>;
}

/// Generic page-table walker.
///
/// Consumers parameterize over per-arch [`PagingMetaData`], per-arch or
/// per-vendor [`PageTableEntry`], and a static [`FrameAllocator`] that
/// supplies + reclaims intermediate-table frames.
pub struct PageTableWalker<Meta, Entry, Alloc>
where
    Meta: PagingMetaData,
    Entry: PageTableEntry,
    Alloc: FrameAllocator,
{
    root: PhysAddr,
    _p: PhantomData<(Meta, Entry, Alloc)>,
}

// SAFETY: PageTableWalker owns a physical frame address; the actual table
// memory is reached only through `Alloc::phys_to_virt` which returns a
// VirtAddr the consumer guarantees to be valid for the lifetime of the
// PageTableWalker. No interior mutability is shared across the boundary.
unsafe impl<Meta, Entry, Alloc> Send for PageTableWalker<Meta, Entry, Alloc>
where
    Meta: PagingMetaData,
    Entry: PageTableEntry,
    Alloc: FrameAllocator,
{
}

unsafe impl<Meta, Entry, Alloc> Sync for PageTableWalker<Meta, Entry, Alloc>
where
    Meta: PagingMetaData,
    Entry: PageTableEntry,
    Alloc: FrameAllocator,
{
}

impl<Meta, Entry, Alloc> PageTableWalker<Meta, Entry, Alloc>
where
    Meta: PagingMetaData,
    Entry: PageTableEntry,
    Alloc: FrameAllocator,
{
    /// Allocate a fresh root frame and return a walker bound to it.
    /// The frame is zeroed — `Alloc::allocate()` must guarantee that
    /// per the trait contract.
    ///
    pub fn try_new() -> PagingResult<Self> {
        assert_walker_layout::<Meta, Entry>();
        let root = alloc_table_frame::<Meta, Alloc>()?;
        Ok(Self {
            root,
            _p: PhantomData,
        })
    }

    /// Construct a walker over an *existing* root table the kernel
    /// already owns — the bootloader-provided root, an address space
    /// inherited at fork, etc. Unlike [`Self::try_new`] this does not
    /// allocate; the caller asserts the root is a valid, zeroed-or-
    /// populated `Meta`-shaped table and that the underlying frames
    /// will not be released while this walker holds them.
    ///
    /// The walker takes over mapping/unmapping the existing tree as if
    /// it had built it. New internal nodes added by subsequent map
    /// operations come from `Alloc`; existing nodes are left as-is.
    pub fn adopt(root: PhysAddr) -> Self {
        assert_walker_layout::<Meta, Entry>();
        Self {
            root,
            _p: PhantomData,
        }
    }

    /// Physical address of the root table — used by the consumer to
    /// program the hardware translation register (CR3, TTBR1_EL1, SATP).
    #[inline]
    pub fn root(&self) -> PhysAddr {
        self.root
    }

    #[inline]
    fn cursor<'pt, Tlb>(
        &'pt mut self,
        tlb: &'pt Tlb,
    ) -> PageTableCursor<'pt, Meta, Entry, Alloc, Tlb>
    where
        Tlb: TlbInvalidation<Meta::VirtAddr>,
    {
        PageTableCursor::new(self, tlb)
    }

    /// Look up the leaf mapping covering `vaddr`.
    pub fn query(&self, vaddr: Meta::VirtAddr) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        check_vaddr::<Meta>(vaddr)?;

        let mut current = self.root;
        let mut level = Meta::LEVELS as u8;

        loop {
            let pte =
                unsafe { read_entry::<Entry, Alloc>(current, index_at::<Meta>(level, vaddr)) };

            if !pte.is_present() {
                return Err(PagingError::NotMapped);
            }

            // Terminal: explicit leaf flag (block at intermediate level)
            // OR innermost level reached.
            if entry_is_leaf::<Entry>(pte, level)? {
                let size = Meta::leaf_size_at_level(level).ok_or(PagingError::UnsupportedSize)?;
                let range = leaf_range::<Meta>(vaddr, size)?;
                return Ok(Mapping::new(range, pte.paddr(), pte.flags()));
            }

            current = pte.paddr();
            level -= 1;
        }
    }

    fn exact_leaf_slot(
        &self,
        range: AddrRange<Meta::VirtAddr>,
    ) -> PagingResult<(PhysAddr, usize, Entry, PageSize)> {
        let vaddr = range.start;
        check_vaddr::<Meta>(vaddr)?;

        let mut current = self.root;
        let mut level = Meta::LEVELS as u8;

        loop {
            let idx = index_at::<Meta>(level, vaddr);
            let pte = unsafe { read_entry::<Entry, Alloc>(current, idx) };

            if !pte.is_present() {
                return Err(PagingError::NotMapped);
            }

            if entry_is_leaf::<Entry>(pte, level)? {
                let size = Meta::leaf_size_at_level(level).ok_or(PagingError::UnsupportedSize)?;
                if leaf_range::<Meta>(vaddr, size)? != range {
                    return Err(PagingError::NotMapped);
                }
                return Ok((current, idx, pte, size));
            }

            current = pte.paddr();
            level -= 1;
        }
    }

    /// Install fresh mappings over a virtually-contiguous range.
    ///
    /// `backing` supplies either one physical range or an ordered
    /// scatter list. [`MappingFlags::contiguity`] decides whether the
    /// request requires one contiguous extent or maps every scattered
    /// leaf at an explicit granule.
    pub fn map<'a, B, F>(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        backing: B,
        flags: F,
        tlb: &impl TlbInvalidation<Meta::VirtAddr>,
    ) -> PagingResult
    where
        B: IntoMapBacking<'a>,
        F: Into<MappingFlags<Entry::Flags>>,
    {
        let flags = flags.into();
        let backing = backing.into_map_backing(range.size())?;
        if let (MappingContiguity::Contiguous, MapBacking::Contiguous(phys)) =
            (flags.contiguity(), backing)
        {
            if phys.is_empty() || range.size() != phys.size() {
                return Err(PagingError::InvalidMappingShape);
            }
            if let Some((size, target_level)) = exact_leaf_size::<Meta>(range, phys.start, None)? {
                self.map_leaf_direct_sized_no_flush(
                    range,
                    phys.start,
                    flags.leaf(),
                    size,
                    target_level,
                )?;
                self.flush_leaf_range(range, tlb);
                return Ok(());
            }
        }

        let mut cursor = self.cursor(tlb);
        cursor.map(range, backing, flags)?;
        cursor.finish()
    }

    fn map_leaf_direct_sized_no_flush(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        paddr: PhysAddr,
        flags: Entry::Flags,
        size: PageSize,
        target_level: u8,
    ) -> PagingResult {
        let vaddr = range.start;

        let mut current = self.root;
        let mut level = Meta::LEVELS as u8;

        while level > target_level {
            let idx = index_at::<Meta>(level, vaddr);
            let pte = unsafe { read_entry::<Entry, Alloc>(current, idx) };

            current = if pte.is_present() {
                if entry_is_leaf::<Entry>(pte, level)? {
                    return Err(PagingError::AlreadyMapped);
                }
                pte.paddr()
            } else {
                let child = alloc_table_frame::<Meta, Alloc>()?;
                unsafe {
                    write_entry::<Entry, Alloc>(current, idx, Entry::new_table(child, level));
                }
                child
            };

            level -= 1;
        }

        let idx = index_at::<Meta>(level, vaddr);
        let existing = unsafe { read_entry::<Entry, Alloc>(current, idx) };
        if existing.is_present() {
            return Err(PagingError::AlreadyMapped);
        }

        unsafe {
            write_entry::<Entry, Alloc>(current, idx, Entry::new_leaf(paddr, flags, size));
        }
        Ok(())
    }

    /// Replace the physical address and flags of an exact existing
    /// leaf, keeping the leaf size unchanged.
    pub fn remap(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        paddr: PhysAddr,
        flags: Entry::Flags,
        tlb: &impl TlbInvalidation<Meta::VirtAddr>,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        let mapping = self.remap_no_flush(range, paddr, flags)?;
        self.flush_leaf_range(mapping.range, tlb);
        Ok(mapping)
    }

    pub(crate) fn remap_no_flush(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        paddr: PhysAddr,
        flags: Entry::Flags,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        check_paddr::<Meta>(paddr)?;
        let (current, idx, _, size) = self.exact_leaf_slot(range)?;
        if !size.is_aligned(paddr.as_usize()) {
            return Err(PagingError::NotAligned);
        }
        unsafe {
            write_entry::<Entry, Alloc>(current, idx, Entry::new_leaf(paddr, flags, size));
        }
        Ok(Mapping::new(range, paddr, flags))
    }

    /// Modify only the flags of an exact existing leaf, keeping paddr
    /// and size.
    pub fn protect(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        flags: Entry::Flags,
        tlb: &impl TlbInvalidation<Meta::VirtAddr>,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        let mapping = self.protect_no_flush(range, flags)?;
        self.flush_leaf_range(mapping.range, tlb);
        Ok(mapping)
    }

    pub(crate) fn protect_no_flush(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        flags: Entry::Flags,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        let (current, idx, pte, size) = self.exact_leaf_slot(range)?;
        let paddr = pte.paddr();
        unsafe {
            write_entry::<Entry, Alloc>(current, idx, Entry::new_leaf(paddr, flags, size));
        }
        Ok(Mapping::new(range, paddr, flags))
    }

    /// Remove the exact leaf at `range`. After clearing the leaf, walk
    /// back up the parent stack and reclaim any intermediate tables
    /// whose entries are now all unused — preserving the same
    /// "tight tree" semantics kraph inherits from upstream today.
    pub fn unmap(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        tlb: &impl TlbInvalidation<Meta::VirtAddr>,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        let result = self.unmap_no_flush(range)?;
        self.flush_leaf_range(result.range, tlb);
        Ok(result)
    }

    pub(crate) fn unmap_no_flush(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        let vaddr = range.start;
        check_vaddr::<Meta>(vaddr)?;

        // Stack of (table_phys, slot_index) for every table walked
        // above the leaf. After clearing the leaf, we walk back up
        // until we hit a still-populated intermediate.
        let mut parents: [(PhysAddr, usize); MAX_LEVELS] =
            [(PhysAddr::from_usize(0), 0); MAX_LEVELS];
        let mut depth: usize = 0;

        let mut current = self.root;
        let mut level = Meta::LEVELS as u8;

        let result = loop {
            let idx = index_at::<Meta>(level, vaddr);
            let pte = unsafe { read_entry::<Entry, Alloc>(current, idx) };

            if !pte.is_present() {
                return Err(PagingError::NotMapped);
            }

            if entry_is_leaf::<Entry>(pte, level)? {
                // Clear the leaf
                let paddr = pte.paddr();
                let flags = pte.flags();
                let size = Meta::leaf_size_at_level(level).ok_or(PagingError::UnsupportedSize)?;
                let leaf_range = leaf_range::<Meta>(vaddr, size)?;
                if leaf_range != range {
                    return Err(PagingError::NotMapped);
                }
                let mut cleared = pte;
                cleared.clear();
                unsafe {
                    write_entry::<Entry, Alloc>(current, idx, cleared);
                }
                break (Mapping::new(leaf_range, paddr, flags), current);
            }

            if depth >= MAX_LEVELS {
                return Err(PagingError::AddressOutOfRange);
            }
            parents[depth] = (current, idx);
            depth += 1;
            current = pte.paddr();
            level -= 1;
        };

        let (mapping, mut child_table) = result;

        // Walk back up: dealloc intermediate tables that are now empty.
        // Stop as soon as we find a table that still holds at least one
        // present entry — that means a sibling mapping is still alive
        // and we must not free its containing table.
        while depth > 0 {
            depth -= 1;
            let (parent_phys, parent_idx) = parents[depth];

            if !is_table_empty::<Meta, Entry, Alloc>(child_table) {
                break;
            }

            // Clear the parent's pointer to the now-orphaned child.
            unsafe {
                let mut zero = Entry::from_bits(0);
                zero.clear(); // belt-and-suspenders; from_bits(0) should already be unused.
                write_entry::<Entry, Alloc>(parent_phys, parent_idx, zero);
            }
            dealloc_table_frame::<Meta, Alloc>(child_table)?;
            child_table = parent_phys;
        }

        Ok(mapping)
    }

    /// Split the exact leaf at `range` into one table of entries one level
    /// finer. Returns the new leaf size. Errors if the entry is not
    /// present or already at the finest granularity.
    pub fn split_at(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        tlb: &impl TlbInvalidation<Meta::VirtAddr>,
    ) -> PagingResult<PageSize> {
        let vaddr = range.start;
        check_vaddr::<Meta>(vaddr)?;

        let mut current = self.root;
        let mut level = Meta::LEVELS as u8;

        loop {
            let idx = index_at::<Meta>(level, vaddr);
            let pte = unsafe { read_entry::<Entry, Alloc>(current, idx) };

            if !pte.is_present() {
                return Err(PagingError::NotMapped);
            }

            // Only block-style leaves can be split. If we hit a leaf at
            // level 1 it's already 4K — nothing to do.
            if entry_is_leaf::<Entry>(pte, level)? {
                if level == 1 {
                    return Err(PagingError::NotMapped);
                }

                let leaf_paddr = pte.paddr();
                let leaf_flags = pte.flags();
                let leaf_size =
                    Meta::leaf_size_at_level(level).ok_or(PagingError::UnsupportedSize)?;
                if leaf_range::<Meta>(vaddr, leaf_size)? != range {
                    return Err(PagingError::NotMapped);
                }
                let finer_level = level - 1;
                let finer_size =
                    Meta::leaf_size_at_level(finer_level).ok_or(PagingError::UnsupportedSize)?;

                // Allocate child table and populate with 512 finer leaves.
                let child = alloc_table_frame::<Meta, Alloc>()?;
                for i in 0..Meta::ENTRIES_PER_TABLE {
                    let sub_paddr = leaf_paddr + i * finer_size.bytes();
                    let entry = Entry::new_leaf(sub_paddr, leaf_flags, finer_size);
                    unsafe {
                        write_entry::<Entry, Alloc>(child, i, entry);
                    }
                }

                // Atomic 8-byte swap: parent slot now points at the new
                // table. Concurrent walkers see either the old block
                // leaf or the new table — both translate to identical
                // (paddr, flags, granule).
                unsafe {
                    write_entry::<Entry, Alloc>(current, idx, Entry::new_table(child, level));
                }

                self.flush_leaf_range(range, tlb);

                return Ok(finer_size);
            }

            // Already at a 4K leaf — caller's expectation that this is
            // a block-style entry doesn't hold.
            if level == 1 {
                return Err(PagingError::NotMapped);
            }

            current = pte.paddr();
            level -= 1;
        }
    }

    /// Coalesce all finer-grained sibling leaves in `range` into a
    /// single block leaf.
    ///
    /// Preconditions verified by this method:
    /// * `range.start` is aligned to `range.size()`.
    /// * The level above the children holds a non-leaf entry pointing
    ///   at a child table.
    /// * All child entries are present and leaves.
    /// * All children share identical [`PageTableEntry::Flags`].
    /// * All children are physically contiguous starting at a
    ///   range-size-aligned base address.
    ///
    /// Any failure → [`PagingError::NotCoalescable`] with the tree
    /// unmodified.
    pub fn merge_at(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        tlb: &impl TlbInvalidation<Meta::VirtAddr>,
    ) -> PagingResult<PageSize> {
        let target_size = check_leaf_range::<Meta>(range)?;
        let vaddr = range.start;

        let target_level = Meta::size_to_level(target_size).ok_or(PagingError::UnsupportedSize)?;
        if target_level <= 1 {
            // 4K is the finest granule — nothing to merge into.
            return Err(PagingError::NotCoalescable);
        }

        // Descend to the table containing the entry at target_level.
        let mut current = self.root;
        let mut level = Meta::LEVELS as u8;
        while level > target_level {
            let idx = index_at::<Meta>(level, vaddr);
            let pte = unsafe { read_entry::<Entry, Alloc>(current, idx) };
            if !pte.is_present() || entry_is_leaf::<Entry>(pte, level)? {
                return Err(PagingError::NotMapped);
            }
            current = pte.paddr();
            level -= 1;
        }

        // The slot at `target_level`. If it's already a leaf at this
        // size, nothing to do.
        let idx = index_at::<Meta>(target_level, vaddr);
        let parent_pte = unsafe { read_entry::<Entry, Alloc>(current, idx) };
        if !parent_pte.is_present() {
            return Err(PagingError::NotMapped);
        }
        if entry_is_leaf::<Entry>(parent_pte, level)? {
            return Ok(target_size);
        }

        let child_table_phys = parent_pte.paddr();
        let finer_level = target_level - 1;
        let finer_size =
            Meta::leaf_size_at_level(finer_level).ok_or(PagingError::UnsupportedSize)?;

        // Verify all 512 children meet the merge contract.
        let first = unsafe { read_entry::<Entry, Alloc>(child_table_phys, 0) };
        if !first.is_present() {
            return Err(PagingError::NotCoalescable);
        }
        // Children must be leaves at finer_level.
        if !entry_is_leaf::<Entry>(first, finer_level)? {
            return Err(PagingError::NotCoalescable);
        }

        let base_paddr = first.paddr();
        let base_flags = first.flags();
        if !target_size.is_aligned(base_paddr.as_usize()) {
            return Err(PagingError::NotCoalescable);
        }

        for i in 1..Meta::ENTRIES_PER_TABLE {
            let pte = unsafe { read_entry::<Entry, Alloc>(child_table_phys, i) };
            if !pte.is_present() {
                return Err(PagingError::NotCoalescable);
            }
            if !entry_is_leaf::<Entry>(pte, finer_level)? {
                return Err(PagingError::NotCoalescable);
            }
            if pte.flags() != base_flags {
                return Err(PagingError::NotCoalescable);
            }
            let expected = base_paddr.as_usize() + i * finer_size.bytes();
            if pte.paddr().as_usize() != expected {
                return Err(PagingError::NotCoalescable);
            }
        }

        // All checks passed — install the merged leaf and reclaim the
        // child table. The parent's 8-byte slot transitions atomically
        // from "table pointer" to "block leaf"; concurrent walkers see
        // either form and translate identically.
        unsafe {
            write_entry::<Entry, Alloc>(
                current,
                idx,
                Entry::new_leaf(base_paddr, base_flags, target_size),
            );
        }
        dealloc_table_frame::<Meta, Alloc>(child_table_phys)?;
        self.flush_leaf_range(range, tlb);

        Ok(target_size)
    }

    #[inline]
    fn flush_leaf_range(
        &self,
        range: AddrRange<Meta::VirtAddr>,
        tlb: &impl TlbInvalidation<Meta::VirtAddr>,
    ) {
        let count_pages = range.size() / Meta::BASE_PAGE_SIZE.bytes();
        if tlb.prefer_full_flush(count_pages) {
            tlb.flush_tlb_all_local();
        } else {
            tlb.flush_tlb_range_local(range.start, Meta::BASE_PAGE_SIZE, count_pages);
        }
    }
}

impl<Meta, Entry, Alloc> PageTable<Meta::VirtAddr> for PageTableWalker<Meta, Entry, Alloc>
where
    Meta: PagingMetaData,
    Entry: PageTableEntry,
    Alloc: FrameAllocator,
{
    const INPUT_ADDR_BITS: u8 = Meta::VA_MAX_BITS as u8;
    const OUTPUT_ADDR_BITS: u8 = Meta::PA_MAX_BITS as u8;

    type Entry = Entry;

    #[inline]
    fn root(&self) -> PhysAddr {
        PageTableWalker::root(self)
    }

    #[inline]
    fn query(&self, vaddr: Meta::VirtAddr) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        PageTableWalker::query(self, vaddr)
    }

    #[inline]
    fn map<'a, B, F, Tlb>(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        backing: B,
        flags: F,
        tlb: &Tlb,
    ) -> PagingResult
    where
        B: IntoMapBacking<'a>,
        F: Into<MappingFlags<Entry::Flags>>,
        Tlb: TlbInvalidation<Meta::VirtAddr>,
    {
        PageTableWalker::map(self, range, backing, flags, tlb)
    }

    #[inline]
    fn remap<Tlb>(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        paddr: PhysAddr,
        flags: Entry::Flags,
        tlb: &Tlb,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>>
    where
        Tlb: TlbInvalidation<Meta::VirtAddr>,
    {
        PageTableWalker::remap(self, range, paddr, flags, tlb)
    }

    #[inline]
    fn protect<Tlb>(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        flags: Entry::Flags,
        tlb: &Tlb,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>>
    where
        Tlb: TlbInvalidation<Meta::VirtAddr>,
    {
        PageTableWalker::protect(self, range, flags, tlb)
    }

    #[inline]
    fn unmap<Tlb>(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        tlb: &Tlb,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>>
    where
        Tlb: TlbInvalidation<Meta::VirtAddr>,
    {
        PageTableWalker::unmap(self, range, tlb)
    }

    #[inline]
    fn split_at<Tlb>(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        tlb: &Tlb,
    ) -> PagingResult<PageSize>
    where
        Tlb: TlbInvalidation<Meta::VirtAddr>,
    {
        PageTableWalker::split_at(self, range, tlb)
    }

    #[inline]
    fn merge_at<Tlb>(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        tlb: &Tlb,
    ) -> PagingResult<PageSize>
    where
        Tlb: TlbInvalidation<Meta::VirtAddr>,
    {
        PageTableWalker::merge_at(self, range, tlb)
    }
}

impl<Meta, Entry, Alloc> Drop for PageTableWalker<Meta, Entry, Alloc>
where
    Meta: PagingMetaData,
    Entry: PageTableEntry,
    Alloc: FrameAllocator,
{
    /// Recursive subtree reclaim. Walks the entire tree depth-first
    /// and returns every intermediate + the root frame to the
    /// allocator. Leaf-pointed frames (user data) are *not* freed —
    /// the consumer manages their lifetime separately.
    fn drop(&mut self) {
        unsafe {
            drop_subtree::<Meta, Entry, Alloc>(self.root, Meta::LEVELS as u8);
        }
    }
}

// ── Cursor (batch API) ──────────────────────────────────────────────────────

/// Maximum number of disjoint flush ranges a cursor stores before
/// falling back to a full local-TLB flush. This is storage capacity, not
/// a hardware policy threshold; [`TlbInvalidation::prefer_full_flush`]
/// decides when full flush is cheaper for a successfully queued batch.
const FLUSH_RANGE_CAP: usize = 128;
const LEAF_SIZES_DESC: [PageSize; 10] = [
    PageSize::Size4T,
    PageSize::Size512G,
    PageSize::Size64G,
    PageSize::Size1G,
    PageSize::Size512M,
    PageSize::Size32M,
    PageSize::Size2M,
    PageSize::Size64K,
    PageSize::Size16K,
    PageSize::Size4K,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FlushRange {
    start: usize,
    pages: usize,
}

impl FlushRange {
    #[inline]
    fn end(self, page_size: PageSize) -> Option<usize> {
        self.pages
            .checked_mul(page_size.bytes())
            .and_then(|bytes| self.start.checked_add(bytes))
    }
}

/// Internal cursor for batched mapping operations.
///
/// A cursor caches the descent (the chain of frame pointers from root
/// down to the leaf-bearing table) used by the previous operation. As
/// long as a follow-up operation's vaddr shares a high-level prefix, we
/// reuse the cached descent and skip the corresponding root-side reads.
///
/// Bulk-mapping a 1 GiB region as 4 KiB pages with a fresh
/// [`PageTableWalker::map`] call per entry costs `262144 × LEVELS` Entry reads;
/// with a cursor it costs roughly `262144 + (LEVELS - 1) × outer-table-
/// crossings`, plus a single batched TLB flush at the end.
///
/// TLB flushes are batched as coalesced base-granule ranges. Contiguous
/// mutations usually collapse to one pending range regardless of page
/// count; highly fragmented batches fall back to a full local TLB flush
/// only when the range queue overflows or the TLB policy says a full
/// flush is cheaper. The cursor flushes automatically on `Drop`.
///
/// # Lifetime
///
/// The cursor borrows the [`PageTableWalker`] mutably for its lifetime; while
/// the cursor exists, the table cannot be mutated through any other
/// path. This guarantees the cached descent stays valid.
///
/// # Errors recover the descent
///
/// If a cursor op returns `Err`, the descent cache is **invalidated** —
/// the next op re-walks fully. This costs the cursor's caching benefit
/// for that one op but ensures we never act on stale cached state after
/// a partial failure.
struct PageTableCursor<'pt, Meta, Entry, Alloc, Tlb>
where
    Meta: PagingMetaData,
    Entry: PageTableEntry,
    Alloc: FrameAllocator,
    Tlb: TlbInvalidation<Meta::VirtAddr>,
{
    pt: &'pt mut PageTableWalker<Meta, Entry, Alloc>,
    tlb: &'pt Tlb,
    /// Frames cached by the previous walk, indexed `[level - 1]`:
    ///   `descent[L - 1] = Some(frame_phys)` means we know the table at
    ///   level `L` for the previous vaddr's path.
    /// The root is not stored here; it is read from `pt.root` directly.
    descent: [Option<PhysAddr>; MAX_LEVELS],
    /// Last vaddr we mutated; `None` for a freshly-opened cursor or
    /// after an error. Lets us compute the LCA on the next op.
    last_vaddr: Option<usize>,
    /// Pending coalesced 4 KiB-granule ranges.
    pending_flushes: Vec<FlushRange, FLUSH_RANGE_CAP, u8>,
    /// True once the range queue overflowed; finish/Drop emits a full
    /// local-TLB flush instead of range invalidations.
    needs_full_flush: bool,
}

impl<'pt, Meta, Entry, Alloc, Tlb> PageTableCursor<'pt, Meta, Entry, Alloc, Tlb>
where
    Meta: PagingMetaData,
    Entry: PageTableEntry,
    Alloc: FrameAllocator,
    Tlb: TlbInvalidation<Meta::VirtAddr>,
{
    fn new(pt: &'pt mut PageTableWalker<Meta, Entry, Alloc>, tlb: &'pt Tlb) -> Self {
        // Root is never stored in `descent` — the walker reads it
        // directly from `pt.root` on every walk start. Only levels
        // 1..LEVELS-1 (intermediate tables) get cached; the slot at
        // `descent[LEVELS - 1]` (which would be root) stays `None`.
        Self {
            pt,
            tlb,
            descent: [None; MAX_LEVELS],
            last_vaddr: None,
            pending_flushes: Vec::new(),
            needs_full_flush: false,
        }
    }

    /// Deepest cached table that remains selected by the requested
    /// `vaddr`. If the first differing index is level `D`, the table at
    /// level `D` is still selected by identical higher-level indices;
    /// only child tables below `D` must be re-walked.
    fn reusable_level(&self, vaddr: usize) -> u8 {
        let Some(last) = self.last_vaddr else {
            // Fresh cursor: no non-root table is cached.
            return Meta::LEVELS as u8;
        };
        for level in (1..=Meta::LEVELS as u8).rev() {
            let shift = Meta::level_shift(level);
            let mask = index_mask::<Meta>(level);
            if ((last >> shift) & mask) != ((vaddr >> shift) & mask) {
                return level;
            }
        }
        // Identical vaddr: all cached tables are reusable.
        1
    }

    /// Invalidate cache slots below `level`.
    fn invalidate_below(&mut self, level: u8) {
        let max = (level as usize).min(MAX_LEVELS);
        for slot in &mut self.descent[..max.saturating_sub(1)] {
            *slot = None;
        }
    }

    /// Queue a base-granule TLB flush range, merging overlapping or
    /// adjacent ranges in-place. If the fragmented range set exceeds
    /// [`FLUSH_RANGE_CAP`], escalate to a full local flush.
    fn queue_flush_range(&mut self, range: AddrRange<Meta::VirtAddr>) {
        if self.needs_full_flush {
            return;
        }
        let page_size = Meta::BASE_PAGE_SIZE;
        let pages = range.size() / page_size.bytes();
        if pages == 0 {
            return;
        }

        let mut start = <Meta::VirtAddr as Into<usize>>::into(range.start);
        let Some(mut end) = pages
            .checked_mul(page_size.bytes())
            .and_then(|bytes| start.checked_add(bytes))
        else {
            self.needs_full_flush = true;
            self.pending_flushes.clear();
            return;
        };

        let mut idx = 0;
        while idx < self.pending_flushes.len() {
            let range = self.pending_flushes[idx];
            let Some(range_end) = range.end(page_size) else {
                self.needs_full_flush = true;
                self.pending_flushes.clear();
                return;
            };

            if end < range.start || range_end < start {
                idx += 1;
                continue;
            }

            start = start.min(range.start);
            end = end.max(range_end);
            self.pending_flushes.swap_remove(idx);
        }

        let merged_pages = (end - start) / page_size.bytes();
        if self
            .pending_flushes
            .push(FlushRange {
                start,
                pages: merged_pages,
            })
            .is_err()
        {
            self.needs_full_flush = true;
            self.pending_flushes.clear();
        }
    }

    /// Emit accumulated TLB flushes. Called by `finish` and by `Drop`.
    ///
    /// Decision: full flush iff the coalesced range queue overflowed
    /// [`FLUSH_RANGE_CAP`] (cap reached) **or** the TLB layer's
    /// [`TlbInvalidation::prefer_full_flush`] policy says so for the
    /// accumulated base-granule count. Otherwise issue range flushes.
    fn flush_pending(&mut self) {
        let count = self
            .pending_flushes
            .iter()
            .fold(0usize, |acc, range| acc.saturating_add(range.pages));
        if self.needs_full_flush {
            self.tlb.flush_tlb_all_local();
            self.pending_flushes.clear();
            self.needs_full_flush = false;
            return;
        }
        if count == 0 {
            return;
        }
        if self.tlb.prefer_full_flush(count) {
            self.tlb.flush_tlb_all_local();
        } else {
            self.pending_flushes
                .sort_unstable_by_key(|range| range.start);

            for range in self.pending_flushes.iter().copied() {
                self.tlb.flush_tlb_range_local(
                    Meta::VirtAddr::from(range.start),
                    Meta::BASE_PAGE_SIZE,
                    range.pages,
                );
            }
        }
        self.pending_flushes.clear();
        self.needs_full_flush = false;
    }

    /// Drain pending TLB invalidations and consume the cursor.
    ///
    /// Equivalent to letting the cursor drop, but lets the caller
    /// observe (currently-vacuous) errors and is the preferred form
    /// when the caller cares about the explicit flush boundary.
    fn finish(mut self) -> PagingResult {
        self.flush_pending();
        Ok(())
    }

    /// Map a virtually-contiguous range using cached descent where
    /// possible.
    ///
    /// Same semantics as [`PageTableWalker::map`] (refuses on conflict, etc.)
    /// but TLB flush is queued for batched emission rather than fired
    /// immediately.
    fn map<'a, B, F>(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        backing: B,
        flags: F,
    ) -> PagingResult
    where
        B: IntoMapBacking<'a>,
        F: Into<MappingFlags<Entry::Flags>>,
    {
        let flags = flags.into();
        let backing = backing.into_map_backing(range.size())?;
        match (flags.contiguity(), backing) {
            (MappingContiguity::Contiguous, MapBacking::Contiguous(phys)) => {
                self.map_contiguous(range, phys, flags.leaf(), None)
            }
            (MappingContiguity::Contiguous, MapBacking::Scattered(_)) => {
                Err(PagingError::InvalidMappingShape)
            }
            (MappingContiguity::Scattered(granule), backing) => {
                self.map_scattered(range, backing, flags.leaf(), granule)
            }
        }
    }

    fn map_contiguous(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        phys: PhysAddrRange,
        flags: Entry::Flags,
        fixed: Option<PageSize>,
    ) -> PagingResult {
        if range.is_empty() || phys.is_empty() || range.size() != phys.size() {
            return Err(PagingError::InvalidMappingShape);
        }
        if let Some((size, target_level)) = exact_leaf_size::<Meta>(range, phys.start, fixed)? {
            self.map_leaf_sized_at_level(range, phys.start, flags, size, target_level)?;
            return Ok(());
        }

        let mut offset = 0usize;
        let total = range.size();
        while offset < total {
            let vaddr = offset_vaddr::<Meta>(range.start, offset)?;
            let paddr = offset_paddr(phys.start, offset)?;
            let remaining = total - offset;
            let size = match fixed {
                Some(size) => {
                    if remaining < size.bytes() {
                        return Err(PagingError::InvalidMappingShape);
                    }
                    size
                }
                None => best_leaf_size::<Meta>(vaddr, paddr, remaining)?,
            };
            let leaf = AddrRange::try_from_start_size(vaddr, size.bytes())
                .ok_or(PagingError::AddressOutOfRange)?;
            self.map_leaf(leaf, paddr, flags)?;
            offset += size.bytes();
        }
        Ok(())
    }

    fn map_scattered(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        backing: MapBacking<'_>,
        flags: Entry::Flags,
        granule: PageSize,
    ) -> PagingResult {
        if range.is_empty() || !granule.is_aligned(range.size()) {
            return Err(PagingError::InvalidMappingShape);
        }
        if Meta::size_to_level(granule).is_none() {
            return Err(PagingError::UnsupportedSize);
        }

        match backing {
            MapBacking::Contiguous(phys) => {
                self.check_scatter_segment(phys, granule)?;
                if phys.size() != range.size() {
                    return Err(PagingError::InvalidMappingShape);
                }
                self.map_contiguous(range, phys, flags, Some(granule))?;
            }
            MapBacking::Scattered(ranges) => {
                if ranges.is_empty() {
                    return Err(PagingError::InvalidMappingShape);
                }
                let mut total = 0usize;
                for phys in ranges.iter().copied() {
                    self.check_scatter_segment(phys, granule)?;
                    let Some(end) = total.checked_add(phys.size()) else {
                        return Err(PagingError::InvalidMappingShape);
                    };
                    if end > range.size() {
                        return Err(PagingError::InvalidMappingShape);
                    }
                    total = end;
                }
                if total != range.size() {
                    return Err(PagingError::InvalidMappingShape);
                }

                let mut offset = 0usize;
                for phys in ranges.iter().copied() {
                    let vaddr = offset_vaddr::<Meta>(range.start, offset)?;
                    let vrange = AddrRange::try_from_start_size(vaddr, phys.size())
                        .ok_or(PagingError::AddressOutOfRange)?;
                    self.map_contiguous(vrange, phys, flags, Some(granule))?;
                    offset += phys.size();
                }
            }
        }
        Ok(())
    }

    fn check_scatter_segment(&self, phys: PhysAddrRange, granule: PageSize) -> PagingResult {
        if phys.is_empty()
            || !granule.is_aligned(phys.start.as_usize())
            || !granule.is_aligned(phys.size())
        {
            Err(PagingError::InvalidMappingShape)
        } else {
            Ok(())
        }
    }

    /// Map one leaf using cached descent where possible.
    fn map_leaf(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        paddr: PhysAddr,
        flags: Entry::Flags,
    ) -> PagingResult {
        let size = check_leaf_range::<Meta>(range)?;
        let vaddr = range.start;
        check_paddr::<Meta>(paddr)?;
        let v: usize = vaddr.into();
        if !size.is_aligned(v) || !size.is_aligned(paddr.as_usize()) {
            return Err(PagingError::NotAligned);
        }
        self.map_leaf_sized(range, paddr, flags, size)
    }

    fn map_leaf_sized(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        paddr: PhysAddr,
        flags: Entry::Flags,
        size: PageSize,
    ) -> PagingResult {
        let target_level = Meta::size_to_level(size).ok_or(PagingError::UnsupportedSize)?;
        self.map_leaf_sized_at_level(range, paddr, flags, size, target_level)
    }

    fn map_leaf_sized_at_level(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        paddr: PhysAddr,
        flags: Entry::Flags,
        size: PageSize,
        target_level: u8,
    ) -> PagingResult {
        let vaddr = range.start;
        let v: usize = vaddr.into();
        // Reuse cached descent for shared upper levels.
        let reusable_level = self.reusable_level(v);
        self.invalidate_below(reusable_level);

        // Find the deepest cached frame still valid. Walk down from
        // there. Root is always read directly
        // from `self.pt.root` — no caching, no fallible unwrap, since
        // the root frame doesn't change for the lifetime of the table.
        let mut level = Meta::LEVELS as u8;
        let mut current = self.pt.root;
        while level > reusable_level.max(target_level) {
            level -= 1;
            if let Some(frame) = self.descent[(level as usize).saturating_sub(1)] {
                current = frame;
            } else {
                level += 1;
                break;
            }
        }

        // From `current` at `level`, descend to `target_level`.
        while level > target_level {
            let idx = index_at::<Meta>(level, vaddr);
            let pte = unsafe { read_entry::<Entry, Alloc>(current, idx) };
            current = if pte.is_present() {
                if entry_is_leaf::<Entry>(pte, level)? {
                    self.last_vaddr = None; // descent is no longer trustworthy
                    return Err(PagingError::AlreadyMapped);
                }
                pte.paddr()
            } else {
                let child = match alloc_table_frame::<Meta, Alloc>() {
                    Ok(c) => c,
                    Err(e) => {
                        self.last_vaddr = None;
                        return Err(e);
                    }
                };
                unsafe {
                    write_entry::<Entry, Alloc>(current, idx, Entry::new_table(child, level));
                }
                child
            };
            level -= 1;
            // Slot for level L = L - 1; `level` post-decrement is the
            // new (deeper) level we just descended into.
            self.descent[(level as usize).saturating_sub(1)] = Some(current);
        }

        // Install leaf at `current` (the table at `target_level`).
        let idx = index_at::<Meta>(level, vaddr);
        let existing = unsafe { read_entry::<Entry, Alloc>(current, idx) };
        if existing.is_present() {
            self.last_vaddr = None;
            return Err(PagingError::AlreadyMapped);
        }
        unsafe {
            write_entry::<Entry, Alloc>(current, idx, Entry::new_leaf(paddr, flags, size));
        }

        self.last_vaddr = Some(v);
        self.queue_flush_range(range);
        Ok(())
    }

    /// Unmap the leaf covering `vaddr`. As with [`PageTableWalker::unmap`],
    /// emptied intermediates are reclaimed; cached descent for those
    /// frames is invalidated.
    #[allow(dead_code)]
    fn unmap(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        // Falls back to PageTableWalker::unmap-without-flush — descent
        // caching here is messier because reclaim may dealloc cached
        // frames mid-flight. Cleanest: invalidate every cached level
        // below root (root never moves).
        self.invalidate_below(Meta::LEVELS as u8);
        self.last_vaddr = None;

        let result = match self.pt.unmap_no_flush(range) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        self.queue_flush_range(result.range);
        Ok(result)
    }

    /// Batched form of [`PageTableWalker::protect`] — flush is queued.
    #[allow(dead_code)]
    fn protect(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        flags: Entry::Flags,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        let mapping = self.pt.protect_no_flush(range, flags)?;
        self.queue_flush_range(mapping.range);
        Ok(mapping)
    }

    /// Batched form of [`PageTableWalker::remap`] — flush is queued.
    #[allow(dead_code)]
    fn remap(
        &mut self,
        range: AddrRange<Meta::VirtAddr>,
        paddr: PhysAddr,
        flags: Entry::Flags,
    ) -> PagingResult<Mapping<Entry, Meta::VirtAddr>> {
        let mapping = self.pt.remap_no_flush(range, paddr, flags)?;
        self.queue_flush_range(mapping.range);
        Ok(mapping)
    }
}

impl<'pt, Meta, Entry, Alloc, Tlb> Drop for PageTableCursor<'pt, Meta, Entry, Alloc, Tlb>
where
    Meta: PagingMetaData,
    Entry: PageTableEntry,
    Alloc: FrameAllocator,
    Tlb: TlbInvalidation<Meta::VirtAddr>,
{
    fn drop(&mut self) {
        self.flush_pending();
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Compute the table index for `vaddr` at `level`. Honors per-level metadata
/// so non-9-bit granules and narrower root tables work without changing the
/// walker.
#[inline]
fn index_at<Meta: PagingMetaData>(level: u8, vaddr: Meta::VirtAddr) -> usize {
    let v: usize = vaddr.into();
    (v >> Meta::level_shift(level)) & index_mask::<Meta>(level)
}

#[inline]
fn index_mask<Meta: PagingMetaData>(level: u8) -> usize {
    let bits = Meta::level_index_bits(level);
    if bits >= usize::BITS {
        usize::MAX
    } else {
        (1usize << bits) - 1
    }
}

/// Validate `vaddr` against the metadata's input-address range.
#[inline]
fn check_vaddr<Meta: PagingMetaData>(vaddr: Meta::VirtAddr) -> PagingResult {
    if Meta::vaddr_is_valid(vaddr) {
        Ok(())
    } else {
        Err(PagingError::AddressOutOfRange)
    }
}

/// Validate `paddr` against the metadata's physical-address range.
#[inline]
fn check_paddr<Meta: PagingMetaData>(paddr: PhysAddr) -> PagingResult {
    if Meta::paddr_is_valid(paddr) {
        Ok(())
    } else {
        Err(PagingError::AddressOutOfRange)
    }
}

#[inline]
fn check_leaf_range<Meta: PagingMetaData>(
    range: AddrRange<Meta::VirtAddr>,
) -> PagingResult<PageSize> {
    if range.is_empty() {
        return Err(PagingError::UnsupportedSize);
    }
    let size = PageSize::from_bytes(range.size()).ok_or(PagingError::UnsupportedSize)?;
    check_vaddr::<Meta>(range.start)?;
    let last = range
        .end
        .checked_sub(1)
        .ok_or(PagingError::AddressOutOfRange)?;
    check_vaddr::<Meta>(last)?;
    if !size.is_aligned(<Meta::VirtAddr as Into<usize>>::into(range.start)) {
        return Err(PagingError::NotAligned);
    }
    Ok(size)
}

#[inline]
fn leaf_range<Meta: PagingMetaData>(
    vaddr: Meta::VirtAddr,
    size: PageSize,
) -> PagingResult<AddrRange<Meta::VirtAddr>> {
    let base = vaddr.align_down(size.bytes());
    AddrRange::try_from_start_size(base, size.bytes()).ok_or(PagingError::AddressOutOfRange)
}

#[inline]
fn offset_vaddr<Meta: PagingMetaData>(
    base: Meta::VirtAddr,
    offset: usize,
) -> PagingResult<Meta::VirtAddr> {
    <Meta::VirtAddr as Into<usize>>::into(base)
        .checked_add(offset)
        .map(<Meta::VirtAddr as From<usize>>::from)
        .ok_or(PagingError::AddressOutOfRange)
}

#[inline]
fn offset_paddr(base: PhysAddr, offset: usize) -> PagingResult<PhysAddr> {
    base.as_usize()
        .checked_add(offset)
        .map(PhysAddr::from_usize)
        .ok_or(PagingError::AddressOutOfRange)
}

fn best_leaf_size<Meta: PagingMetaData>(
    vaddr: Meta::VirtAddr,
    paddr: PhysAddr,
    remaining: usize,
) -> PagingResult<PageSize> {
    let v = <Meta::VirtAddr as Into<usize>>::into(vaddr);
    let p = paddr.as_usize();
    for size in LEAF_SIZES_DESC {
        if remaining >= size.bytes()
            && size.is_aligned(v)
            && size.is_aligned(p)
            && Meta::size_to_level(size).is_some()
        {
            return Ok(size);
        }
    }
    Err(PagingError::UnsupportedSize)
}

fn exact_leaf_size<Meta: PagingMetaData>(
    range: AddrRange<Meta::VirtAddr>,
    paddr: PhysAddr,
    fixed: Option<PageSize>,
) -> PagingResult<Option<(PageSize, u8)>> {
    let Some(size) = PageSize::from_bytes(range.size()) else {
        return Ok(None);
    };
    if fixed.is_some_and(|fixed| fixed != size) {
        return Ok(None);
    }
    check_vaddr::<Meta>(range.start)?;
    let last = range
        .end
        .checked_sub(1)
        .ok_or(PagingError::AddressOutOfRange)?;
    check_vaddr::<Meta>(last)?;
    if !size.is_aligned(<Meta::VirtAddr as Into<usize>>::into(range.start)) {
        return Err(PagingError::NotAligned);
    }
    check_paddr::<Meta>(paddr)?;
    if !size.is_aligned(paddr.as_usize()) {
        return Err(PagingError::NotAligned);
    }
    let target_level = Meta::size_to_level(size).ok_or(PagingError::UnsupportedSize)?;
    Ok(Some((size, target_level)))
}

#[inline]
fn entry_is_leaf<Entry: PageTableEntry>(pte: Entry, level: u8) -> PagingResult<bool> {
    match pte.entry_kind(level) {
        PageTableEntryKind::Leaf => Ok(true),
        PageTableEntryKind::Table => Ok(false),
        PageTableEntryKind::Invalid => Err(PagingError::InvalidEntryKind),
    }
}

/// Check the unsafe assumptions the pointer-casting walker relies on.
#[inline]
fn assert_walker_layout<Meta: PagingMetaData, Entry: PageTableEntry>() {
    assert_eq!(size_of::<Entry>(), size_of::<u64>());
    assert_eq!(align_of::<Entry>(), align_of::<u64>());
    assert!(Meta::LEVELS <= MAX_LEVELS);
    assert!(Meta::ENTRIES_PER_TABLE * size_of::<Entry>() <= Meta::TABLE_FRAME_SIZE.bytes());
}

#[inline]
fn alloc_table_frame<Meta: PagingMetaData, Alloc: FrameAllocator>() -> PagingResult<PhysAddr> {
    let table_size = Meta::TABLE_FRAME_SIZE;
    let range = Alloc::allocate(table_size.bytes(), table_size)?;
    if range.size() != table_size.bytes() || !table_size.is_aligned(range.start.as_usize()) {
        let _ = Alloc::deallocate(range);
        return Err(PagingError::NotAligned);
    }
    Ok(range.start)
}

#[inline]
fn dealloc_table_frame<Meta: PagingMetaData, Alloc: FrameAllocator>(
    paddr: PhysAddr,
) -> PagingResult {
    Alloc::deallocate(PhysAddrRange::from_start_size(
        paddr,
        Meta::TABLE_FRAME_SIZE.bytes(),
    ))
}

/// SAFETY: caller must ensure `paddr` was obtained from
/// `Alloc::allocate` (or is the live root) and not yet freed, and
/// `idx` is within the `Meta::ENTRIES_PER_TABLE` bound the frame's
/// layout was sized for. Walker callers obtain `idx` from
/// [`index_at`], which masks against `(1 << INDEX_BITS) - 1`, so the
/// bound holds for any well-formed Meta.
#[inline]
unsafe fn read_entry<Entry: PageTableEntry, Alloc: FrameAllocator>(
    paddr: PhysAddr,
    idx: usize,
) -> Entry {
    let table = Alloc::phys_to_virt(paddr).as_usize() as *const Entry;
    unsafe { core::ptr::read(table.add(idx)) }
}

/// SAFETY: same preconditions as [`read_entry`]; additionally, `entry`
/// must be a structurally-valid Entry for the given level (the walker
/// caller is responsible for choosing leaf vs table form).
#[inline]
unsafe fn write_entry<Entry: PageTableEntry, Alloc: FrameAllocator>(
    paddr: PhysAddr,
    idx: usize,
    entry: Entry,
) {
    let table = Alloc::phys_to_virt(paddr).as_usize() as *mut Entry;
    unsafe {
        core::ptr::write(table.add(idx), entry);
    }
}

/// Returns true iff the table at `paddr` has zero present entries.
fn is_table_empty<Meta: PagingMetaData, Entry: PageTableEntry, Alloc: FrameAllocator>(
    paddr: PhysAddr,
) -> bool {
    for i in 0..Meta::ENTRIES_PER_TABLE {
        let pte = unsafe { read_entry::<Entry, Alloc>(paddr, i) };
        if pte.is_present() {
            return false;
        }
    }
    true
}

/// SAFETY: `paddr` must be a live, walker-owned table frame at the
/// given level. After this call, `paddr` is freed and any further
/// access is UB.
unsafe fn drop_subtree<Meta, Entry, Alloc>(paddr: PhysAddr, level: u8)
where
    Meta: PagingMetaData,
    Entry: PageTableEntry,
    Alloc: FrameAllocator,
{
    if level > 1 {
        for i in 0..Meta::ENTRIES_PER_TABLE {
            let pte = unsafe { read_entry::<Entry, Alloc>(paddr, i) };
            if pte.is_present() && pte.entry_kind(level).is_table() {
                // Recurse into intermediate; leaf entries point at user
                // data which the consumer owns.
                unsafe {
                    drop_subtree::<Meta, Entry, Alloc>(pte.paddr(), level - 1);
                }
            }
        }
    }
    let _ = dealloc_table_frame::<Meta, Alloc>(paddr);
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! Walker tests use a host-side mock Entry / Meta / Alloc trio. The
    //! mock Entry has a deliberately simple bit layout so test assertions
    //! can verify exact post-condition state without relying on any
    //! arch-specific encoding.

    use super::*;
    use crate::{AccessFlags, CachePolicy, NoFlush};
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::alloc::{Layout, alloc_zeroed, dealloc};
    use std::collections::HashMap;
    use std::sync::Mutex;

    // ── Mock Entry ────────────────────────────────────────────────────────────

    /// Bit layout (chosen for clarity, not realism):
    /// - bit 0 = PRESENT
    /// - bit 1 = LEAF (intermediate-level block leaf)
    /// - bits 2..6 = AccessFlags
    /// - bits 6..8 = CachePolicy as u2
    /// - bits 12..52 = paddr >> 12 (40-bit PFN)
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    struct MockPte(u64);

    const MOCK_PRESENT: u64 = 1 << 0;
    const MOCK_LEAF: u64 = 1 << 1;
    const MOCK_INVALID: u64 = 1 << 63;
    const MOCK_PADDR_MASK: u64 = 0x000f_ffff_ffff_f000;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct MockFlags {
        access: AccessFlags,
        cache: CachePolicy,
    }

    impl PageTableEntry for MockPte {
        type Flags = MockFlags;

        fn new_leaf(paddr: PhysAddr, flags: MockFlags, _size: PageSize) -> Self {
            let mut bits = (paddr.as_usize() as u64) & MOCK_PADDR_MASK;
            bits |= MOCK_PRESENT | MOCK_LEAF;
            bits |= ((flags.access.bits() as u64) & 0xf) << 2;
            bits |= ((flags.cache as u64) & 0x3) << 6;
            Self(bits)
        }

        fn new_table(paddr: PhysAddr, _level: u8) -> Self {
            Self(((paddr.as_usize() as u64) & MOCK_PADDR_MASK) | MOCK_PRESENT)
        }

        fn paddr(&self) -> PhysAddr {
            PhysAddr::from_usize((self.0 & MOCK_PADDR_MASK) as usize)
        }

        fn flags(&self) -> MockFlags {
            let access_bits = ((self.0 >> 2) & 0xf) as u8;
            let cache_bits = ((self.0 >> 6) & 0x3) as u8;
            // Reconstruct AccessFlags by ORing recognized constants
            let mut access = AccessFlags::empty();
            if access_bits & 0b0001 != 0 {
                access |= AccessFlags::READ;
            }
            if access_bits & 0b0010 != 0 {
                access |= AccessFlags::WRITE;
            }
            if access_bits & 0b0100 != 0 {
                access |= AccessFlags::EXECUTE;
            }
            if access_bits & 0b1000 != 0 {
                access |= AccessFlags::USER;
            }
            let cache = match cache_bits {
                0 => CachePolicy::Writeback,
                1 => CachePolicy::Uncached,
                2 => CachePolicy::WriteCombine,
                _ => CachePolicy::WriteThrough,
            };
            MockFlags { access, cache }
        }

        fn is_present(&self) -> bool {
            self.0 & MOCK_PRESENT != 0
        }

        fn entry_kind(&self, level: u8) -> PageTableEntryKind {
            if self.0 & MOCK_INVALID != 0 {
                PageTableEntryKind::Invalid
            } else if level == 1 || self.0 & MOCK_LEAF != 0 {
                PageTableEntryKind::Leaf
            } else {
                PageTableEntryKind::Table
            }
        }

        fn clear(&mut self) {
            self.0 = 0;
        }

        fn bits(&self) -> u64 {
            self.0
        }

        fn from_bits(bits: u64) -> Self {
            Self(bits)
        }
    }

    // ── Mock Meta ───────────────────────────────────────────────────────────

    struct MockMeta;

    impl PagingMetaData for MockMeta {
        const LEVELS: usize = 4;
        const PA_MAX_BITS: usize = 52;
        const VA_MAX_BITS: usize = 48;

        type VirtAddr = memory_addr::VirtAddr;

        fn level_shift(level: u8) -> u32 {
            12 + ((level as u32) - 1) * 9
        }

        fn level_supports_leaf(level: u8, size: PageSize) -> bool {
            matches!(
                (level, size),
                (1, PageSize::Size4K) | (2, PageSize::Size2M) | (3, PageSize::Size1G)
            )
        }
    }

    // ── Mock Alloc ──────────────────────────────────────────────────────────
    //
    // Backed by a `HashMap<usize, *mut u8>` keyed by PhysAddr.
    // The "phys" address is just the heap pointer, and `phys_to_virt`
    // is the identity. This keeps tests deterministic and lets us
    // count live frames to verify Drop reclaim.

    /// Live frames are tracked as raw pointers from 4 KiB-aligned
    /// allocations. Alignment matters because Entry encoders mask off the
    /// low 12 address bits.
    struct FrameBlock {
        ptr: *mut u8,
        layout: Layout,
    }

    type FrameStore = HashMap<usize, FrameBlock>;
    // SAFETY: tests don't hand frame pointers across threads; the
    // Mutex<Option<FrameStore>> serializes all access.
    struct StoreCell(Mutex<Option<FrameStore>>);
    unsafe impl Sync for StoreCell {}
    static FRAMES: StoreCell = StoreCell(Mutex::new(None));
    static LIVE_COUNT: AtomicUsize = AtomicUsize::new(0);
    static RANGE_FLUSH_CALLS: AtomicUsize = AtomicUsize::new(0);
    static LOCAL_FLUSH_CALLS: AtomicUsize = AtomicUsize::new(0);
    static FULL_FLUSH_CALLS: AtomicUsize = AtomicUsize::new(0);
    static LAST_RANGE_START: AtomicUsize = AtomicUsize::new(0);
    static LAST_RANGE_PAGES: AtomicUsize = AtomicUsize::new(0);

    fn frames_lock() -> std::sync::MutexGuard<'static, Option<FrameStore>> {
        let mut g = FRAMES.0.lock().unwrap();
        if g.is_none() {
            *g = Some(HashMap::new());
        }
        g
    }

    fn live_count() -> usize {
        LIVE_COUNT.load(Ordering::SeqCst)
    }

    /// Reset between tests (`cargo test` serializes via the FRAMES
    /// mutex on every alloc/dealloc).
    fn reset_alloc() {
        let mut g = frames_lock();
        if let Some(store) = g.as_mut() {
            // Drop any leftover frames from a previous test.
            for (_, block) in store.drain() {
                unsafe { dealloc(block.ptr, block.layout) };
            }
        }
        LIVE_COUNT.store(0, Ordering::SeqCst);
    }

    struct MockAlloc;

    impl FrameAllocator for MockAlloc {
        fn allocate(size: usize, align: PageSize) -> PagingResult<PhysAddrRange> {
            if size == 0 || (size & 0xfff) != 0 {
                return Err(PagingError::NotAligned);
            }
            let layout = Layout::from_size_align(size, align.bytes())
                .map_err(|_| PagingError::NotAligned)?;
            let raw = unsafe { alloc_zeroed(layout) };
            if raw.is_null() {
                return Err(PagingError::OutOfMemory);
            }
            let key = raw as usize;
            frames_lock()
                .as_mut()
                .unwrap()
                .insert(key, FrameBlock { ptr: raw, layout });
            LIVE_COUNT.fetch_add(1, Ordering::SeqCst);
            Ok(PhysAddrRange::from_start_size(
                PhysAddr::from_usize(key),
                size,
            ))
        }

        fn deallocate(range: PhysAddrRange) -> PagingResult {
            let Some(block) = frames_lock()
                .as_mut()
                .unwrap()
                .remove(&range.start.as_usize())
            else {
                return Err(PagingError::AddressOutOfRange);
            };
            if block.layout.size() != range.size() {
                frames_lock()
                    .as_mut()
                    .unwrap()
                    .insert(range.start.as_usize(), block);
                return Err(PagingError::NotAligned);
            }
            unsafe { dealloc(block.ptr, block.layout) };
            LIVE_COUNT.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        }

        fn phys_to_virt(paddr: PhysAddr) -> memory_addr::VirtAddr {
            memory_addr::VirtAddr::from_usize(paddr.as_usize())
        }
    }

    // Walker tests use NoFlush — they exercise the tree mechanics, not
    // the TLB layer. CPU TLB invalidation is covered separately in the
    // `arch::x86_64::X86Tlb` impl tests.
    type Pt = PageTableWalker<MockMeta, MockPte, MockAlloc>;

    struct WideTableMeta;

    impl PagingMetaData for WideTableMeta {
        const LEVELS: usize = 2;
        const PA_MAX_BITS: usize = 52;
        const VA_MAX_BITS: usize = 48;
        const INDEX_BITS: u32 = 11;
        const BASE_PAGE_SIZE: PageSize = PageSize::Size16K;
        const TABLE_FRAME_SIZE: PageSize = PageSize::Size16K;

        type VirtAddr = memory_addr::VirtAddr;

        fn level_shift(level: u8) -> u32 {
            14 + ((level as u32) - 1) * 11
        }

        fn level_supports_leaf(level: u8, size: PageSize) -> bool {
            matches!((level, size), (1, PageSize::Size16K))
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct RecordingTlb;

    impl TlbInvalidation<memory_addr::VirtAddr> for RecordingTlb {
        fn flush_tlb_local(&self, _vaddr: memory_addr::VirtAddr) {
            LOCAL_FLUSH_CALLS.fetch_add(1, Ordering::SeqCst);
        }

        fn flush_tlb_all_local(&self) {
            FULL_FLUSH_CALLS.fetch_add(1, Ordering::SeqCst);
        }

        fn flush_tlb_range_local(
            &self,
            start: memory_addr::VirtAddr,
            _page_size: PageSize,
            count_pages: usize,
        ) {
            RANGE_FLUSH_CALLS.fetch_add(1, Ordering::SeqCst);
            LAST_RANGE_START.store(start.as_usize(), Ordering::SeqCst);
            LAST_RANGE_PAGES.store(count_pages, Ordering::SeqCst);
        }

        fn prefer_full_flush(&self, _pending_count: usize) -> bool {
            false
        }
    }

    type RecordingPt = PageTableWalker<MockMeta, MockPte, MockAlloc>;

    const NO_FLUSH: NoFlush = NoFlush;
    const RECORDING_TLB: RecordingTlb = RecordingTlb;

    fn flags(access: AccessFlags) -> MockFlags {
        MockFlags {
            access,
            cache: CachePolicy::Writeback,
        }
    }

    fn vrange(start: usize, size: PageSize) -> memory_addr::VirtAddrRange {
        memory_addr::VirtAddrRange::from_start_size(
            memory_addr::VirtAddr::from_usize(start),
            size.bytes(),
        )
    }

    /// All tests share one MockAlloc backing store. Take the lock at the
    /// top of each test so they serialize even under `cargo test`'s
    /// default parallel runner.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn test_setup() -> std::sync::MutexGuard<'static, ()> {
        // Recover from a panicking sibling: PoisonError still hands us
        // the guard so we can clean up the backing store.
        let guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_alloc();
        RANGE_FLUSH_CALLS.store(0, Ordering::SeqCst);
        LOCAL_FLUSH_CALLS.store(0, Ordering::SeqCst);
        FULL_FLUSH_CALLS.store(0, Ordering::SeqCst);
        LAST_RANGE_START.store(0, Ordering::SeqCst);
        LAST_RANGE_PAGES.store(0, Ordering::SeqCst);
        guard
    }

    // ── Tests ───────────────────────────────────────────────────────────────

    #[test]
    fn map_query_unmap_4k_round_trip() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x1000_0000);
        let p = PhysAddr::from_usize(0xdead_0000);
        let f = flags(AccessFlags::READ | AccessFlags::WRITE);

        let range = vrange(v.as_usize(), PageSize::Size4K);

        pt.map(range, p, f, &NO_FLUSH).unwrap();
        let mapping = pt.query(v).unwrap();
        assert_eq!(mapping.paddr.as_usize(), p.as_usize());
        assert_eq!(mapping.range, range);
        assert_eq!(mapping.size(), Some(PageSize::Size4K));
        assert!(mapping.flags.access.contains(AccessFlags::WRITE));

        let unmapped = pt.unmap(range, &NO_FLUSH).unwrap();
        assert_eq!(unmapped.paddr.as_usize(), p.as_usize());
        assert_eq!(unmapped.range, range);
        assert_eq!(unmapped.size(), Some(PageSize::Size4K));
        assert!(matches!(pt.query(v), Err(PagingError::NotMapped)));
    }

    #[test]
    fn present_invalid_entry_fails_closed() {
        let _g = test_setup();
        let pt = Pt::try_new().unwrap();
        unsafe {
            write_entry::<MockPte, MockAlloc>(pt.root, 0, MockPte(MOCK_PRESENT | MOCK_INVALID));
        }

        assert_eq!(
            pt.query(memory_addr::VirtAddr::from_usize(0)),
            Err(PagingError::InvalidEntryKind)
        );
    }

    #[test]
    fn metadata_controls_table_frame_granule() {
        let _g = test_setup();
        {
            let pt = PageTableWalker::<WideTableMeta, MockPte, MockAlloc>::try_new().unwrap();
            assert!(PageSize::Size16K.is_aligned(pt.root().as_usize()));
            assert_eq!(LIVE_COUNT.load(Ordering::SeqCst), 1);
        }
        assert_eq!(LIVE_COUNT.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn map_2m_huge_then_query() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x4000_0000);
        let p = PhysAddr::from_usize(0x2_0000_0000);
        let f = flags(AccessFlags::READ);

        let range = vrange(v.as_usize(), PageSize::Size2M);
        pt.map(range, p, f, &NO_FLUSH).unwrap();
        let mapping = pt.query(v).unwrap();
        assert_eq!(mapping.range, range);
        assert_eq!(mapping.size(), Some(PageSize::Size2M));
    }

    #[test]
    fn exact_range_ops_refuse_interior_subranges() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x4000_0000);
        let p = PhysAddr::from_usize(0x2_0000_0000);
        let range = vrange(v.as_usize(), PageSize::Size2M);
        let interior = vrange(v.as_usize() + 0x1000, PageSize::Size4K);

        pt.map(range, p, flags(AccessFlags::READ), &NO_FLUSH)
            .unwrap();

        assert!(matches!(
            pt.protect(
                interior,
                flags(AccessFlags::READ | AccessFlags::WRITE),
                &NO_FLUSH
            ),
            Err(PagingError::NotMapped)
        ));
        assert!(matches!(
            pt.remap(
                interior,
                PhysAddr::from_usize(0x3_0000_0000),
                flags(AccessFlags::READ),
                &NO_FLUSH
            ),
            Err(PagingError::NotMapped)
        ));
        assert!(matches!(
            pt.unmap(interior, &NO_FLUSH),
            Err(PagingError::NotMapped)
        ));

        let mapping = pt.query(v).unwrap();
        assert_eq!(mapping.range, range);
        assert_eq!(mapping.paddr.as_usize(), p.as_usize());
    }

    #[test]
    fn double_map_returns_already_mapped() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x2000);
        let p = PhysAddr::from_usize(0x4000);
        let f = flags(AccessFlags::READ);

        let range = vrange(v.as_usize(), PageSize::Size4K);
        pt.map(range, p, f, &NO_FLUSH).unwrap();
        assert!(matches!(
            pt.map(range, p, f, &NO_FLUSH),
            Err(PagingError::AlreadyMapped)
        ));
    }

    #[test]
    fn multiple_virtual_ranges_may_share_one_physical_frame() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let p = PhysAddr::from_usize(0x9000_0000);
        let r1 = vrange(0x1000_0000, PageSize::Size4K);
        let r2 = vrange(0x2000_0000, PageSize::Size4K);
        let f = flags(AccessFlags::READ | AccessFlags::WRITE);

        pt.map(r1, p, f, &NO_FLUSH).unwrap();
        pt.map(r2, p, f, &NO_FLUSH).unwrap();

        assert_eq!(
            pt.query(memory_addr::VirtAddr::from_usize(0x1000_0000))
                .unwrap()
                .paddr,
            p
        );
        assert_eq!(
            pt.query(memory_addr::VirtAddr::from_usize(0x2000_0000))
                .unwrap()
                .paddr,
            p
        );

        pt.unmap(r1, &NO_FLUSH).unwrap();
        assert!(matches!(
            pt.query(memory_addr::VirtAddr::from_usize(0x1000_0000)),
            Err(PagingError::NotMapped)
        ));
        assert_eq!(
            pt.query(memory_addr::VirtAddr::from_usize(0x2000_0000))
                .unwrap()
                .paddr,
            p
        );
    }

    #[test]
    fn unmap_reclaims_emptied_intermediates() {
        let _g = test_setup();
        let baseline = live_count(); // 0 before try_new
        let mut pt = Pt::try_new().unwrap();
        // 1 root frame
        assert_eq!(live_count(), baseline + 1);

        let v = memory_addr::VirtAddr::from_usize(0x1234_5000);
        let p = PhysAddr::from_usize(0x6000);
        let f = flags(AccessFlags::READ);

        let range = vrange(v.as_usize(), PageSize::Size4K);
        pt.map(range, p, f, &NO_FLUSH).unwrap();
        // Walker allocated 3 intermediate frames (PML4 was the root, +
        // PDPT, PD, PT) — total live should be 4.
        assert_eq!(live_count(), baseline + 4);

        pt.unmap(range, &NO_FLUSH).unwrap();
        // unmap reclaims the 3 emptied intermediates. Root remains.
        assert_eq!(live_count(), baseline + 1);
    }

    #[test]
    fn drop_reclaims_entire_subtree() {
        let _g = test_setup();
        let baseline = live_count();
        {
            let mut pt = Pt::try_new().unwrap();
            // Map several pages spread across multiple PD/PT subtrees.
            for i in 0..5 {
                let v = memory_addr::VirtAddr::from_usize(0x1000_0000 + i * 0x4000_0000);
                let p = PhysAddr::from_usize(0x10_0000 + i * 0x1000);
                pt.map(
                    vrange(v.as_usize(), PageSize::Size4K),
                    p,
                    flags(AccessFlags::READ),
                    &NO_FLUSH,
                )
                .unwrap();
            }
            assert!(live_count() > baseline + 1);
        }
        // After Drop, all walker-owned frames must be reclaimed.
        assert_eq!(live_count(), baseline);
    }

    #[test]
    fn split_2m_into_4k_leaves() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x4000_0000);
        let p = PhysAddr::from_usize(0x8000_0000);

        let range = vrange(v.as_usize(), PageSize::Size2M);
        pt.map(range, p, flags(AccessFlags::READ), &NO_FLUSH)
            .unwrap();
        let new_size = pt.split_at(range, &NO_FLUSH).unwrap();
        assert_eq!(new_size, PageSize::Size4K);

        // Each of the 512 4K children must resolve to the right paddr.
        for i in 0..MockMeta::ENTRIES_PER_TABLE {
            let qv = memory_addr::VirtAddr::from_usize(v.as_usize() + i * 0x1000);
            let mapping = pt.query(qv).unwrap();
            assert_eq!(mapping.size(), Some(PageSize::Size4K));
            assert_eq!(mapping.paddr.as_usize(), p.as_usize() + i * 0x1000);
        }
    }

    #[test]
    fn merge_after_split_round_trips() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x4000_0000);
        let p = PhysAddr::from_usize(0x8000_0000);

        let range = vrange(v.as_usize(), PageSize::Size2M);
        pt.map(range, p, flags(AccessFlags::READ), &NO_FLUSH)
            .unwrap();
        pt.split_at(range, &NO_FLUSH).unwrap();

        let merged = pt.merge_at(range, &NO_FLUSH).unwrap();
        assert_eq!(merged, PageSize::Size2M);

        let mapping = pt.query(v).unwrap();
        assert_eq!(mapping.size(), Some(PageSize::Size2M));
    }

    #[test]
    fn merge_refuses_when_not_contiguous() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x4000_0000);

        // Map 512 4K pages but with a hole in the middle of the
        // physical range — non-contiguous → merge must refuse.
        for i in 0..MockMeta::ENTRIES_PER_TABLE {
            let qv = memory_addr::VirtAddr::from_usize(v.as_usize() + i * 0x1000);
            // Physical addresses are non-contiguous: i=0..256 from 0x8000_0000,
            // i=256..512 from 0xc000_0000.
            let phys_base = if i < 256 { 0x8000_0000 } else { 0xc000_0000 };
            let qp = PhysAddr::from_usize(phys_base + (i % 256) * 0x1000);
            pt.map(
                vrange(qv.as_usize(), PageSize::Size4K),
                qp,
                flags(AccessFlags::READ),
                &NO_FLUSH,
            )
            .unwrap();
        }

        assert!(matches!(
            pt.merge_at(vrange(v.as_usize(), PageSize::Size2M), &NO_FLUSH),
            Err(PagingError::NotCoalescable)
        ));
    }

    #[test]
    fn merge_refuses_on_mismatched_flags() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x4000_0000);
        let p_base = 0x8000_0000;

        for i in 0..MockMeta::ENTRIES_PER_TABLE {
            let qv = memory_addr::VirtAddr::from_usize(v.as_usize() + i * 0x1000);
            let qp = PhysAddr::from_usize(p_base + i * 0x1000);
            // Flip one entry's flags to break the merge contract.
            let f = if i == 100 {
                flags(AccessFlags::READ | AccessFlags::WRITE)
            } else {
                flags(AccessFlags::READ)
            };
            pt.map(vrange(qv.as_usize(), PageSize::Size4K), qp, f, &NO_FLUSH)
                .unwrap();
        }

        assert!(matches!(
            pt.merge_at(vrange(v.as_usize(), PageSize::Size2M), &NO_FLUSH),
            Err(PagingError::NotCoalescable)
        ));
    }

    #[test]
    fn protect_keeps_paddr_changes_flags() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x1000);
        let p = PhysAddr::from_usize(0x2000);
        let range = vrange(v.as_usize(), PageSize::Size4K);
        pt.map(range, p, flags(AccessFlags::READ), &NO_FLUSH)
            .unwrap();

        let _ = pt
            .protect(
                range,
                flags(AccessFlags::READ | AccessFlags::WRITE),
                &NO_FLUSH,
            )
            .unwrap();
        let mapping = pt.query(v).unwrap();
        assert_eq!(mapping.paddr.as_usize(), p.as_usize());
        assert!(mapping.flags.access.contains(AccessFlags::WRITE));
    }

    #[test]
    fn remap_changes_paddr() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x1000);
        let p1 = PhysAddr::from_usize(0x2000);
        let p2 = PhysAddr::from_usize(0x5000);
        let range = vrange(v.as_usize(), PageSize::Size4K);
        pt.map(range, p1, flags(AccessFlags::READ), &NO_FLUSH)
            .unwrap();

        pt.remap(range, p2, flags(AccessFlags::READ), &NO_FLUSH)
            .unwrap();
        let mapping = pt.query(v).unwrap();
        assert_eq!(mapping.paddr.as_usize(), p2.as_usize());
    }

    // ── Cursor tests ────────────────────────────────────────────────────────

    #[test]
    fn cursor_maps_contiguous_range_correctly() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let base_v = 0x1000_0000usize;
        let base_p = 0x4000_0000usize;
        let count = 64;

        {
            let mut c = pt.cursor(&NO_FLUSH);
            for i in 0..count {
                let v = memory_addr::VirtAddr::from_usize(base_v + i * 0x1000);
                let p = PhysAddr::from_usize(base_p + i * 0x1000);
                c.map(
                    vrange(v.as_usize(), PageSize::Size4K),
                    p,
                    flags(AccessFlags::READ),
                )
                .unwrap();
            }
            c.finish().unwrap();
        }

        // Each entry must be queryable post-cursor.
        for i in 0..count {
            let v = memory_addr::VirtAddr::from_usize(base_v + i * 0x1000);
            let mapping = pt.query(v).unwrap();
            assert_eq!(mapping.size(), Some(PageSize::Size4K));
            assert_eq!(mapping.paddr.as_usize(), base_p + i * 0x1000);
        }
    }

    #[test]
    fn cursor_maps_across_level2_boundary_correctly() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let addrs = [0x1ff000usize, 0x200000usize, 0x201000usize];

        {
            let mut c = pt.cursor(&NO_FLUSH);
            for (i, v) in addrs.into_iter().enumerate() {
                c.map(
                    vrange(v, PageSize::Size4K),
                    PhysAddr::from_usize(0x4000_0000 + i * 0x1000),
                    flags(AccessFlags::READ),
                )
                .unwrap();
            }
        }

        for (i, v) in addrs.into_iter().enumerate() {
            let mapping = pt.query(memory_addr::VirtAddr::from_usize(v)).unwrap();
            assert_eq!(mapping.size(), Some(PageSize::Size4K));
            assert_eq!(mapping.paddr.as_usize(), 0x4000_0000 + i * 0x1000);
        }
    }

    #[test]
    fn cursor_drop_flushes_pending() {
        // Cursor's auto-flush on Drop is the contract callers rely on.
        // We can't observe TLB state directly, but we verify a cursor
        // that goes out of scope without explicit `finish` still leaves
        // the table queryable (i.e., didn't deadlock on pending flush).
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        {
            let mut c = pt.cursor(&NO_FLUSH);
            c.map(
                vrange(0x1000, PageSize::Size4K),
                PhysAddr::from_usize(0x2000),
                flags(AccessFlags::READ),
            )
            .unwrap();
            // Implicit Drop here.
        }
        let mapping = pt.query(memory_addr::VirtAddr::from_usize(0x1000)).unwrap();
        assert_eq!(mapping.paddr.as_usize(), 0x2000);
    }

    #[test]
    fn cursor_flushes_contiguous_pending_pages_as_range() {
        let _g = test_setup();
        let mut pt = RecordingPt::try_new().unwrap();
        let base_v = 0x5000_0000usize;
        let count = FLUSH_RANGE_CAP + 5;

        {
            let mut c = pt.cursor(&RECORDING_TLB);
            for i in 0..count {
                c.map(
                    vrange(base_v + i * 0x1000, PageSize::Size4K),
                    PhysAddr::from_usize(0x9000_0000 + i * 0x1000),
                    flags(AccessFlags::READ),
                )
                .unwrap();
            }
            c.finish().unwrap();
        }

        assert_eq!(RANGE_FLUSH_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(LOCAL_FLUSH_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(FULL_FLUSH_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(LAST_RANGE_START.load(Ordering::SeqCst), base_v);
        assert_eq!(LAST_RANGE_PAGES.load(Ordering::SeqCst), count);
    }

    #[test]
    fn cursor_fragmented_range_overflow_falls_back_to_full_flush() {
        // The storage cap is on disjoint ranges, not page count. A
        // highly fragmented batch that exceeds the range queue still
        // falls back to full flush while leaving all mappings installed.
        let _g = test_setup();
        let mut pt = RecordingPt::try_new().unwrap();
        let base_v = 0x4000_0000usize;
        let base_p = 0x8000_0000usize;
        let count = FLUSH_RANGE_CAP + 1;

        {
            let mut c = pt.cursor(&RECORDING_TLB);
            for i in 0..count {
                c.map(
                    vrange(base_v + i * 0x2000, PageSize::Size4K),
                    PhysAddr::from_usize(base_p + i * 0x1000),
                    flags(AccessFlags::READ),
                )
                .unwrap();
            }
            assert!(c.needs_full_flush);
        }

        assert_eq!(RANGE_FLUSH_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(LOCAL_FLUSH_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(FULL_FLUSH_CALLS.load(Ordering::SeqCst), 1);

        // All `count` entries should be installed and queryable.
        for i in 0..count {
            let v = memory_addr::VirtAddr::from_usize(base_v + i * 0x2000);
            let mapping = pt.query(v).unwrap();
            assert_eq!(mapping.paddr.as_usize(), base_p + i * 0x1000);
        }
    }

    #[test]
    fn cursor_unmap_then_remap_roundtrip() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v = memory_addr::VirtAddr::from_usize(0x1000_0000);
        let p1 = PhysAddr::from_usize(0x4000_0000);
        let p2 = PhysAddr::from_usize(0x8000_0000);
        let range = vrange(v.as_usize(), PageSize::Size4K);
        pt.map(range, p1, flags(AccessFlags::READ), &NO_FLUSH)
            .unwrap();

        {
            let mut c = pt.cursor(&NO_FLUSH);
            c.unmap(range).unwrap();
            c.map(range, p2, flags(AccessFlags::READ | AccessFlags::WRITE))
                .unwrap();
        }

        let mapping = pt.query(v).unwrap();
        assert_eq!(mapping.paddr.as_usize(), p2.as_usize());
        assert!(mapping.flags.access.contains(AccessFlags::WRITE));
    }

    #[test]
    fn map_scattered_backing_accepts_array_reference() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let ranges: [PhysAddrRange; 16] = core::array::from_fn(|i| {
            PhysAddrRange::from_start_size(PhysAddr::from_usize(0x6000_0000 + i * 0x2000), 0x1000)
        });
        pt.map(
            vrange(0x2000_0000, PageSize::Size64K),
            &ranges,
            MappingFlags::scattered(flags(AccessFlags::READ), PageSize::Size4K),
            &NO_FLUSH,
        )
        .unwrap();

        // Verify all 16 entries.
        for i in 0..16 {
            let v = memory_addr::VirtAddr::from_usize(0x2000_0000 + i * 0x1000);
            let mapping = pt.query(v).unwrap();
            assert_eq!(mapping.paddr.as_usize(), 0x6000_0000 + i * 0x2000);
        }
    }

    #[test]
    fn map_scattered_backing_accepts_heapless_vec() {
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let mut ranges = heapless::Vec::<PhysAddrRange, 4>::new();
        for i in 0..4 {
            ranges
                .push(PhysAddrRange::from_start_size(
                    PhysAddr::from_usize(0x7000_0000 + i * 0x3000),
                    0x1000,
                ))
                .unwrap();
        }

        pt.map(
            vrange(0x3000_0000, PageSize::Size16K),
            &ranges,
            MappingFlags::new(flags(AccessFlags::READ))
                .with_contiguity(MappingContiguity::Scattered(PageSize::Size4K)),
            &NO_FLUSH,
        )
        .unwrap();

        for i in 0..4 {
            let v = memory_addr::VirtAddr::from_usize(0x3000_0000 + i * 0x1000);
            let mapping = pt.query(v).unwrap();
            assert_eq!(mapping.paddr.as_usize(), 0x7000_0000 + i * 0x3000);
        }
    }

    #[test]
    fn cursor_error_invalidates_descent() {
        // Map once, then a cursor that double-maps the same vaddr
        // should fail and invalidate the cursor cache. A subsequent
        // call must walk fresh and succeed at a different address.
        let _g = test_setup();
        let mut pt = Pt::try_new().unwrap();
        let v1 = memory_addr::VirtAddr::from_usize(0x1000);
        let v2 = memory_addr::VirtAddr::from_usize(0x4000_0000);
        let r1 = vrange(v1.as_usize(), PageSize::Size4K);
        let r2 = vrange(v2.as_usize(), PageSize::Size4K);
        let f = flags(AccessFlags::READ);

        pt.map(r1, PhysAddr::from_usize(0x2000), f, &NO_FLUSH)
            .unwrap();

        {
            let mut c = pt.cursor(&NO_FLUSH);
            // Attempting to remap v1 via cursor.map (a fresh map) fails.
            assert!(matches!(
                c.map(r1, PhysAddr::from_usize(0x2000), f),
                Err(PagingError::AlreadyMapped)
            ));
            // Cursor cache invalidated; mapping at v2 still works.
            c.map(r2, PhysAddr::from_usize(0x5000_0000), f).unwrap();
        }

        let mapping = pt.query(v2).unwrap();
        assert_eq!(mapping.paddr.as_usize(), 0x5000_0000);
    }
}
