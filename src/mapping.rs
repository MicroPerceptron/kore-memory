use core::ptr;

use memory_addr::{AddrRange, MemoryAddr, PhysAddr, PhysAddrRange, VirtAddr};

use crate::{PageSize, PageTableEntry, PagingError, PagingResult};

/// A resolved leaf mapping.
///
/// `range` is the exact virtual/IOVA range covered by the leaf; its
/// length determines the leaf size. `paddr` is the aligned physical base
/// the leaf points at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapping<Entry, V, P = PhysAddr>
where
    Entry: PageTableEntry,
    V: MemoryAddr,
    P: MemoryAddr,
{
    pub range: AddrRange<V>,
    pub paddr: P,
    pub flags: Entry::Flags,
}

impl<Entry, V, P> Mapping<Entry, V, P>
where
    Entry: PageTableEntry,
    V: MemoryAddr,
    P: MemoryAddr,
{
    #[inline]
    pub const fn new(range: AddrRange<V>, paddr: P, flags: Entry::Flags) -> Self {
        Self {
            range,
            paddr,
            flags,
        }
    }

    #[inline]
    pub fn size(&self) -> Option<PageSize> {
        PageSize::from_bytes(self.range.size())
    }
}

impl<Entry, P> Mapping<Entry, VirtAddr, P>
where
    Entry: PageTableEntry,
    P: MemoryAddr, // MMIO Address type
{
    #[inline]
    pub fn start(&self) -> VirtAddr {
        self.range.start
    }

    #[inline]
    pub fn end(&self) -> VirtAddr {
        self.range.end
    }

    #[inline]
    pub fn contains(&self, vaddr: VirtAddr) -> bool {
        self.range.contains(vaddr)
    }

    #[inline(always)]
    fn offset_with<T: Copy>(&self, offset: usize) -> Result<VirtAddr, PagingError> {
        let end = offset.checked_add(size_of::<T>()).unwrap_or(usize::MAX);
        if end <= self.range.size() {
            if let Some(addr) = self.start().checked_add(offset) {
                Ok(addr)
            } else {
                Err(PagingError::AddressOutOfRange)
            }
        } else {
            Err(PagingError::UnsupportedSize)
        }
    }

    #[inline(always)]
    pub fn as_ptr_of<T: Copy>(&self, offset: usize) -> Result<*const T, PagingError> {
        self.offset_with::<T>(offset)
            .map(|addr| addr.as_ptr_of::<T>())
    }

    #[inline(always)]
    pub fn as_mut_ptr_of<T: Copy>(&mut self, offset: usize) -> Result<*mut T, PagingError> {
        self.offset_with::<T>(offset)
            .map(|addr| addr.as_mut_ptr_of::<T>())
    }

    /// # Safety
    ///
    /// Caller must ensure the range is currently mapped and readable.
    /// Uses `ptr::read_unaligned` — for aligned reads, prefer `as_ptr_of`.
    #[inline(always)]
    pub unsafe fn read_unaligned<T: Copy>(&self, offset: usize) -> Result<T, PagingError> {
        let src = self.as_ptr_of::<T>(offset)? as *const T;
        Ok(unsafe { src.read_unaligned() })
    }

    /// # Safety
    ///
    /// Caller must ensure the range is currently mapped and writable.
    /// Uses `ptr::write_unaligned` — for aligned writes, prefer `as_mut_ptr_of`.
    #[inline(always)]
    pub unsafe fn write_unaligned<T: Copy>(
        &mut self,
        offset: usize,
        value: T,
    ) -> Result<(), PagingError> {
        let dst = self.as_mut_ptr_of::<T>(offset)? as *mut T;
        unsafe { dst.write_unaligned(value) };
        Ok(())
    }

    /// # Safety
    ///
    /// Caller must ensure the range is currently mapped and readable.
    #[inline(always)]
    pub unsafe fn copy_into_slice<T: Copy>(
        &self,
        offset: usize,
        dst: &mut [T],
    ) -> Result<(), PagingError> {
        let src = self.as_ptr_of::<T>(offset)? as *const T;
        unsafe { ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    /// # Safety
    ///
    /// Caller must ensure the range is currently mapped and writable.
    #[inline(always)]
    pub unsafe fn copy_from_slice<T: Copy>(
        &mut self,
        offset: usize,
        src: &[T],
    ) -> Result<(), PagingError> {
        let dst = self.as_mut_ptr_of::<T>(offset)? as *mut T;
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────
    // mmio read/write helpers — convenience methods for common use case
    // ─────────────────────────────────────────────────────────────────

    #[inline(always)]
    fn read_ux_volatile<T: Copy>(&self, offset: usize) -> Result<T, PagingError> {
        self.as_ptr_of::<T>(offset)
            .map(|r| unsafe { ptr::read_volatile(r) })
    }

    #[inline(always)]
    fn write_ux_volatile<T: Copy>(&mut self, offset: usize, value: T) -> Result<(), PagingError> {
        self.as_mut_ptr_of::<T>(offset)
            .map(|r| unsafe { ptr::write_volatile(r, value) })
    }

    #[inline(always)]
    fn modify_ux_volatile<T: Copy>(
        &mut self,
        offset: usize,
        f: impl FnOnce(T) -> T,
    ) -> Result<T, PagingError> {
        let r = self.as_mut_ptr_of::<T>(offset)?;
        let old = unsafe { core::ptr::read_volatile(r) };
        unsafe { core::ptr::write_volatile(r, f(old)) };
        Ok(old)
    }

    #[inline(always)]
    pub fn read_vo8(&self, offset: usize) -> Result<u8, PagingError> {
        self.read_ux_volatile(offset)
    }

    #[inline(always)]
    pub fn write_vo8(&mut self, offset: usize, value: u8) -> Result<(), PagingError> {
        self.write_ux_volatile(offset, value)
    }

    #[inline(always)]
    pub fn modify_vo8(
        &mut self,
        offset: usize,
        f: impl FnOnce(u8) -> u8,
    ) -> Result<u8, PagingError> {
        self.modify_ux_volatile(offset, f)
    }

    #[inline(always)]
    pub fn read_vo16(&self, offset: usize) -> Result<u16, PagingError> {
        self.read_ux_volatile(offset)
    }

    #[inline(always)]
    pub fn write_vo16(&mut self, offset: usize, value: u16) -> Result<(), PagingError> {
        self.write_ux_volatile(offset, value)
    }

    #[inline(always)]
    pub fn modify_vo16(
        &mut self,
        offset: usize,
        f: impl FnOnce(u16) -> u16,
    ) -> Result<u16, PagingError> {
        self.modify_ux_volatile(offset, f)
    }

    #[inline(always)]
    pub fn read_vo32(&self, offset: usize) -> Result<u32, PagingError> {
        self.read_ux_volatile(offset)
    }

    #[inline(always)]
    pub fn write_vo32(&mut self, offset: usize, value: u32) -> Result<(), PagingError> {
        self.write_ux_volatile(offset, value)
    }

    #[inline(always)]
    pub fn modify_vo32(
        &mut self,
        offset: usize,
        f: impl FnOnce(u32) -> u32,
    ) -> Result<u32, PagingError> {
        self.modify_ux_volatile(offset, f)
    }

    #[inline(always)]
    pub fn read_vo64(&self, offset: usize) -> Result<u64, PagingError> {
        self.read_ux_volatile(offset)
    }

    #[inline(always)]
    pub fn write_vo64(&mut self, offset: usize, value: u64) -> Result<(), PagingError> {
        self.write_ux_volatile(offset, value)
    }

    #[inline(always)]
    pub fn modify_vo64(
        &mut self,
        offset: usize,
        f: impl FnOnce(u64) -> u64,
    ) -> Result<u64, PagingError> {
        self.modify_ux_volatile(offset, f)
    }

    #[inline(always)]
    pub fn read_vo128(&self, offset: usize) -> Result<u128, PagingError> {
        self.read_ux_volatile(offset)
    }

    #[inline(always)]
    pub fn write_vo128(&mut self, offset: usize, value: u128) -> Result<(), PagingError> {
        self.write_ux_volatile(offset, value)
    }

    #[inline(always)]
    pub fn modify_vo128(
        &mut self,
        offset: usize,
        f: impl FnOnce(u128) -> u128,
    ) -> Result<u128, PagingError> {
        self.modify_ux_volatile(offset, f)
    }
}

/// Physical backing shape supplied to [`PageTable::map`](crate::PageTable::map).
///
/// `Contiguous` carries one physical extent. `Scattered` carries ordered
/// physical extents that back one virtually-contiguous range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapBacking<'a> {
    Contiguous(PhysAddrRange),
    Scattered(&'a [PhysAddrRange]),
}

impl<'a> MapBacking<'a> {
    #[inline]
    pub const fn contiguous(range: PhysAddrRange) -> Self {
        Self::Contiguous(range)
    }

    #[inline]
    pub fn contiguous_from_start_size(start: PhysAddr, size: usize) -> Self {
        Self::Contiguous(PhysAddrRange::from_start_size(start, size))
    }

    #[inline]
    pub const fn scattered(ranges: &'a [PhysAddrRange]) -> Self {
        Self::Scattered(ranges)
    }
}

impl<'a> From<PhysAddrRange> for MapBacking<'a> {
    #[inline]
    fn from(range: PhysAddrRange) -> Self {
        Self::Contiguous(range)
    }
}

/// Converts ergonomic backing expressions into a [`MapBacking`].
///
/// A bare [`PhysAddr`] is interpreted as one contiguous range with the
/// same byte length as the virtual range being mapped.
pub trait IntoMapBacking<'a> {
    fn into_map_backing(self, virtual_size: usize) -> PagingResult<MapBacking<'a>>;
}

impl<'a> IntoMapBacking<'a> for MapBacking<'a> {
    #[inline]
    fn into_map_backing(self, _virtual_size: usize) -> PagingResult<MapBacking<'a>> {
        Ok(self)
    }
}

impl<'a> IntoMapBacking<'a> for PhysAddr {
    #[inline]
    fn into_map_backing(self, virtual_size: usize) -> PagingResult<MapBacking<'a>> {
        Ok(MapBacking::contiguous_from_start_size(self, virtual_size))
    }
}

impl<'a> IntoMapBacking<'a> for PhysAddrRange {
    #[inline]
    fn into_map_backing(self, _virtual_size: usize) -> PagingResult<MapBacking<'a>> {
        Ok(MapBacking::Contiguous(self))
    }
}

impl<'a> IntoMapBacking<'a> for &'a [PhysAddrRange] {
    #[inline]
    fn into_map_backing(self, _virtual_size: usize) -> PagingResult<MapBacking<'a>> {
        Ok(MapBacking::Scattered(self))
    }
}

impl<'a, const N: usize> IntoMapBacking<'a> for &'a [PhysAddrRange; N] {
    #[inline]
    fn into_map_backing(self, _virtual_size: usize) -> PagingResult<MapBacking<'a>> {
        Ok(MapBacking::Scattered(&self[..]))
    }
}

impl<'a, const N: usize> IntoMapBacking<'a> for &'a heapless::Vec<PhysAddrRange, N> {
    #[inline]
    fn into_map_backing(self, _virtual_size: usize) -> PagingResult<MapBacking<'a>> {
        Ok(MapBacking::Scattered(self.as_slice()))
    }
}

/// Physical contiguity contract for a mapping request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MappingContiguity {
    /// Backing must be one contiguous physical range. The walker may use
    /// the largest legal leaf size at each aligned span.
    #[default]
    Contiguous,
    /// Backing may be multiple physical ranges. Every mapped leaf uses
    /// this granule exactly.
    Scattered(PageSize),
}

/// Mapping-level flags: hardware leaf flags plus backing-shape policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MappingFlags<F> {
    leaf: F,
    contiguity: MappingContiguity,
}

impl<F> MappingFlags<F> {
    #[inline]
    pub const fn new(leaf: F) -> Self {
        Self {
            leaf,
            contiguity: MappingContiguity::Contiguous,
        }
    }

    #[inline]
    pub const fn contiguous(leaf: F) -> Self {
        Self::new(leaf)
    }

    #[inline]
    pub const fn scattered(leaf: F, granule: PageSize) -> Self {
        Self {
            leaf,
            contiguity: MappingContiguity::Scattered(granule),
        }
    }

    #[inline]
    pub const fn with_contiguity(mut self, contiguity: MappingContiguity) -> Self {
        self.contiguity = contiguity;
        self
    }

    #[inline]
    pub const fn leaf(&self) -> F
    where
        F: Copy,
    {
        self.leaf
    }

    #[inline]
    pub const fn contiguity(&self) -> MappingContiguity {
        self.contiguity
    }
}

impl<F> From<F> for MappingFlags<F> {
    #[inline]
    fn from(leaf: F) -> Self {
        Self::new(leaf)
    }
}

#[cfg(test)]
mod tests {
    use memory_addr::VirtAddr;

    use super::*;
    use crate::PageTableEntryKind;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    struct DummyEntry(u64);

    impl PageTableEntry for DummyEntry {
        type Flags = ();

        fn new_leaf(paddr: PhysAddr, _flags: Self::Flags, _size: PageSize) -> Self {
            Self(paddr.as_usize() as u64)
        }

        fn new_table(paddr: PhysAddr, _level: u8) -> Self {
            Self(paddr.as_usize() as u64)
        }

        fn paddr(&self) -> PhysAddr {
            PhysAddr::from_usize(self.0 as usize)
        }

        fn flags(&self) -> Self::Flags {}

        fn is_present(&self) -> bool {
            self.0 != 0
        }

        fn entry_kind(&self, _level: u8) -> PageTableEntryKind {
            PageTableEntryKind::Leaf
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

    #[test]
    fn typed_access_may_end_at_range_boundary() {
        let bytes = [0u8; 8];
        let start = VirtAddr::from_usize(bytes.as_ptr() as usize);
        let end = VirtAddr::from_usize(start.as_usize() + bytes.len());
        let mapping = Mapping::<DummyEntry, VirtAddr>::new(
            AddrRange::new(start, end),
            PhysAddr::from_usize(0),
            (),
        );

        assert!(mapping.as_ptr_of::<u32>(4).is_ok());
        assert!(mapping.as_ptr_of::<u64>(1).is_err());
    }
}
