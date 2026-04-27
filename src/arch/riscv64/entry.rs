use core::fmt;

use memory_addr::{PhysAddr, VirtAddr};

use crate::{
    AccessFlags, CachePolicy, MemoryAttributes, PageSize, PageTableEntry, PageTableEntryKind,
    PagingMetaData,
};

const VALID: u64 = 1 << 0;
const READ: u64 = 1 << 1;
const WRITE: u64 = 1 << 2;
const EXECUTE: u64 = 1 << 3;
const ACCESSED: u64 = 1 << 6;

const PPN_SHIFT: u64 = 10;
const PPN_BITS: u64 = 44;
const PPN_MASK: u64 = ((1u64 << PPN_BITS) - 1) << PPN_SHIFT;
const PADDR_SHIFT: u64 = 12;
const PADDR_MAX_BITS: usize = 56;

/// Svpbmt default/PMA memory type.
pub const RV64_PBMT_PMA: u64 = 0;
/// Svpbmt non-cacheable, idempotent memory type.
pub const RV64_PBMT_NC: u64 = 1;
/// Svpbmt I/O, non-idempotent memory type.
pub const RV64_PBMT_IO: u64 = 2;
const RV64_PBMT_RESERVED: u64 = 3;
/// Raw PTE mask for Svpbmt bits.
pub const RV64_PBMT_MASK: u64 = 0b11 << 61;

const HIGH_RESERVED_MASK: u64 = 0x3ffu64 << 54;
const HIGH_RESERVED_WITH_SVPBMT_MASK: u64 = HIGH_RESERVED_MASK & !RV64_PBMT_MASK;

/// RISC-V PTE flag bundle: access bits + memory attributes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rv64Flags {
    pub access: AccessFlags,
    pub attrs: MemoryAttributes,
}

impl Rv64Flags {
    #[inline]
    pub const fn new(access: AccessFlags) -> Self {
        Self {
            access: access.union(AccessFlags::READ),
            attrs: MemoryAttributes::writeback(),
        }
    }

    #[inline]
    pub const fn with_cache(mut self, cache: CachePolicy) -> Self {
        self.attrs = self.attrs.with_cache(cache);
        self
    }

    #[inline]
    pub const fn with_attrs(mut self, attrs: MemoryAttributes) -> Self {
        self.attrs = attrs;
        self
    }

    #[inline]
    pub const fn cache(self) -> CachePolicy {
        self.attrs.cache()
    }

    #[inline]
    pub const fn attrs(self) -> MemoryAttributes {
        self.attrs
    }

    #[inline(always)]
    fn to_leaf_bits<const SVPBMT: bool>(self) -> u64 {
        let mut access = self.access.bits() as u64;
        access |= (access & AccessFlags::WRITE.bits() as u64) >> 1;

        VALID
            | ((access & 0x1f) << 1)
            | ACCESSED
            | ((((access & AccessFlags::WRITE.bits() as u64) != 0) as u64) << 7)
            | pbmt_bits::<SVPBMT>(self.attrs.cache())
    }

    #[inline(always)]
    fn from_bits<const SVPBMT: bool>(bits: u64) -> Self {
        Self {
            access: AccessFlags::from_bits_truncate(((bits >> 1) & 0x1f) as u8),
            attrs: MemoryAttributes::writeback().with_cache(cache_from_bits::<SVPBMT>(bits)),
        }
    }
}

#[inline(always)]
fn pbmt_bits<const SVPBMT: bool>(cache: CachePolicy) -> u64 {
    if !SVPBMT {
        return 0;
    }

    let pbmt = match cache {
        CachePolicy::Writeback | CachePolicy::WriteThrough => RV64_PBMT_PMA,
        CachePolicy::WriteCombine => RV64_PBMT_NC,
        CachePolicy::Uncached => RV64_PBMT_IO,
    };
    pbmt << 61
}

#[inline(always)]
fn cache_from_bits<const SVPBMT: bool>(bits: u64) -> CachePolicy {
    if !SVPBMT {
        return CachePolicy::Writeback;
    }

    match (bits & RV64_PBMT_MASK) >> 61 {
        RV64_PBMT_PMA => CachePolicy::Writeback,
        RV64_PBMT_NC => CachePolicy::WriteCombine,
        RV64_PBMT_IO => CachePolicy::Uncached,
        _ => CachePolicy::Writeback,
    }
}

#[inline(always)]
fn paddr_to_ppn_bits(paddr: PhysAddr) -> u64 {
    (((paddr.as_usize() as u64) >> PADDR_SHIFT) << PPN_SHIFT) & PPN_MASK
}

#[inline(always)]
fn ppn_bits_to_paddr(bits: u64) -> PhysAddr {
    PhysAddr::from_usize((((bits & PPN_MASK) >> PPN_SHIFT) << PADDR_SHIFT) as usize)
}

#[inline(always)]
fn high_reserved_bits<const SVPBMT: bool>(bits: u64) -> u64 {
    if SVPBMT {
        bits & HIGH_RESERVED_WITH_SVPBMT_MASK
    } else {
        bits & HIGH_RESERVED_MASK
    }
}

#[inline(always)]
fn valid_encoding<const SVPBMT: bool>(bits: u64) -> bool {
    if bits & VALID == 0 {
        return false;
    }
    if high_reserved_bits::<SVPBMT>(bits) != 0 {
        return false;
    }
    if bits & WRITE != 0 && bits & READ == 0 {
        return false;
    }
    if SVPBMT && ((bits & RV64_PBMT_MASK) >> 61) == RV64_PBMT_RESERVED {
        return false;
    }
    true
}

#[repr(transparent)]
struct Rv64PteFor<const SVPBMT: bool> {
    bits: u64,
}

impl<const SVPBMT: bool> Rv64PteFor<SVPBMT> {
    #[inline]
    const fn new(bits: u64) -> Self {
        Self { bits }
    }
}

impl<const SVPBMT: bool> Clone for Rv64PteFor<SVPBMT> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<const SVPBMT: bool> Copy for Rv64PteFor<SVPBMT> {}

impl<const SVPBMT: bool> Default for Rv64PteFor<SVPBMT> {
    #[inline]
    fn default() -> Self {
        Self::new(0)
    }
}

impl<const SVPBMT: bool> PartialEq for Rv64PteFor<SVPBMT> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl<const SVPBMT: bool> Eq for Rv64PteFor<SVPBMT> {}

impl<const SVPBMT: bool> fmt::Debug for Rv64PteFor<SVPBMT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Rv64Pte")
            .field(&format_args!("{:#018x}", self.bits))
            .finish()
    }
}

impl<const SVPBMT: bool> PageTableEntry for Rv64PteFor<SVPBMT> {
    type Flags = Rv64Flags;

    #[inline]
    fn new_leaf(paddr: PhysAddr, flags: Self::Flags, _size: PageSize) -> Self {
        Self::new(paddr_to_ppn_bits(paddr) | flags.to_leaf_bits::<SVPBMT>())
    }

    #[inline]
    fn new_table(paddr: PhysAddr, _level: u8) -> Self {
        Self::new(paddr_to_ppn_bits(paddr) | VALID)
    }

    #[inline]
    fn paddr(&self) -> PhysAddr {
        ppn_bits_to_paddr(self.bits)
    }

    #[inline]
    fn flags(&self) -> Self::Flags {
        Rv64Flags::from_bits::<SVPBMT>(self.bits)
    }

    #[inline]
    fn is_present(&self) -> bool {
        self.bits & VALID != 0
    }

    #[inline]
    fn entry_kind(&self, level: u8) -> PageTableEntryKind {
        if !valid_encoding::<SVPBMT>(self.bits) {
            PageTableEntryKind::Invalid
        } else if self.bits & (READ | WRITE | EXECUTE) != 0 {
            PageTableEntryKind::Leaf
        } else if level > 1 {
            PageTableEntryKind::Table
        } else {
            PageTableEntryKind::Invalid
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.bits = 0;
    }

    #[inline]
    fn bits(&self) -> u64 {
        self.bits
    }

    #[inline]
    fn from_bits(bits: u64) -> Self {
        Self::new(bits)
    }
}

macro_rules! define_rv64_pte {
    ($(#[$meta:meta])* $name:ident, $svpbmt:expr) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Default, PartialEq, Eq)]
        pub struct $name(Rv64PteFor<$svpbmt>);

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($name))
                    .field(&format_args!("{:#018x}", self.0.bits()))
                    .finish()
            }
        }

        impl PageTableEntry for $name {
            type Flags = Rv64Flags;

            #[inline]
            fn new_leaf(paddr: PhysAddr, flags: Self::Flags, size: PageSize) -> Self {
                Self(<Rv64PteFor<$svpbmt> as PageTableEntry>::new_leaf(
                    paddr, flags, size,
                ))
            }

            #[inline]
            fn new_table(paddr: PhysAddr, level: u8) -> Self {
                Self(<Rv64PteFor<$svpbmt> as PageTableEntry>::new_table(
                    paddr, level,
                ))
            }

            #[inline]
            fn paddr(&self) -> PhysAddr {
                self.0.paddr()
            }

            #[inline]
            fn flags(&self) -> Self::Flags {
                self.0.flags()
            }

            #[inline]
            fn is_present(&self) -> bool {
                self.0.is_present()
            }

            #[inline]
            fn entry_kind(&self, level: u8) -> PageTableEntryKind {
                self.0.entry_kind(level)
            }

            #[inline]
            fn clear(&mut self) {
                self.0.clear();
            }

            #[inline]
            fn bits(&self) -> u64 {
                self.0.bits()
            }

            #[inline]
            fn from_bits(bits: u64) -> Self {
                Self(<Rv64PteFor<$svpbmt> as PageTableEntry>::from_bits(bits))
            }
        }
    };
}

define_rv64_pte!(
    /// RISC-V page-table entry using only base PTE bits.
    Rv64Pte,
    false
);
define_rv64_pte!(
    /// RISC-V page-table entry using Svpbmt PBMT bits.
    Rv64SvpbmtPte,
    true
);

macro_rules! define_rv64_meta {
    ($name:ident, $levels:expr, $va_bits:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name;

        impl PagingMetaData for $name {
            const LEVELS: usize = $levels;
            const PA_MAX_BITS: usize = PADDR_MAX_BITS;
            const VA_MAX_BITS: usize = $va_bits;
            type VirtAddr = VirtAddr;

            #[inline]
            fn level_shift(level: u8) -> u32 {
                12 + ((level as u32) - 1) * 9
            }

            #[inline]
            fn level_supports_leaf(level: u8, size: PageSize) -> bool {
                matches!(
                    (level, size),
                    (1, PageSize::Size4K)
                        | (2, PageSize::Size2M)
                        | (3, PageSize::Size1G)
                        | (4, PageSize::Size512G)
                ) && (level as usize) <= Self::LEVELS
            }

            #[inline]
            fn vaddr_is_valid(vaddr: VirtAddr) -> bool {
                canonical_vaddr_is_valid(vaddr, Self::VA_MAX_BITS)
            }
        }
    };
}

define_rv64_meta!(Rv64Meta39, 3, 39);
define_rv64_meta!(Rv64Meta48, 4, 48);
define_rv64_meta!(Rv64Meta57, 5, 57);

fn canonical_vaddr_is_valid(vaddr: VirtAddr, bits: usize) -> bool {
    let v = vaddr.as_usize() as u64;
    if bits >= u64::BITS as usize {
        return true;
    }

    let upper = v >> bits;
    let sign_set = (v & (1u64 << (bits - 1))) != 0;
    if sign_set {
        upper == ((1u64 << (64 - bits)) - 1)
    } else {
        upper == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_round_trip() {
        let flags = Rv64Flags::new(AccessFlags::READ | AccessFlags::WRITE | AccessFlags::USER)
            .with_cache(CachePolicy::WriteCombine);
        let pte = Rv64Pte::new_leaf(PhysAddr::from_usize(0x4000), flags, PageSize::Size4K);

        assert!(pte.is_present());
        assert_eq!(pte.entry_kind(1), PageTableEntryKind::Leaf);
        assert_eq!(pte.paddr(), PhysAddr::from_usize(0x4000));
        assert!(pte.flags().access.contains(AccessFlags::READ));
        assert!(pte.flags().access.contains(AccessFlags::WRITE));
        assert_eq!(pte.flags().cache(), CachePolicy::Writeback);
        assert_eq!(pte.bits() & RV64_PBMT_MASK, 0);
    }

    #[test]
    fn svpbmt_leaf_round_trip_cache_policy() {
        let flags = Rv64Flags::new(AccessFlags::READ).with_cache(CachePolicy::WriteCombine);
        let pte = Rv64SvpbmtPte::new_leaf(PhysAddr::from_usize(0x8000), flags, PageSize::Size4K);
        assert_eq!((pte.bits() & RV64_PBMT_MASK) >> 61, RV64_PBMT_NC);
        assert_eq!(pte.flags().cache(), CachePolicy::WriteCombine);

        let flags = Rv64Flags::new(AccessFlags::READ).with_cache(CachePolicy::Uncached);
        let pte = Rv64SvpbmtPte::new_leaf(PhysAddr::from_usize(0xc000), flags, PageSize::Size4K);
        assert_eq!((pte.bits() & RV64_PBMT_MASK) >> 61, RV64_PBMT_IO);
        assert_eq!(pte.flags().cache(), CachePolicy::Uncached);
    }

    #[test]
    fn base_pte_rejects_svpbmt_bits() {
        let pte = Rv64Pte::from_bits(VALID | READ | ACCESSED | (RV64_PBMT_NC << 61));
        assert_eq!(pte.entry_kind(1), PageTableEntryKind::Invalid);
    }

    #[test]
    fn write_without_read_is_invalid() {
        let pte = Rv64Pte::from_bits(VALID | WRITE | ACCESSED | (1 << 7));
        assert_eq!(pte.entry_kind(1), PageTableEntryKind::Invalid);
    }

    #[test]
    fn table_and_leaf_discriminate_by_rwx() {
        let table = Rv64Pte::new_table(PhysAddr::from_usize(0x4000), 3);
        let leaf = Rv64Pte::new_leaf(
            PhysAddr::from_usize(0x20_0000),
            Rv64Flags::new(AccessFlags::READ),
            PageSize::Size2M,
        );

        assert_eq!(table.entry_kind(3), PageTableEntryKind::Table);
        assert_eq!(table.entry_kind(1), PageTableEntryKind::Invalid);
        assert_eq!(leaf.entry_kind(2), PageTableEntryKind::Leaf);
    }

    #[test]
    fn metadata_covers_sv_widths() {
        assert_eq!(Rv64Meta39::LEVELS, 3);
        assert_eq!(Rv64Meta48::LEVELS, 4);
        assert_eq!(Rv64Meta57::LEVELS, 5);
        assert!(Rv64Meta39::level_supports_leaf(3, PageSize::Size1G));
        assert!(!Rv64Meta39::level_supports_leaf(4, PageSize::Size512G));
        assert!(Rv64Meta48::level_supports_leaf(4, PageSize::Size512G));
        assert!(Rv64Meta57::level_supports_leaf(4, PageSize::Size512G));
    }

    #[test]
    fn canonical_va_accepts_configured_halves() {
        assert!(Rv64Meta39::vaddr_is_valid(VirtAddr::from_usize(
            0x0000_003f_ffff_f000
        )));
        assert!(Rv64Meta39::vaddr_is_valid(VirtAddr::from_usize(
            0xffff_ffff_c000_0000
        )));
        assert!(!Rv64Meta39::vaddr_is_valid(VirtAddr::from_usize(
            0x0000_0040_0000_0000
        )));

        assert!(Rv64Meta57::vaddr_is_valid(VirtAddr::from_usize(
            0x00ff_ffff_ffff_f000
        )));
        assert!(Rv64Meta57::vaddr_is_valid(VirtAddr::from_usize(
            0xff00_0000_0000_0000
        )));
        assert!(!Rv64Meta57::vaddr_is_valid(VirtAddr::from_usize(
            0x0100_0000_0000_0000
        )));
    }
}
