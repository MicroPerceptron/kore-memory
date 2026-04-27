use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

/// Hardware-supported leaf page sizes.
///
/// Values are the page size in bytes. Each per-arch [`crate::PagingMetaData`]
/// impl declares which variants are legal at which levels via
/// [`crate::PagingMetaData::level_supports_leaf`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum PageSize {
    Size4K = 0x1000,
    Size16K = 0x4000,
    Size64K = 0x1_0000,
    Size2M = 0x20_0000,
    Size32M = 0x200_0000,
    Size512M = 0x2000_0000,
    Size1G = 0x4000_0000,
    Size64G = 0x10_0000_0000,
    Size512G = 0x80_0000_0000,
    Size4T = 0x400_0000_0000,
}

impl PageSize {
    #[inline]
    pub const fn bytes(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn is_huge(self) -> bool {
        !matches!(self, Self::Size4K)
    }

    #[inline]
    pub const fn is_aligned(self, value: usize) -> bool {
        (value & (self.bytes() - 1)) == 0
    }

    #[inline]
    pub const fn from_bytes(bytes: usize) -> Option<Self> {
        match bytes {
            0x1000 => Some(Self::Size4K),
            0x4000 => Some(Self::Size16K),
            0x1_0000 => Some(Self::Size64K),
            0x20_0000 => Some(Self::Size2M),
            0x200_0000 => Some(Self::Size32M),
            0x2000_0000 => Some(Self::Size512M),
            0x4000_0000 => Some(Self::Size1G),
            0x10_0000_0000 => Some(Self::Size64G),
            0x80_0000_0000 => Some(Self::Size512G),
            0x400_0000_0000 => Some(Self::Size4T),
            _ => None,
        }
    }
}

/// Access-permission bits for a page mapping.
///
/// Split from [`CachePolicy`] so cache policy stays a first-class enum
/// rather than an ad-hoc pair of bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct AccessFlags(u8);

impl AccessFlags {
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const EXECUTE: Self = Self(1 << 2);
    pub const USER: Self = Self(1 << 3);
    pub const GLOBAL: Self = Self(1 << 4);

    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[inline]
    pub const fn bits(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn from_bits_truncate(bits: u8) -> Self {
        Self(bits & 0x1f)
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for AccessFlags {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for AccessFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for AccessFlags {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for AccessFlags {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Cache behavior for a page mapping.
///
/// Per-arch encoders translate to:
/// - x86_64: PAT/PWT/PCD bit triples.
/// - aarch64: MAIR_EL1 index.
/// - riscv64: Svpbmt bits when supported; clamped to `Writeback` otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum CachePolicy {
    /// Cached, write-back — default for RAM.
    #[default]
    Writeback,
    /// Strongly-uncached — MMIO register frames.
    Uncached,
    /// Write-combining — framebuffer/GPU scratch.
    WriteCombine,
    /// Write-through — niche; included for completeness.
    WriteThrough,
}

/// Hardware-visible sharing domain for mappings that support one.
///
/// Some architectures fold this into cacheability, some expose explicit
/// descriptor bits, and x86 CPU paging ignores it because coherent cache
/// sharing is architectural. IOMMU/SMMU PTE impls can translate this into
/// the appropriate vendor shareability/snoop controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Shareability {
    /// No hardware sharing domain requested.
    NonShareable,
    /// Inner-domain sharing, commonly the CPU cluster / inner cache domain.
    #[default]
    Inner,
    /// Outer-domain sharing.
    Outer,
    /// System-wide sharing domain.
    System,
}

/// Coherency expectation for a mapping.
///
/// The capability/resource layer decides whether a caller may request a
/// coherent or non-coherent mapping. `kpte` only carries the intent to
/// the arch/vendor PTE encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Coherency {
    /// Hardware does not maintain coherence for this mapping.
    NonCoherent,
    /// Hardware maintains coherence for this mapping.
    #[default]
    Coherent,
}

/// Target-neutral memory attributes carried by arch/vendor mapping flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct MemoryAttributes {
    cache: CachePolicy,
    shareability: Shareability,
    coherency: Coherency,
}

impl MemoryAttributes {
    #[inline]
    pub const fn new(cache: CachePolicy, shareability: Shareability, coherency: Coherency) -> Self {
        Self {
            cache,
            shareability,
            coherency,
        }
    }

    #[inline]
    pub const fn writeback() -> Self {
        Self::new(
            CachePolicy::Writeback,
            Shareability::Inner,
            Coherency::Coherent,
        )
    }

    #[inline]
    pub const fn with_cache(mut self, cache: CachePolicy) -> Self {
        self.cache = cache;
        self
    }

    #[inline]
    pub const fn with_shareability(mut self, shareability: Shareability) -> Self {
        self.shareability = shareability;
        self
    }

    #[inline]
    pub const fn with_coherency(mut self, coherency: Coherency) -> Self {
        self.coherency = coherency;
        self
    }

    #[inline]
    pub const fn cache(self) -> CachePolicy {
        self.cache
    }

    #[inline]
    pub const fn shareability(self) -> Shareability {
        self.shareability
    }

    #[inline]
    pub const fn coherency(self) -> Coherency {
        self.coherency
    }
}
