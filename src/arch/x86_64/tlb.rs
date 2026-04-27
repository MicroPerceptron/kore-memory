use memory_addr::VirtAddr;
use x86_64::{
    instructions::tlb::{InvPcidCommand, Invlpgb, Pcid, flush, flush_all, flush_pcid},
    structures::paging::{
        Size4KiB,
        page::{Page, PageRange},
    },
};

use crate::{PageSize, TlbInvalidation};

// ── TLB invalidation ────────────────────────────────────────────────────────

/// Standard CPU TLB invalidation for x86_64 — `INVLPG` per page,
/// `MOV CR3, CR3` for full reload. Pair with
/// [`crate::arch::x86_64::X86Meta48`] or [`crate::arch::x86_64::X86Meta57`]
/// when the consumer is the CPU MMU.
///
/// IOMMU consumers should use [`crate::NoFlush`] instead — IOMMU
/// invalidation is queue-driven by the controller, not synchronous
/// per-PTE-write.
///
/// INVLPGB (Zen3+) and PCID-scoped invalidation are future optimization
/// targets — override `flush_tlb_range_local` once we can plumb CPUID
/// detection through the kraph platform layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct X86Tlb;

impl TlbInvalidation<VirtAddr> for X86Tlb {
    #[inline]
    fn flush_tlb_local(&self, vaddr: VirtAddr) {
        // `INVLPG` invalidates one 4 KiB TLB entry on the local CPU.
        // ~150-200 cycles on modern Intel/AMD; cheap relative to a CR3
        // reload. Cross-core shootdown is layered above kpte.
        flush(x86_64::VirtAddr::new(vaddr.as_usize() as u64));
    }

    #[inline]
    fn flush_tlb_all_local(&self) {
        // Full local flush via `MOV CR3, CR3` reload. Slower path
        // (~300 cycles + cold-fill cost) but constant-time.
        flush_all();
    }
}

/// PCID-tagged TLB invalidation via `INVPCID` (CPUID Fn0000_0007_EBX\[10]).
///
/// Use when the kernel has enabled `CR4.PCIDE` and tagged each
/// address-space's CR3 with a unique PCID. `INVPCID` invalidates entries
/// for one PCID without disturbing other contexts — much cheaper than a
/// CR3 reload's full flush.
///
/// The [`Pcid`] is captured at construction (typically from the
/// platform's `SpaceIdLease`) and held for the lifetime of the parent
/// [`PageTable`](crate::PageTable). Releasing the PCID lease back to
/// the allocator is the consumer's job — when the `PageTable` drops,
/// this type's `Drop` runs after the tree teardown, which is the right
/// point to release.
#[derive(Clone, Copy, Debug)]
pub struct X86PcidTlb {
    pcid: Pcid,
}

impl X86PcidTlb {
    /// Construct from an already-validated [`Pcid`]. Callers with a raw
    /// `u16` should validate via `Pcid::new(raw)?` first — that's where
    /// the 12-bit bound check lives upstream.
    #[inline]
    pub const fn new(pcid: Pcid) -> Self {
        Self { pcid }
    }

    /// The PCID this TLB layer is tagged with.
    #[inline]
    pub const fn pcid(&self) -> Pcid {
        self.pcid
    }
}

impl TlbInvalidation<VirtAddr> for X86PcidTlb {
    #[inline]
    fn flush_tlb_local(&self, vaddr: VirtAddr) {
        // `INVPCID type=0` (Address): invalidate one (PCID, addr) pair.
        // Cheaper than CR3 reload for non-current PCIDs and finer-grained
        // than INVLPG on the current PCID.
        //
        // SAFETY: caller is responsible for ensuring CPUID INVPCID
        // (Fn0000_0007_EBX[10]) is present before instantiating this
        // type; misuse on a CPU without INVPCID will #UD here.
        unsafe {
            flush_pcid(InvPcidCommand::Address(
                x86_64::VirtAddr::new(vaddr.as_usize() as u64),
                self.pcid,
            ));
        }
    }

    #[inline]
    fn flush_tlb_all_local(&self) {
        // `INVPCID type=1` (Single): invalidate all entries for this
        // PCID. Constant-cost; doesn't touch other PCIDs.
        //
        // SAFETY: same as `flush_tlb_local`.
        unsafe {
            flush_pcid(InvPcidCommand::Single(self.pcid));
        }
    }
}

/// Zen3+ INVLPGB-based invalidation with hardware range flush.
///
/// `INVLPGB` (CPUID Fn8000_0008_EBX\[3]) invalidates a contiguous
/// virtual range in one instruction and broadcasts to other cores
/// participating in the same translation context — replacing what
/// would otherwise be a per-page sweep + IPI on every other CPU.
/// Hugely faster for bulk invalidations.
///
/// # Cross-core consistency
///
/// `INVLPGB` is fire-and-forget; remote CPUs receive the broadcast
/// asynchronously. Without a paired `TLBSYNC`, our function can return
/// while remote CPUs still have stale TLB entries — a real correctness
/// hazard if the caller is about to free the page-table memory or
/// reuse the virtual range. This impl `tlbsync`s after every range
/// flush so callers get a synchronous "all CPUs done" guarantee.
///
/// # Capability plumbing
///
/// Beyond bare range flush, `INVLPGB` accepts:
///
/// - **PCID tag** ([`Self::with_pcid`]): scope the broadcast to one
///   PCID, leaving other contexts untouched. Required for combined
///   PCID + INVLPGB kernels.
/// - **Global pages** ([`Self::with_global`]): include kernel mappings
///   marked `PageTableFlags::GLOBAL`. Off by default to match the
///   conservative "user-context flush" semantics; flip on for kernel-
///   side mappings.
///
/// Construct via [`Self::try_new`] which probes CPUID. Returns `None`
/// on CPUs without INVLPGB; consumers fall back to [`X86Tlb`] or
/// [`X86PcidTlb`]. The cached `Invlpgb` instance avoids re-running
/// CPUID on every flush.
#[derive(Clone, Copy, Debug)]
pub struct X86InvlpgbTlb {
    invlpgb: Invlpgb,
    pcid: Option<Pcid>,
    include_global: bool,
}

impl X86InvlpgbTlb {
    /// Probe CPUID for INVLPGB support; returns `Some(Self)` if
    /// available. Must be called from CPL=0 (kernel mode) — the
    /// upstream `Invlpgb::new()` panics otherwise.
    ///
    /// Defaults: no PCID tag, global pages excluded. Use
    /// [`Self::with_pcid`] / [`Self::with_global`] to customize.
    pub fn try_new() -> Option<Self> {
        Invlpgb::new().map(|invlpgb| Self {
            invlpgb,
            pcid: None,
            include_global: false,
        })
    }

    /// Tag broadcast invalidations with a PCID. The kernel must have
    /// `CR4.PCIDE` enabled. Single-page and full-flush paths route
    /// through `INVPCID` once a PCID is set, since that's the cheaper
    /// instruction for non-broadcast scopes.
    #[inline]
    pub const fn with_pcid(mut self, pcid: Pcid) -> Self {
        self.pcid = Some(pcid);
        self
    }

    /// Include global pages in subsequent range flushes. Required when
    /// invalidating kernel mappings (which set `PageTableFlags::GLOBAL`
    /// to survive cross-context CR3 swaps). Default: off.
    #[inline]
    pub const fn with_global(mut self) -> Self {
        self.include_global = true;
        self
    }

    /// Synchronize: wait for all remote CPUs to acknowledge prior
    /// `INVLPGB` broadcasts. Called automatically after every range
    /// flush; exposed publicly so consumers issuing extra invalidations
    /// outside the walker can request the same guarantee.
    #[inline]
    pub fn tlbsync(&self) {
        self.invlpgb.tlbsync();
    }
}

impl TlbInvalidation<VirtAddr> for X86InvlpgbTlb {
    #[inline]
    fn flush_tlb_local(&self, vaddr: VirtAddr) {
        // Single-page: prefer INVPCID-Address when a PCID is set,
        // INVLPG otherwise. INVLPGB count=1 has higher command-encoding
        // overhead, so we skip it for the one-page case.
        if let Some(pcid) = self.pcid {
            // SAFETY: caller asserts INVPCID + CR4.PCIDE at construction
            // (PCID can only have been minted via Pcid::new from a
            // PCID-enabled kernel).
            unsafe {
                flush_pcid(InvPcidCommand::Address(
                    x86_64::VirtAddr::new(vaddr.as_usize() as u64),
                    pcid,
                ));
            }
        } else {
            X86Tlb.flush_tlb_local(vaddr);
        }
    }

    #[inline]
    fn flush_tlb_all_local(&self) {
        if let Some(pcid) = self.pcid {
            // INVPCID type=1: invalidate all entries for this PCID
            // without touching other contexts. Constant-cost.
            // SAFETY: same as `flush_tlb_local`.
            unsafe {
                flush_pcid(InvPcidCommand::Single(pcid));
            }
        } else {
            X86Tlb.flush_tlb_all_local();
        }
    }

    #[inline]
    fn flush_tlb_range_local(&self, start: VirtAddr, page_size: PageSize, count_pages: usize) {
        if page_size != PageSize::Size4K {
            self.flush_tlb_all_local();
            return;
        }
        // INVLPGB invalidates [start, start + count_pages × 4 KiB) on
        // local + remote CPUs in one instruction. Misaligned `start`
        // (not 4 KiB-aligned) is silently skipped — walker callers
        // never hit that branch.
        let start_va = x86_64::VirtAddr::new(start.as_usize() as u64);
        let Ok(start_page) = Page::<Size4KiB>::from_start_address(start_va) else {
            return;
        };
        let end_page = start_page + count_pages as u64;
        let range = PageRange {
            start: start_page,
            end: end_page,
        };

        let mut builder = self.invlpgb.build();
        if let Some(pcid) = self.pcid {
            // SAFETY: caller asserts CR4.PCIDE at PCID construction time.
            unsafe {
                builder.pcid(pcid);
            }
        }
        if self.include_global {
            builder.include_global();
        }
        builder.pages(range).flush();

        // Wait for remote CPUs to acknowledge. Without this, they may
        // walk stale TLB entries into freed page-table memory.
        self.invlpgb.tlbsync();
    }

    #[inline]
    fn prefer_full_flush(&self, _pending_count: usize) -> bool {
        // This backend has a real range invalidation instruction. Let
        // the cursor preserve coalesced ranges instead of inheriting the
        // generic INVLPG-vs-CR3 break-even tuned for per-page loops.
        false
    }
}
