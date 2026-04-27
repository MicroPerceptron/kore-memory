use aarch64_cpu::{
    asm::barrier::{SY, isb},
    registers::{DAIF, MAIR_EL1, Readable, SCTLR_EL1, TCR_EL1, TTBR0_EL1, TTBR1_EL1, Writeable},
};
use memory_addr::{MemoryAddr, PhysAddr};

use crate::{AddrSpaceActivation, AddrSpaceToken, PageSize, PagingError, PagingResult};

/// AArch64 EL1 translation-table base register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum A64Ttbr {
    Ttbr0El1,
    Ttbr1El1,
}

/// Common TTBR root encoding for EL1 stage-1 translation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct A64TtbrConfig {
    register: A64Ttbr,
    asid: u16,
    common_not_private: bool,
}

impl A64TtbrConfig {
    #[inline]
    pub const fn new(register: A64Ttbr) -> Self {
        Self {
            register,
            asid: 0,
            common_not_private: false,
        }
    }

    #[inline]
    pub const fn with_asid(mut self, asid: u16) -> Self {
        self.asid = asid;
        self
    }

    #[inline]
    pub const fn with_common_not_private(mut self, common_not_private: bool) -> Self {
        self.common_not_private = common_not_private;
        self
    }

    #[inline]
    pub const fn register(self) -> A64Ttbr {
        self.register
    }

    #[inline]
    pub const fn asid(self) -> u16 {
        self.asid
    }

    #[inline]
    pub const fn common_not_private(self) -> bool {
        self.common_not_private
    }
}

/// AArch64 EL1 paging-control values to apply around a TTBR switch.
///
/// Activation writes MAIR_EL1 before TCR_EL1, then TTBRx_EL1, then SCTLR_EL1
/// to match the architectural translation-control sequencing. Toggling
/// SCTLR.M changes whether the active TTBR root is interpreted by the MMU, so
/// the root and controls must already describe a coherent enabled/disabled
/// transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct A64PagingControls {
    mair_el1: Option<u64>,
    tcr_el1: Option<u64>,
    ttbr: A64TtbrConfig,
    sctlr_el1: Option<u64>,
}

impl A64PagingControls {
    #[inline]
    pub const fn new(ttbr: A64TtbrConfig) -> Self {
        Self {
            mair_el1: None,
            tcr_el1: None,
            ttbr,
            sctlr_el1: None,
        }
    }

    #[inline]
    pub const fn with_mair_el1(mut self, value: u64) -> Self {
        self.mair_el1 = Some(value);
        self
    }

    #[inline]
    pub const fn with_tcr_el1(mut self, value: u64) -> Self {
        self.tcr_el1 = Some(value);
        self
    }

    #[inline]
    pub const fn with_sctlr_el1(mut self, value: u64) -> Self {
        self.sctlr_el1 = Some(value);
        self
    }

    #[inline]
    pub const fn mair_el1(self) -> Option<u64> {
        self.mair_el1
    }

    #[inline]
    pub const fn tcr_el1(self) -> Option<u64> {
        self.tcr_el1
    }

    #[inline]
    pub const fn ttbr(self) -> A64TtbrConfig {
        self.ttbr
    }

    #[inline]
    pub const fn sctlr_el1(self) -> Option<u64> {
        self.sctlr_el1
    }
}

/// Installed AArch64 address-space token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct A64PagingToken {
    root: PhysAddr,
    controls: A64PagingControls,
    ttbr_value: u64,
}

impl A64PagingToken {
    #[inline]
    pub const fn root(self) -> PhysAddr {
        self.root
    }

    #[inline]
    pub const fn controls(self) -> A64PagingControls {
        self.controls
    }

    #[inline]
    pub const fn ttbr_value(self) -> u64 {
        self.ttbr_value
    }
}

impl AddrSpaceToken for A64PagingToken {
    #[inline]
    fn root(self) -> PhysAddr {
        self.root
    }
}

/// CPU-local AArch64 EL1 paging activation policy.
#[derive(Clone, Copy, Debug)]
pub struct A64PagingActivation {
    current_ttbr: A64Ttbr,
}

impl A64PagingActivation {
    #[inline]
    pub const fn new(current_ttbr: A64Ttbr) -> Self {
        Self { current_ttbr }
    }

    #[inline]
    pub const fn ttbr0_el1() -> Self {
        Self::new(A64Ttbr::Ttbr0El1)
    }

    #[inline]
    pub const fn ttbr1_el1() -> Self {
        Self::new(A64Ttbr::Ttbr1El1)
    }

    #[inline]
    pub const fn current_ttbr(self) -> A64Ttbr {
        self.current_ttbr
    }

    #[inline]
    pub unsafe fn write_mair_el1(value: u64) {
        with_irq_disabled(|| {
            MAIR_EL1.set(value);
        });
    }

    #[inline]
    pub unsafe fn write_tcr_el1(value: u64) {
        with_irq_disabled(|| {
            TCR_EL1.set(value);
            instruction_synchronization_barrier();
        });
    }

    #[inline]
    pub unsafe fn write_ttbr0_el1(value: u64) {
        with_irq_disabled(|| {
            TTBR0_EL1.set(value);
            instruction_synchronization_barrier();
        });
    }

    #[inline]
    pub unsafe fn write_ttbr1_el1(value: u64) {
        with_irq_disabled(|| {
            TTBR1_EL1.set(value);
            instruction_synchronization_barrier();
        });
    }

    #[inline]
    pub unsafe fn write_sctlr_el1(value: u64) {
        with_irq_disabled(|| {
            SCTLR_EL1.set(value);
            instruction_synchronization_barrier();
        });
    }
}

impl AddrSpaceActivation for A64PagingActivation {
    type Token = A64PagingToken;
    type Controls = A64PagingControls;

    #[inline]
    fn install(&self, root: PhysAddr, controls: Self::Controls) -> PagingResult<Self::Token> {
        let ttbr_value = encode_ttbr(root, controls.ttbr())?;
        Ok(A64PagingToken {
            root,
            controls,
            ttbr_value,
        })
    }

    #[inline]
    unsafe fn activate(&self, token: Self::Token) -> PagingResult {
        with_irq_disabled(|| {
            // ARM translation-control order: MAIR, TCR, TTBR, then SCTLR.M.
            if let Some(value) = token.controls.mair_el1() {
                MAIR_EL1.set(value);
            }
            if let Some(value) = token.controls.tcr_el1() {
                TCR_EL1.set(value);
                instruction_synchronization_barrier();
            }
            match token.controls.ttbr().register() {
                A64Ttbr::Ttbr0El1 => TTBR0_EL1.set(token.ttbr_value),
                A64Ttbr::Ttbr1El1 => TTBR1_EL1.set(token.ttbr_value),
            }
            instruction_synchronization_barrier();
            if let Some(value) = token.controls.sctlr_el1() {
                SCTLR_EL1.set(value);
                instruction_synchronization_barrier();
            }
        });
        Ok(())
    }

    #[inline]
    fn current(&self) -> PagingResult<Option<Self::Token>> {
        with_irq_disabled(|| {
            let register = self.current_ttbr;
            let ttbr_value = match register {
                A64Ttbr::Ttbr0El1 => TTBR0_EL1.get(),
                A64Ttbr::Ttbr1El1 => TTBR1_EL1.get(),
            };
            let ttbr = match register {
                A64Ttbr::Ttbr0El1 => A64TtbrConfig::new(register)
                    .with_asid(TTBR0_EL1.read(TTBR0_EL1::ASID) as u16)
                    .with_common_not_private(TTBR0_EL1.read(TTBR0_EL1::CnP) != 0),
                A64Ttbr::Ttbr1El1 => A64TtbrConfig::new(register)
                    .with_asid(TTBR1_EL1.read(TTBR1_EL1::ASID) as u16)
                    .with_common_not_private(TTBR1_EL1.read(TTBR1_EL1::CnP) != 0),
            };
            Ok(Some(A64PagingToken {
                root: PhysAddr::from(ttbr_root(ttbr_value) as usize),
                controls: A64PagingControls {
                    mair_el1: Some(MAIR_EL1.get()),
                    tcr_el1: Some(TCR_EL1.get()),
                    ttbr,
                    sctlr_el1: Some(SCTLR_EL1.get()),
                },
                ttbr_value,
            }))
        })
    }
}

#[inline]
fn encode_ttbr(root: PhysAddr, ttbr: A64TtbrConfig) -> PagingResult<u64> {
    if !root.is_aligned(PageSize::Size4K.bytes()) {
        return Err(PagingError::NotAligned);
    }

    let root = root.as_usize() as u64;
    if root & !0x0000_ffff_ffff_fffe != 0 {
        return Err(PagingError::AddressOutOfRange);
    }

    let cnp = u64::from(ttbr.common_not_private());
    Ok(match ttbr.register() {
        A64Ttbr::Ttbr0El1 => {
            {
                TTBR0_EL1::ASID.val(ttbr.asid() as u64)
                    + TTBR0_EL1::BADDR.val(root >> 1)
                    + TTBR0_EL1::CnP.val(cnp)
            }
            .value
        }
        A64Ttbr::Ttbr1El1 => {
            {
                TTBR1_EL1::ASID.val(ttbr.asid() as u64)
                    + TTBR1_EL1::BADDR.val(root >> 1)
                    + TTBR1_EL1::CnP.val(cnp)
            }
            .value
        }
    })
}

#[inline]
fn ttbr_root(value: u64) -> u64 {
    TTBR0_EL1::BADDR.read(value) << 1
}

// Intentionally per-arch: interrupt masking touches DAIF on AArch64.
#[inline]
fn with_irq_disabled<R>(f: impl FnOnce() -> R) -> R {
    let saved = DAIF.get();
    DAIF.set(saved | (1 << 7));
    let result = f();
    DAIF.set(saved);
    result
}

#[inline]
fn instruction_synchronization_barrier() {
    isb(SY);
}
