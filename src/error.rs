/// Errors returned by walker operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagingError {
    /// Allocator exhausted while allocating a page-table frame.
    OutOfMemory,
    /// Address or size not aligned to the requested page size.
    NotAligned,
    /// No mapping present for this address.
    NotMapped,
    /// Mapping already present where a fresh one was requested.
    AlreadyMapped,
    /// Walk hit a leaf where an intermediate table was expected.
    IntermediateIsLeaf,
    /// Address outside the metadata's VA/IOVA representable range.
    AddressOutOfRange,
    /// Requested page size not supported at the target level.
    UnsupportedSize,
    /// `merge_at` preconditions were not met.
    NotCoalescable,
    /// Mapping request shape is internally inconsistent.
    InvalidMappingShape,
    /// A present entry uses an encoding that is invalid at its walk level.
    InvalidEntryKind,
}

/// Shared result type for walker operations.
pub type PagingResult<T = ()> = Result<T, PagingError>;
