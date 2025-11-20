//! Interval expression evaluation pipeline

pub mod machine;
pub mod profile;
pub mod run;

pub mod ast {
    pub use super::ops::Expr;
}

pub(crate) mod adjust;
pub(crate) mod execute;
pub(crate) mod instructions;
pub(crate) mod macros;
pub(crate) mod ops;
pub(crate) mod tricks;
