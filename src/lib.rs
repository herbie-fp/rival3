mod eval;
mod interval;
mod mpfr;

pub use eval::ast::Expr;
pub use eval::machine::{Discretization, Hint, Machine, MachineBuilder};
pub use eval::profile::Execution;
pub use eval::run::RivalError;
pub use interval::{ErrorFlags, Ival};
