pub mod eval;
pub mod interval;
mod mpfr;

pub use eval::ast::Expr;
pub use eval::machine::{Discretization, Machine, MachineBuilder};
pub use eval::profile::{Execution, Profiler};
pub use eval::run::RivalError;
pub use interval::{ErrorFlags, Ival};
