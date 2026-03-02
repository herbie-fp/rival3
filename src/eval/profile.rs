//! Per-instruction profiling for the rival machine.
//!
//! Rival's evaluator exposes profiling data via [`Execution`] records,
//! which can be accessed via [`Machine::execution_records`](crate::Machine::execution_records)
//! or [`Machine::take_executions`](crate::Machine::take_executions).

/// A single recorded execution of a Rival interval operator.
///
/// Each execution corresponds to a single interval operator being executed.
/// The [`name`](Execution::name) names the operator, except the special
/// name `"adjust"`, in which case it refers to an internal precision-tuning
/// pass.
/// The [`number`](Execution::number) gives its position in the compiled
/// instruction sequence; this allows disambiguating if an expression contains,
/// say, multiple addition operations.
/// The [`precision`](Execution::precision) is the working precision in bits
/// that the operator was executed at,
/// the [`time_ms`](Execution::time_ms) is the time, in milliseconds, that
/// the execution took,
/// and the [`iteration`](Execution::iteration) records which sampling
/// iteration triggered the execution.
///
/// Note that, because Rival executes the register machine multiple times,
/// the same operator (with the same name and number) can appear multiple
/// times for a single point. On the other hand, in some iterations Rival
/// might skip some operators, if the precision is unchanged from previous
/// iterations, so not every operator may show up in the executions list
/// the same number of times.
#[derive(Clone, Copy, Debug)]
pub struct Execution {
    /// The name of the operator, or `"adjust"` for precision-tuning passes.
    pub name: &'static str,
    /// Position in the compiled instruction sequence (-1 for adjust passes).
    pub number: i32,
    /// Working precision in bits for this execution.
    pub precision: u32,
    /// Wall-clock time in milliseconds.
    pub time_ms: f64,
    /// The sampling iteration that triggered this execution.
    pub iteration: usize,
}

impl Default for Execution {
    fn default() -> Self {
        Execution {
            name: "",
            number: 0,
            precision: 0,
            time_ms: 0.0,
            iteration: 0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Profiler {
    records: Vec<Execution>,
    ptr: usize,
}

impl Profiler {
    /// Create a profiler with a fixed capacity (number of records)
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            records: vec![Execution::default(); capacity],
            ptr: 0,
        }
    }

    /// Reset the profiler write pointer without clearing memory
    #[inline]
    pub(crate) fn reset(&mut self) {
        self.ptr = 0;
    }

    /// Slice of current execution records
    #[inline]
    pub(crate) fn records(&self) -> &[Execution] {
        &self.records[..self.ptr]
    }

    /// Record an execution if capacity allows
    #[inline]
    pub(crate) fn record(&mut self, exec: Execution) {
        if self.ptr < self.records.len() {
            self.records[self.ptr] = exec;
            self.ptr += 1;
        }
    }
}
