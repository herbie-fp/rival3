//! Per-instruction profiling for the rival machine

#[derive(Clone, Copy, Debug)]
pub struct Execution {
    pub name: &'static str,
    pub number: i32,
    pub precision: u32,
    pub time_ms: f64,
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
