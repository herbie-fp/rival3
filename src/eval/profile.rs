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
pub struct Profiler {
    records: Vec<Execution>,
    ptr: usize,
}

impl Profiler {
    /// Create a profiler with a fixed capacity (number of records)
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            records: vec![Execution::default(); capacity],
            ptr: 0,
        }
    }

    /// Reset the profiler write pointer without clearing memory
    #[inline]
    pub fn reset(&mut self) {
        self.ptr = 0;
    }

    /// Current number of buffered records
    #[inline]
    pub fn len(&self) -> usize {
        self.ptr
    }

    /// Check if the profiler has no records
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ptr == 0
    }

    /// Capacity of the buffer
    #[inline]
    pub fn capacity(&self) -> usize {
        self.records.len()
    }

    /// Slice of current execution records
    #[inline]
    pub fn records(&self) -> &[Execution] {
        &self.records[..self.ptr]
    }

    /// Record an execution if capacity allows
    #[inline]
    pub fn record(&mut self, exec: Execution) {
        if self.ptr < self.records.len() {
            self.records[self.ptr] = exec;
            self.ptr += 1;
        }
    }
}
