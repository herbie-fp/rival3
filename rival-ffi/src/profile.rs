use std::collections::HashMap;

#[repr(C)]
#[derive(Clone)]
pub struct RivalExecution {
    pub instruction_idx: i32,
    pub precision: u32,
    pub time_ms: f64,
    pub iteration: u32,
}

#[repr(C)]
#[derive(Clone)]
pub struct RivalAggregatedProfile {
    pub instruction_idx: i32,
    pub precision_bucket: u32,
    pub total_time_ms: f64,
    pub count: usize,
}

#[repr(C)]
pub struct RivalProfileSummary {
    pub entries: *const RivalAggregatedProfile,
    pub len: usize,
    pub bumps: u32,
    pub iterations: u32,
}

pub(crate) struct ProfileCache {
    pub executions: Vec<RivalExecution>,
    pub aggregated: Vec<RivalAggregatedProfile>,
    pub last_bumps: u32,
    pub last_iterations: u32,
}

impl ProfileCache {
    pub fn new() -> Self {
        Self {
            executions: Vec::new(),
            aggregated: Vec::new(),
            last_bumps: 0,
            last_iterations: 0,
        }
    }

    pub fn aggregate_from<'a, I>(
        &mut self,
        records: I,
        bucket_size: u32,
        bumps: usize,
        iterations: usize,
    ) -> RivalProfileSummary
    where
        I: Iterator<Item = &'a rival::Execution> + Clone,
    {
        let bucket_size = bucket_size.max(1);
        self.cache_executions(records.clone());
        let mut aggregation: HashMap<(i32, u32), (f64, usize)> = HashMap::new();

        for exec in records {
            let precision_bucket = exec.precision - (exec.precision % bucket_size);
            let key = (exec.number, precision_bucket);
            let entry = aggregation.entry(key).or_insert((0.0, 0));
            entry.0 += exec.time_ms;
            entry.1 += 1;
        }

        self.aggregated.clear();
        self.aggregated.reserve(aggregation.len());
        for ((instruction_idx, precision_bucket), (total_time_ms, count)) in aggregation {
            self.aggregated.push(RivalAggregatedProfile {
                instruction_idx,
                precision_bucket,
                total_time_ms,
                count,
            });
        }
        self.aggregated
            .sort_by_key(|e| (e.instruction_idx, e.precision_bucket));

        self.last_bumps = bumps as u32;
        self.last_iterations = iterations as u32;

        RivalProfileSummary {
            entries: if self.aggregated.is_empty() {
                std::ptr::null()
            } else {
                self.aggregated.as_ptr()
            },
            len: self.aggregated.len(),
            bumps: self.last_bumps,
            iterations: self.last_iterations,
        }
    }

    pub fn summary_from_cache(&self) -> RivalProfileSummary {
        RivalProfileSummary {
            entries: if self.aggregated.is_empty() {
                std::ptr::null()
            } else {
                self.aggregated.as_ptr()
            },
            len: self.aggregated.len(),
            bumps: self.last_bumps,
            iterations: self.last_iterations,
        }
    }

    pub fn cache_executions<'a, I>(&mut self, records: I)
    where
        I: Iterator<Item = &'a rival::Execution>,
    {
        self.executions.clear();
        for exec in records {
            self.executions.push(RivalExecution {
                instruction_idx: exec.number,
                precision: exec.precision,
                time_ms: exec.time_ms,
                iteration: exec.iteration as u32,
            });
        }
    }
}
