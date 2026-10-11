use crate::machine::RivalMachine;
use crate::{RivalError, c_slice_mut, status};
use rival::Execution;
use std::ffi::{CString, c_char};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RivalExecution {
    pub instruction_idx: i32,
    pub precision: u32,
    pub time_ms: f64,
    pub iteration: u32,
}

impl From<&Execution> for RivalExecution {
    fn from(exec: &Execution) -> Self {
        Self {
            instruction_idx: exec.number,
            precision: exec.precision,
            time_ms: exec.time_ms,
            iteration: exec.iteration as u32,
        }
    }
}

pub(crate) struct InstructionNames {
    // Holds the NUL-terminated names that `ptrs` points into.
    _names: Vec<CString>,
    ptrs: Vec<*const c_char>,
}

impl InstructionNames {
    pub fn new(names: &[&str]) -> Self {
        let names: Vec<CString> = names
            .iter()
            .map(|&name| CString::new(name).expect("instruction names contain no NUL"))
            .collect();
        let ptrs = names.iter().map(|name| name.as_ptr()).collect();
        Self {
            _names: names,
            ptrs,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_set_profiling(
    machine: *mut RivalMachine,
    enabled: bool,
) -> RivalError {
    status(|| {
        let machine = unsafe { machine.as_mut() }.ok_or(RivalError::Misuse)?;
        machine.machine.set_profiling_enabled(enabled);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_get_profiling(machine: *const RivalMachine) -> bool {
    unsafe { machine.as_ref() }.is_some_and(|machine| machine.machine.profiling_enabled())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_profiler_reset(machine: *mut RivalMachine) -> RivalError {
    status(|| {
        let machine = unsafe { machine.as_mut() }.ok_or(RivalError::Misuse)?;
        machine.machine.clear_executions();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_profiler_executions(
    machine: *const RivalMachine,
    out: *mut RivalExecution,
    capacity: usize,
    out_len: *mut usize,
) -> RivalError {
    status(|| {
        let machine = unsafe { machine.as_ref() }.ok_or(RivalError::Misuse)?;
        let out = unsafe { c_slice_mut(out, capacity)? };
        let out_len = unsafe { out_len.as_mut() }.ok_or(RivalError::Misuse)?;

        let records = machine.machine.execution_records();
        for (slot, record) in out.iter_mut().zip(records) {
            *slot = record.into();
        }
        *out_len = records.len();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_instruction_names(
    machine: *const RivalMachine,
    out_names: *mut *const *const c_char,
    out_len: *mut usize,
) -> RivalError {
    status(|| {
        let machine = unsafe { machine.as_ref() }.ok_or(RivalError::Misuse)?;
        let out_names = unsafe { out_names.as_mut() }.ok_or(RivalError::Misuse)?;
        let out_len = unsafe { out_len.as_mut() }.ok_or(RivalError::Misuse)?;

        let names = &machine.instruction_names.ptrs;
        *out_names = names.as_ptr();
        *out_len = names.len();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_iterations(machine: *const RivalMachine) -> u32 {
    unsafe { machine.as_ref() }.map_or(0, |machine| machine.machine.iterations() as u32)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_bumps(machine: *const RivalMachine) -> u32 {
    unsafe { machine.as_ref() }.map_or(0, |machine| machine.machine.bumps() as u32)
}
