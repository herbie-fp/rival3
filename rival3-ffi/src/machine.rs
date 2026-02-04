use crate::RivalError;
use crate::discretization::RivalDiscretization;
use crate::expr::RivalExprArena;
use crate::hints::RivalHints;
use crate::profile::{ProfileCache, RivalExecution, RivalProfileSummary};
use gmp_mpfr_sys::mpfr::{self, mpfr_t};
use rival::eval::machine::{Machine, MachineBuilder};
use rival::{Discretization, ErrorFlags, Ival, RivalError as CoreError};
use rug::Float;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::panic;
use std::ptr;
use std::slice;

pub struct RivalMachine {
    pub(crate) machine: Machine<RivalDiscretization>,
    pub(crate) arg_buf: Vec<Ival>,
    pub(crate) rect_buf: Vec<Ival>,
    pub(crate) profile_cache: ProfileCache,
    pub(crate) instruction_names_cache: Vec<u8>,
    pub(crate) n_vars: usize,
    pub(crate) n_exprs: usize,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RivalProfilingMode {
    Off = 0,
    On = 1,
}

#[repr(C)]
pub struct RivalAnalyzeResult {
    pub error: RivalError,
    pub is_error: bool,
    pub maybe_error: bool,
    pub converged: bool,
    pub hints: *mut RivalHints,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_configure_baseline(machine: *mut RivalMachine) -> bool {
    let result = panic::catch_unwind(|| {
        if machine.is_null() {
            return false;
        }

        let wrapper = unsafe { &mut *machine };
        wrapper.machine.configure_baseline();
        true
    });

    result.unwrap_or(false)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_new(
    arena: *const RivalExprArena,
    expr_handles: *const u32,
    n_exprs: usize,
    vars: *const *const c_char,
    n_vars: usize,
    disc: *const RivalDiscretization,
    max_precision: u32,
    profile_capacity: usize,
) -> *mut RivalMachine {
    let result = panic::catch_unwind(|| {
        if arena.is_null() || disc.is_null() || (expr_handles.is_null() && n_exprs > 0) {
            return ptr::null_mut();
        }

        unsafe {
            let arena_ref = &*arena;
            let handles = if n_exprs > 0 {
                slice::from_raw_parts(expr_handles, n_exprs)
            } else {
                &[]
            };

            let exprs_vec = match arena_ref.materialize_all(handles) {
                Some(exprs) => exprs,
                None => return ptr::null_mut(),
            };

            let vars_vec: Vec<String> = if vars.is_null() || n_vars == 0 {
                Vec::new()
            } else {
                let vars_slice = slice::from_raw_parts(vars, n_vars);
                let result: Option<Vec<String>> = vars_slice
                    .iter()
                    .map(|ptr| {
                        if ptr.is_null() {
                            None
                        } else {
                            Some(CStr::from_ptr(*ptr).to_string_lossy().into_owned())
                        }
                    })
                    .collect();
                match result {
                    Some(v) if v.len() == n_vars => v,
                    _ => return ptr::null_mut(),
                }
            };

            let disc_cloned = (*disc).clone();
            let machine = MachineBuilder::new(disc_cloned)
                .max_precision(max_precision)
                .profile_capacity(profile_capacity)
                .build(exprs_vec, vars_vec);

            let arg_prec = machine.disc.target().max(machine.min_precision);
            let arg_buf: Vec<Ival> = (0..n_vars).map(|_| Ival::zero(arg_prec)).collect();
            let rect_buf: Vec<Ival> = (0..n_vars)
                .map(|_| {
                    Ival::from_lo_hi(Float::with_val(arg_prec, 0), Float::with_val(arg_prec, 0))
                })
                .collect();

            Box::into_raw(Box::new(RivalMachine {
                machine,
                arg_buf,
                rect_buf,
                profile_cache: ProfileCache::new(),
                instruction_names_cache: Vec::new(),
                n_vars,
                n_exprs,
            }))
        }
    });

    result.unwrap_or(ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_free(machine: *mut RivalMachine) {
    if !machine.is_null() {
        unsafe { drop(Box::from_raw(machine)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_instruction_count(machine: *const RivalMachine) -> usize {
    if machine.is_null() {
        0
    } else {
        unsafe { (*machine).machine.instruction_count() }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_var_count(machine: *const RivalMachine) -> usize {
    if machine.is_null() {
        0
    } else {
        unsafe { (*machine).n_vars }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_expr_count(machine: *const RivalMachine) -> usize {
    if machine.is_null() {
        0
    } else {
        unsafe { (*machine).n_exprs }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_apply(
    machine: *mut RivalMachine,
    args: *const *const mpfr_t,
    n_args: usize,
    out: *const *mut mpfr_t,
    n_out: usize,
    hints: *const RivalHints,
    max_iterations: usize,
    max_precision: u32,
) -> RivalError {
    let result = panic::catch_unwind(|| {
        if machine.is_null() || out.is_null() || (args.is_null() && n_args > 0) {
            return RivalError::InvalidInput;
        }

        let wrapper = unsafe { &mut *machine };

        if n_args != wrapper.n_vars || n_out != wrapper.n_exprs {
            return RivalError::InvalidInput;
        }

        if !hints.is_null() && unsafe { (*hints).hints.len() } != wrapper.machine.instructions.len()
        {
            return RivalError::InvalidInput;
        }

        let m = &mut wrapper.machine;
        m.max_precision = max_precision;

        if n_args > 0 {
            let arg_ptrs = unsafe { slice::from_raw_parts(args, n_args) };

            // Validate all MPFR pointers before use
            for &ptr in arg_ptrs.iter() {
                if ptr.is_null() {
                    return RivalError::InvalidInput;
                }
            }

            for (i, &ptr) in arg_ptrs.iter().enumerate() {
                let ival = &mut wrapper.arg_buf[i];
                // Match precisions
                let src_prec = unsafe { mpfr::get_prec(ptr) };
                ival.lo.as_float_mut().set_prec(src_prec as u32);
                ival.hi.as_float_mut().set_prec(src_prec as u32);
                unsafe {
                    mpfr::set(ival.lo.as_float_mut().as_raw_mut(), ptr, mpfr::rnd_t::RNDN);
                    mpfr::set(ival.hi.as_float_mut().as_raw_mut(), ptr, mpfr::rnd_t::RNDN);
                }
                ival.lo.immovable = true;
                ival.hi.immovable = true;
                ival.err = if ival.lo.as_float().is_finite() {
                    ErrorFlags::none()
                } else {
                    ErrorFlags::error()
                };
            }
        }

        let hints_opt = if hints.is_null() {
            None
        } else {
            Some(unsafe { (*hints).hints.as_slice() })
        };
        let apply_result = m.apply(&wrapper.arg_buf, hints_opt, max_iterations);

        match apply_result {
            Ok(outputs) => {
                if outputs.len() != n_out {
                    return RivalError::InvalidInput;
                }
                let out_ptrs = unsafe { slice::from_raw_parts(out, n_out) };
                for &out_ptr in out_ptrs.iter() {
                    if out_ptr.is_null() {
                        return RivalError::InvalidInput;
                    }
                }
                for (i, val) in outputs.iter().enumerate() {
                    unsafe {
                        mpfr::set(out_ptrs[i], val.lo.as_float().as_raw(), mpfr::rnd_t::RNDN)
                    };
                }
                RivalError::Ok
            }
            Err(CoreError::InvalidInput) => RivalError::InvalidInput,
            Err(CoreError::Unsamplable) => RivalError::Unsamplable,
        }
    });

    result.unwrap_or(RivalError::Panic)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_apply_baseline(
    machine: *mut RivalMachine,
    args: *const *const mpfr_t,
    n_args: usize,
    out: *const *mut mpfr_t,
    n_out: usize,
    hints: *const RivalHints,
    max_precision: u32,
) -> RivalError {
    let result = panic::catch_unwind(|| {
        if machine.is_null() || out.is_null() || (args.is_null() && n_args > 0) {
            return RivalError::InvalidInput;
        }

        let wrapper = unsafe { &mut *machine };

        if n_args != wrapper.n_vars || n_out != wrapper.n_exprs {
            return RivalError::InvalidInput;
        }

        if !hints.is_null() && unsafe { (*hints).hints.len() } != wrapper.machine.instructions.len()
        {
            return RivalError::InvalidInput;
        }

        let m = &mut wrapper.machine;
        m.max_precision = max_precision;

        if n_args > 0 {
            let arg_ptrs = unsafe { slice::from_raw_parts(args, n_args) };

            // Validate all MPFR pointers before use
            for &ptr in arg_ptrs.iter() {
                if ptr.is_null() {
                    return RivalError::InvalidInput;
                }
            }

            for (i, &ptr) in arg_ptrs.iter().enumerate() {
                let ival = &mut wrapper.arg_buf[i];
                // Match precisions
                let src_prec = unsafe { mpfr::get_prec(ptr) };
                ival.lo.as_float_mut().set_prec(src_prec as u32);
                ival.hi.as_float_mut().set_prec(src_prec as u32);
                unsafe {
                    mpfr::set(ival.lo.as_float_mut().as_raw_mut(), ptr, mpfr::rnd_t::RNDN);
                    mpfr::set(ival.hi.as_float_mut().as_raw_mut(), ptr, mpfr::rnd_t::RNDN);
                }
                ival.lo.immovable = true;
                ival.hi.immovable = true;
                ival.err = if ival.lo.as_float().is_finite() {
                    ErrorFlags::none()
                } else {
                    ErrorFlags::error()
                };
            }
        }

        let hints_opt = if hints.is_null() {
            None
        } else {
            Some(unsafe { (*hints).hints.as_slice() })
        };
        let apply_result = m.apply_baseline(&wrapper.arg_buf, hints_opt);

        match apply_result {
            Ok(outputs) => {
                if outputs.len() != n_out {
                    return RivalError::InvalidInput;
                }
                let out_ptrs = unsafe { slice::from_raw_parts(out, n_out) };
                for &out_ptr in out_ptrs.iter() {
                    if out_ptr.is_null() {
                        return RivalError::InvalidInput;
                    }
                }
                for (i, val) in outputs.iter().enumerate() {
                    unsafe {
                        mpfr::set(out_ptrs[i], val.lo.as_float().as_raw(), mpfr::rnd_t::RNDN)
                    };
                }
                RivalError::Ok
            }
            Err(CoreError::InvalidInput) => RivalError::InvalidInput,
            Err(CoreError::Unsamplable) => RivalError::Unsamplable,
        }
    });

    result.unwrap_or(RivalError::Panic)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_analyze_with_hints(
    machine: *mut RivalMachine,
    rect: *const *const mpfr_t,
    n_args: usize,
    hints: *const RivalHints,
) -> RivalAnalyzeResult {
    let result = panic::catch_unwind(|| {
        if machine.is_null() || rect.is_null() {
            return RivalAnalyzeResult {
                error: RivalError::InvalidInput,
                is_error: true,
                maybe_error: true,
                converged: false,
                hints: ptr::null_mut(),
            };
        }

        let wrapper = unsafe { &mut *machine };

        if n_args != wrapper.n_vars {
            return RivalAnalyzeResult {
                error: RivalError::InvalidInput,
                is_error: true,
                maybe_error: true,
                converged: false,
                hints: ptr::null_mut(),
            };
        }

        if !hints.is_null() && unsafe { (*hints).hints.len() } != wrapper.machine.instructions.len()
        {
            return RivalAnalyzeResult {
                error: RivalError::InvalidInput,
                is_error: true,
                maybe_error: true,
                converged: false,
                hints: ptr::null_mut(),
            };
        }

        let m = &mut wrapper.machine;
        let rect_ptrs = unsafe { slice::from_raw_parts(rect, n_args * 2) };

        // Validate all MPFR pointers before use
        for &ptr in rect_ptrs.iter() {
            if ptr.is_null() {
                return RivalAnalyzeResult {
                    error: RivalError::InvalidInput,
                    is_error: true,
                    maybe_error: true,
                    converged: false,
                    hints: ptr::null_mut(),
                };
            }
        }

        for i in 0..n_args {
            let lo_ptr = rect_ptrs[2 * i];
            let hi_ptr = rect_ptrs[2 * i + 1];

            let lo_prec = unsafe { mpfr::get_prec(lo_ptr) } as u32;
            let hi_prec = unsafe { mpfr::get_prec(hi_ptr) } as u32;
            let prec = lo_prec.max(hi_prec);

            let ival = &mut wrapper.rect_buf[i];
            ival.lo.as_float_mut().set_prec(prec);
            ival.hi.as_float_mut().set_prec(prec);

            unsafe {
                mpfr::set(
                    ival.lo.as_float_mut().as_raw_mut(),
                    lo_ptr,
                    mpfr::rnd_t::RNDN,
                );
                mpfr::set(
                    ival.hi.as_float_mut().as_raw_mut(),
                    hi_ptr,
                    mpfr::rnd_t::RNDN,
                );
            }

            let fixed = { ival.lo.as_float() == ival.hi.as_float() };
            let err = {
                let lo = ival.lo.as_float();
                let hi = ival.hi.as_float();
                lo.is_nan() || hi.is_nan() || (fixed && lo.is_infinite())
            };
            ival.lo.immovable = fixed;
            ival.hi.immovable = fixed;
            ival.err = if err {
                ErrorFlags::error()
            } else {
                ErrorFlags::none()
            };
        }

        let hints_opt = if hints.is_null() {
            None
        } else {
            Some(unsafe { (*hints).hints.as_slice() })
        };
        let (status, next_hints, converged) = m.analyze_with_hints(&wrapper.rect_buf, hints_opt);

        RivalAnalyzeResult {
            error: RivalError::Ok,
            is_error: !status.lo.as_float().is_zero(),
            maybe_error: !status.hi.as_float().is_zero(),
            converged,
            hints: Box::into_raw(Box::new(RivalHints { hints: next_hints })),
        }
    });

    result.unwrap_or(RivalAnalyzeResult {
        error: RivalError::Panic,
        is_error: true,
        maybe_error: true,
        converged: false,
        hints: ptr::null_mut(),
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_analyze_baseline_with_hints(
    machine: *mut RivalMachine,
    rect: *const *const mpfr_t,
    n_args: usize,
    hints: *const RivalHints,
) -> RivalAnalyzeResult {
    let result = panic::catch_unwind(|| {
        if machine.is_null() || rect.is_null() {
            return RivalAnalyzeResult {
                error: RivalError::InvalidInput,
                is_error: true,
                maybe_error: true,
                converged: false,
                hints: ptr::null_mut(),
            };
        }

        let wrapper = unsafe { &mut *machine };

        if n_args != wrapper.n_vars {
            return RivalAnalyzeResult {
                error: RivalError::InvalidInput,
                is_error: true,
                maybe_error: true,
                converged: false,
                hints: ptr::null_mut(),
            };
        }

        if !hints.is_null() && unsafe { (*hints).hints.len() } != wrapper.machine.instructions.len()
        {
            return RivalAnalyzeResult {
                error: RivalError::InvalidInput,
                is_error: true,
                maybe_error: true,
                converged: false,
                hints: ptr::null_mut(),
            };
        }

        let m = &mut wrapper.machine;
        let rect_ptrs = unsafe { slice::from_raw_parts(rect, n_args * 2) };

        // Validate all MPFR pointers before use
        for &ptr in rect_ptrs.iter() {
            if ptr.is_null() {
                return RivalAnalyzeResult {
                    error: RivalError::InvalidInput,
                    is_error: true,
                    maybe_error: true,
                    converged: false,
                    hints: ptr::null_mut(),
                };
            }
        }

        for i in 0..n_args {
            let lo_ptr = rect_ptrs[2 * i];
            let hi_ptr = rect_ptrs[2 * i + 1];

            let lo_prec = unsafe { mpfr::get_prec(lo_ptr) } as u32;
            let hi_prec = unsafe { mpfr::get_prec(hi_ptr) } as u32;
            let prec = lo_prec.max(hi_prec);

            let ival = &mut wrapper.rect_buf[i];
            ival.lo.as_float_mut().set_prec(prec);
            ival.hi.as_float_mut().set_prec(prec);

            unsafe {
                mpfr::set(
                    ival.lo.as_float_mut().as_raw_mut(),
                    lo_ptr,
                    mpfr::rnd_t::RNDN,
                );
                mpfr::set(
                    ival.hi.as_float_mut().as_raw_mut(),
                    hi_ptr,
                    mpfr::rnd_t::RNDN,
                );
            }

            let fixed = { ival.lo.as_float() == ival.hi.as_float() };
            let err = {
                let lo = ival.lo.as_float();
                let hi = ival.hi.as_float();
                lo.is_nan() || hi.is_nan() || (fixed && lo.is_infinite())
            };
            ival.lo.immovable = fixed;
            ival.hi.immovable = fixed;
            ival.err = if err {
                ErrorFlags::error()
            } else {
                ErrorFlags::none()
            };
        }

        let hints_opt = if hints.is_null() {
            None
        } else {
            Some(unsafe { (*hints).hints.as_slice() })
        };
        let (status, next_hints, converged) = m.analyze_baseline_with_hints(&wrapper.rect_buf, hints_opt);

        RivalAnalyzeResult {
            error: RivalError::Ok,
            is_error: !status.lo.as_float().is_zero(),
            maybe_error: !status.hi.as_float().is_zero(),
            converged,
            hints: Box::into_raw(Box::new(RivalHints { hints: next_hints })),
        }
    });

    result.unwrap_or(RivalAnalyzeResult {
        error: RivalError::Panic,
        is_error: true,
        maybe_error: true,
        converged: false,
        hints: ptr::null_mut(),
    })
}

// TODO: not ffi bound, look into naming (I think we made an API mistake)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_analyze(
    machine: *mut RivalMachine,
    rect: *const *const mpfr_t,
    n_args: usize,
    is_error: *mut bool,
    maybe_error: *mut bool,
) -> RivalError {
    let result = unsafe { rival_analyze_with_hints(machine, rect, n_args, ptr::null()) };
    if !is_error.is_null() {
        unsafe { *is_error = result.is_error };
    }
    if !maybe_error.is_null() {
        unsafe { *maybe_error = result.maybe_error };
    }
    if !result.hints.is_null() {
        unsafe { crate::hints::rival_hints_free(result.hints) };
    }
    result.error
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_profiler_count(machine: *const RivalMachine) -> usize {
    if machine.is_null() {
        0
    } else {
        unsafe { (*machine).machine.profiler.records().len() }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_profiler_get(
    machine: *const RivalMachine,
    idx: usize,
    out: *mut RivalExecution,
) -> bool {
    if machine.is_null() || out.is_null() {
        return false;
    }
    let records = unsafe { (*machine).machine.profiler.records() };
    if idx >= records.len() {
        return false;
    }
    let exec = &records[idx];
    unsafe {
        *out = RivalExecution {
            instruction_idx: exec.number,
            precision: exec.precision,
            time_ms: exec.time_ms,
            iteration: exec.iteration as u32,
        };
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_profiler_reset(machine: *mut RivalMachine) {
    if !machine.is_null() {
        unsafe { (*machine).machine.profiler.reset() };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_profiler_aggregate(
    machine: *mut RivalMachine,
    bucket_size: u32,
) -> RivalProfileSummary {
    let result = panic::catch_unwind(|| {
        if machine.is_null() {
            return RivalProfileSummary {
                entries: ptr::null(),
                len: 0,
                bumps: 0,
                iterations: 0,
            };
        }

        let wrapper = unsafe { &mut *machine };
        let bumps = wrapper.machine.bumps;
        let iterations = wrapper.machine.iteration;
        let records = wrapper.machine.profiler.records();

        let summary = if records.is_empty() {
            wrapper.profile_cache.summary_from_cache()
        } else {
            let summary = wrapper.profile_cache.aggregate_from(
                records.iter(),
                bucket_size,
                bumps,
                iterations,
            );
            wrapper.machine.profiler.reset();
            summary
        };

        summary
    });

    result.unwrap_or(RivalProfileSummary {
        entries: ptr::null(),
        len: 0,
        bumps: 0,
        iterations: 0,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_profiler_executions(
    machine: *mut RivalMachine,
    out_len: *mut usize,
) -> *const RivalExecution {
    let result = panic::catch_unwind(|| {
        if machine.is_null() || out_len.is_null() {
            if !out_len.is_null() {
                unsafe { *out_len = 0 };
            }
            return ptr::null();
        }

        let wrapper = unsafe { &mut *machine };
        let records = wrapper.machine.profiler.records();

        if !records.is_empty() {
            wrapper.profile_cache.cache_executions(records.iter());
            wrapper.machine.profiler.reset();
        } else {
            wrapper.profile_cache.executions.clear();
        }

        unsafe { *out_len = wrapper.profile_cache.executions.len() };
        if wrapper.profile_cache.executions.is_empty() {
            ptr::null()
        } else {
            wrapper.profile_cache.executions.as_ptr()
        }
    });

    result.unwrap_or_else(|_| {
        if !out_len.is_null() {
            unsafe { *out_len = 0 };
        }
        ptr::null()
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_instruction_names(
    machine: *mut RivalMachine,
    out_len: *mut usize,
) -> *const u8 {
    let result = panic::catch_unwind(|| {
        if machine.is_null() || out_len.is_null() {
            if !out_len.is_null() {
                unsafe { *out_len = 0 };
            }
            return ptr::null();
        }

        let wrapper = unsafe { &mut *machine };

        if wrapper.instruction_names_cache.is_empty() {
            let names: Vec<&str> = wrapper
                .machine
                .instructions
                .iter()
                .map(|instr| instr.data.name_static())
                .collect();
            wrapper.instruction_names_cache = names.join("\0").into_bytes();
        }

        unsafe { *out_len = wrapper.instruction_names_cache.len() };
        wrapper.instruction_names_cache.as_ptr()
    });

    result.unwrap_or_else(|_| {
        if !out_len.is_null() {
            unsafe { *out_len = 0 };
        }
        ptr::null()
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_iterations(machine: *const RivalMachine) -> u32 {
    if machine.is_null() {
        0
    } else {
        unsafe { (*machine).machine.iteration as u32 }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_bumps(machine: *const RivalMachine) -> u32 {
    if machine.is_null() {
        0
    } else {
        unsafe { (*machine).machine.bumps as u32 }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_set_profiling(
    machine: *mut RivalMachine,
    mode: RivalProfilingMode,
) {
    if !machine.is_null() {
        let wrapper = unsafe { &mut *machine };
        wrapper.machine.profiling_enabled = mode == RivalProfilingMode::On;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_get_profiling(
    machine: *const RivalMachine,
) -> RivalProfilingMode {
    if machine.is_null() {
        RivalProfilingMode::Off
    } else {
        let wrapper = unsafe { &*machine };
        if wrapper.machine.profiling_enabled {
            RivalProfilingMode::On
        } else {
            RivalProfilingMode::Off
        }
    }
}
