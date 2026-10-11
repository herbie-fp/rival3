use crate::{RivalError, status};
use rival::{Discretization, Execution, MachineBuilder, Strategy};
use rug::float::{prec_max, prec_min};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RivalStrategy {
    Adaptive = 0,
    Baseline = 1,
}

impl RivalStrategy {
    fn from_abi(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::Adaptive,
            1 => Self::Baseline,
            _ => return None,
        })
    }
}

pub struct RivalMachineOptions {
    target_precision: u32,
    max_precision: Option<u32>,
    profile_capacity: Option<usize>,
    strategy: Option<Strategy>,
}

impl Default for RivalMachineOptions {
    fn default() -> Self {
        Self {
            target_precision: f64::MANTISSA_DIGITS,
            max_precision: None,
            profile_capacity: None,
            strategy: None,
        }
    }
}

impl RivalMachineOptions {
    pub(crate) fn target_precision(&self) -> u32 {
        self.target_precision
    }

    pub(crate) fn configure<D: Discretization>(
        &self,
        mut builder: MachineBuilder<D>,
    ) -> MachineBuilder<D> {
        if let Some(strategy) = self.strategy {
            builder = builder.strategy(strategy);
        }
        if let Some(max_precision) = self.max_precision {
            builder = builder.max_precision(max_precision);
        }
        if let Some(profile_capacity) = self.profile_capacity {
            builder = builder.profile_capacity(profile_capacity);
        }
        builder
    }
}

fn checked_precision(precision: u32) -> Result<u32, RivalError> {
    (prec_min()..=prec_max())
        .contains(&precision)
        .then_some(precision)
        .ok_or(RivalError::Misuse)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_options_new(
    out: *mut *mut RivalMachineOptions,
) -> RivalError {
    status(|| {
        let out = unsafe { out.as_mut() }.ok_or(RivalError::Misuse)?;
        *out = Box::into_raw(Box::new(RivalMachineOptions::default()));
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_options_free(options: *mut RivalMachineOptions) {
    if !options.is_null() {
        unsafe { drop(Box::from_raw(options)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_options_set_target_precision(
    options: *mut RivalMachineOptions,
    precision: u32,
) -> RivalError {
    status(|| {
        let options = unsafe { options.as_mut() }.ok_or(RivalError::Misuse)?;
        options.target_precision = checked_precision(precision)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_options_set_max_precision(
    options: *mut RivalMachineOptions,
    precision: u32,
) -> RivalError {
    status(|| {
        let options = unsafe { options.as_mut() }.ok_or(RivalError::Misuse)?;
        options.max_precision = Some(checked_precision(precision)?);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_options_set_profile_capacity(
    options: *mut RivalMachineOptions,
    capacity: usize,
) -> RivalError {
    status(|| {
        let options = unsafe { options.as_mut() }.ok_or(RivalError::Misuse)?;
        if capacity > isize::MAX as usize / size_of::<Execution>() {
            return Err(RivalError::Misuse);
        }
        options.profile_capacity = Some(capacity);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_options_set_strategy(
    options: *mut RivalMachineOptions,
    strategy: u32,
) -> RivalError {
    status(|| {
        let options = unsafe { options.as_mut() }.ok_or(RivalError::Misuse)?;
        options.strategy = Some(
            match RivalStrategy::from_abi(strategy).ok_or(RivalError::Misuse)? {
                RivalStrategy::Adaptive => Strategy::Adaptive,
                RivalStrategy::Baseline => Strategy::Baseline,
            },
        );
        Ok(())
    })
}
