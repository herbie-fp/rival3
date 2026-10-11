use rival::Discretization;
use rug::Float;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RivalDiscType {
    Bool = 0,
    F32 = 1,
    F64 = 2,
}

impl RivalDiscType {
    pub(crate) fn from_abi(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::Bool,
            1 => Self::F32,
            2 => Self::F64,
            _ => return None,
        })
    }
}

#[derive(Clone)]
pub struct RivalDiscretization {
    precision: u32,
    types: Vec<RivalDiscType>,
}

impl RivalDiscretization {
    pub(crate) fn new(types: Vec<RivalDiscType>, precision: u32) -> Self {
        Self { precision, types }
    }
}

impl Discretization for RivalDiscretization {
    fn target(&self) -> u32 {
        self.precision
    }

    fn convert(&self, idx: usize, v: &Float) -> Float {
        match self.types[idx] {
            RivalDiscType::F64 => Float::with_val(f64::MANTISSA_DIGITS, v.to_f64()),
            RivalDiscType::F32 => Float::with_val(f32::MANTISSA_DIGITS, v.to_f32()),
            RivalDiscType::Bool => v.clone(),
        }
    }

    fn distance(&self, idx: usize, lo: &Float, hi: &Float) -> usize {
        match self.types[idx] {
            RivalDiscType::Bool => {
                if lo == hi {
                    0
                } else {
                    2
                }
            }
            RivalDiscType::F32 => ordinal_distance_f32(lo.to_f32(), hi.to_f32()),
            RivalDiscType::F64 => ordinal_distance_f64(lo.to_f64(), hi.to_f64()),
        }
    }
}

fn ordinal_distance_f32(x: f32, y: f32) -> usize {
    if x == y {
        return 0;
    }
    let to_ordinal = |v: f32| -> i32 {
        if v == 0.0 {
            return 0;
        }
        let bits = v.to_bits() as i32;
        if bits < 0 { !bits } else { bits }
    };
    to_ordinal(y).wrapping_sub(to_ordinal(x)).unsigned_abs() as usize
}

fn ordinal_distance_f64(x: f64, y: f64) -> usize {
    if x == y {
        return 0;
    }
    let to_ordinal = |v: f64| -> i64 {
        if v == 0.0 {
            return 0;
        }
        let bits = v.to_bits() as i64;
        if bits < 0 { !bits } else { bits }
    };
    to_ordinal(y).wrapping_sub(to_ordinal(x)).unsigned_abs() as usize
}
