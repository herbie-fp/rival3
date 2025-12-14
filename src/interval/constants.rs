use super::value::Ival;
use crate::mpfr::{mpfr_e, mpfr_pi};
use rug::float::Round;

impl Ival {
    pub fn set_pi(&mut self) {
        mpfr_pi(self.lo.as_float_mut(), Round::Down);
        mpfr_pi(self.hi.as_float_mut(), Round::Up);
        self.lo.immovable = false;
        self.hi.immovable = false;
    }

    pub fn set_e(&mut self) {
        mpfr_e(self.lo.as_float_mut(), Round::Down);
        mpfr_e(self.hi.as_float_mut(), Round::Up);
        self.lo.immovable = false;
        self.hi.immovable = false;
    }
}
