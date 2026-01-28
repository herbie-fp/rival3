use rival::eval::machine::Hint;

pub struct RivalHints {
    pub(crate) hints: Vec<Hint>,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_hints_free(hints: *mut RivalHints) {
    if !hints.is_null() {
        unsafe { drop(Box::from_raw(hints)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_hints_len(hints: *const RivalHints) -> usize {
    if hints.is_null() {
        0
    } else {
        unsafe { (*hints).hints.len() }
    }
}
