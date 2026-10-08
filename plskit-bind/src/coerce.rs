//! Coercion shared by top-level arguments, `args` records and model
//! records. Each function returns the reason as a `String`; the caller
//! picks the error code and prefixes the argument name.
//!
//! The rules: an integer parameter takes
//! `I64`, `U64`, or an `F64` with no fractional part; a seed also takes a
//! decimal string; a scalar widens to a 1-element vector where a vector
//! is expected, because R has no scalar / length-1 distinction.

use crate::value::{MatF64, Value, VecF64};

/// 2^53: every integer up to here is exactly representable as an `f64`.
const F64_EXACT_MAX: f64 = 9_007_199_254_740_992.0;

/// A short description of a value for error messages.
pub(crate) fn describe(v: &Value<'_>) -> String {
    match v {
        Value::Null => "null".to_owned(),
        Value::Bool(b) => b.to_string(),
        Value::I64(n) => n.to_string(),
        Value::U64(n) => n.to_string(),
        Value::F64(x) => x.to_string(),
        Value::Str(s) => format!("the string {s:?}"),
        other => {
            let kind = other.kind_name();
            let article = if kind.starts_with(['a', 'e', 'i', 'o', 'u']) {
                "an"
            } else {
                "a"
            };
            format!("{article} {kind}")
        }
    }
}

fn whole_f64(x: f64) -> bool {
    x.is_finite() && x.fract() == 0.0 && (0.0..=F64_EXACT_MAX).contains(&x)
}

/// A count: non-negative and whole.
pub(crate) fn to_usize(v: &Value<'_>) -> Result<usize, String> {
    let bad = || format!("must be a non-negative whole number, got {}", describe(v));
    match *v {
        Value::I64(n) => usize::try_from(n).map_err(|_| bad()),
        Value::U64(n) => i64::try_from(n)
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(bad),
        Value::F64(x) if whole_f64(x) => Ok(x as usize),
        _ => Err(bad()),
    }
}

/// A seed: any `u64`. A float must be whole and at most 2^53; larger
/// seeds travel as decimal strings (R's form).
pub(crate) fn to_seed(v: &Value<'_>) -> Result<u64, String> {
    let bad = || {
        format!(
            "must be a whole number in [0, 2^64) (a decimal string above 2^53), got {}",
            describe(v)
        )
    };
    match v {
        Value::U64(n) => Ok(*n),
        Value::I64(n) => u64::try_from(*n).map_err(|_| bad()),
        Value::F64(x) if whole_f64(*x) => Ok(*x as u64),
        Value::Str(s) if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) => {
            s.parse::<u64>().map_err(|_| bad())
        }
        _ => Err(bad()),
    }
}

/// A float; integers convert.
pub(crate) fn to_f64(v: &Value<'_>) -> Result<f64, String> {
    match *v {
        Value::F64(x) => Ok(x),
        Value::I64(n) => Ok(n as f64),
        Value::U64(n) => Ok(n as f64),
        _ => Err(format!("must be a number, got {}", describe(v))),
    }
}

/// A bool; nothing converts.
pub(crate) fn to_bool(v: &Value<'_>) -> Result<bool, String> {
    match *v {
        Value::Bool(b) => Ok(b),
        _ => Err(format!("must be a bool, got {}", describe(v))),
    }
}

/// A string.
pub(crate) fn to_str<'v>(v: &'v Value<'_>) -> Result<&'v str, String> {
    match v {
        Value::Str(s) => Ok(s),
        _ => Err(format!("must be a string, got {}", describe(v))),
    }
}

/// A bool as a number: `true` is 1, `false` is 0 (numpy's cast).
fn bool_f64(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// A 1-D float array. A scalar widens to one element; an integer or bool
/// vector converts (inputs are f64; `true`/`false` become 1/0, as numpy
/// casts a bool array). A bool here is data, never a flag: flags are
/// read by [`to_bool`]. A seam hands bind no missing bool (R's `NA` in a
/// logical is rejected at the R seam or promoted to `NaN` before it).
pub(crate) fn to_vec(v: Value<'_>) -> Result<VecF64<'_>, String> {
    match v {
        Value::Vec(x) => Ok(x),
        Value::F64(x) => Ok(VecF64::Owned(vec![x])),
        Value::I64(n) => Ok(VecF64::Owned(vec![n as f64])),
        Value::U64(n) => Ok(VecF64::Owned(vec![n as f64])),
        Value::Bool(b) => Ok(VecF64::Owned(vec![bool_f64(b)])),
        Value::IntVec(xs) => Ok(VecF64::Owned(xs.iter().map(|&n| n as f64).collect())),
        Value::BoolVec(bs) => Ok(VecF64::Owned(bs.into_iter().map(bool_f64).collect())),
        Value::Mat(_) => Err("must be 1-D, got 2-D".to_owned()),
        other => Err(format!(
            "must be a numeric vector, got {}",
            describe(&other)
        )),
    }
}

/// A 2-D float array.
pub(crate) fn to_mat(v: Value<'_>) -> Result<MatF64<'_>, String> {
    match v {
        Value::Mat(m) => Ok(m),
        Value::Vec(_) | Value::IntVec(_) | Value::BoolVec(_) => {
            Err("must be 2-D, got 1-D".to_owned())
        }
        Value::F64(_) | Value::I64(_) | Value::U64(_) | Value::Bool(_) => {
            Err("must be 2-D, got 0-D".to_owned())
        }
        other => Err(format!(
            "must be a numeric matrix, got {}",
            describe(&other)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_numbers() {
        assert_eq!(to_usize(&Value::I64(3)), Ok(3));
        assert_eq!(to_usize(&Value::U64(3)), Ok(3));
        assert_eq!(to_usize(&Value::F64(3.0)), Ok(3));
        for bad in [
            Value::F64(3.5),
            Value::F64(-1.0),
            Value::I64(-1),
            Value::U64(i64::MAX as u64 + 1),
            Value::F64(f64::NAN),
            Value::F64(1e300),
            Value::Bool(true),
            Value::text("3"),
        ] {
            assert!(to_usize(&bad).is_err(), "{bad:?} accepted");
        }
        assert_eq!(
            to_usize(&Value::F64(2.5)).unwrap_err(),
            "must be a non-negative whole number, got 2.5"
        );
    }

    #[test]
    fn seeds() {
        assert_eq!(to_seed(&Value::U64(u64::MAX)), Ok(u64::MAX));
        assert_eq!(to_seed(&Value::text("18446744073709551615")), Ok(u64::MAX));
        assert_eq!(to_seed(&Value::text("42")), Ok(42));
        assert_eq!(to_seed(&Value::I64(42)), Ok(42));
        assert_eq!(to_seed(&Value::F64(9_007_199_254_740_992.0)), Ok(1 << 53));
        for bad in [
            Value::text("18446744073709551616"),
            Value::text("-1"),
            Value::text("+1"),
            Value::text("1e5"),
            Value::text(""),
            Value::I64(-5),
            Value::F64(9_007_199_254_740_994.0),
            Value::F64(0.5),
        ] {
            assert!(to_seed(&bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn scalars_widen_to_vectors() {
        let v = to_vec(Value::F64(2.0)).unwrap();
        assert_eq!(v.as_slice(), &[2.0]);
        let v = to_vec(Value::IntVec(vec![1, 2])).unwrap();
        assert_eq!(v.as_slice(), &[1.0, 2.0]);
        let m = Value::Mat(MatF64::Owned {
            data: vec![0.0; 4],
            nrows: 2,
            ncols: 2,
        });
        assert_eq!(to_vec(m).unwrap_err(), "must be 1-D, got 2-D");
    }

    #[test]
    fn bools_convert_to_vectors_and_matrices_must_be_2d() {
        let v = to_vec(Value::BoolVec(vec![true, false, true])).unwrap();
        assert_eq!(v.as_slice(), &[1.0, 0.0, 1.0]);
        assert_eq!(to_vec(Value::Bool(true)).unwrap().as_slice(), &[1.0]);
        assert_eq!(to_vec(Value::Bool(false)).unwrap().as_slice(), &[0.0]);
        assert_eq!(
            to_mat(Value::BoolVec(vec![true])).unwrap_err(),
            "must be 2-D, got 1-D"
        );
        assert_eq!(
            to_mat(Value::Bool(true)).unwrap_err(),
            "must be 2-D, got 0-D"
        );
        let v = Value::Vec(VecF64::Owned(vec![1.0]));
        assert_eq!(to_mat(v).unwrap_err(), "must be 2-D, got 1-D");
        assert_eq!(to_mat(Value::F64(1.0)).unwrap_err(), "must be 2-D, got 0-D");
    }
}
