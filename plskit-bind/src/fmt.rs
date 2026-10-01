//! Python-compatible renderings, so a message built here reads the same
//! as the one the Python wrapper builds.

/// Python's `format(x, '.4g')`.
pub(crate) fn g4(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    // Round to 4 significant digits first: the notation depends on the
    // exponent of the rounded value (9999.5 -> 1.000e4 -> "1e+04").
    let sci = format!("{x:.3e}");
    let (mantissa, exp) = sci.split_once('e').expect("e-notation");
    let exp: i32 = exp.parse().expect("integer exponent");
    if (-4..4).contains(&exp) {
        let decimals = usize::try_from(3 - exp).unwrap_or(0);
        trim_zeros(&format!("{x:.decimals$}"))
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim_zeros(mantissa), exp.abs())
    }
}

fn trim_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        s.to_owned()
    }
}

/// Python's `repr` of a plain string: `'s'`.
pub(crate) fn py_repr(s: &str) -> String {
    format!("'{s}'")
}

/// Python's `repr` of a list of plain strings: `['a', 'b']`.
pub(crate) fn py_list(items: &[&str]) -> String {
    let inner: Vec<String> = items.iter().map(|s| py_repr(s)).collect();
    format!("[{}]", inner.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g4_matches_python() {
        let cases = [
            (12.3456, "12.35"),
            (0.000_012_34, "1.234e-05"),
            (123_456.0, "1.235e+05"),
            (50.0, "50"),
            (0.0001, "0.0001"),
            (9999.5, "1e+04"),
            (3.0, "3"),
            (0.0, "0"),
            (-0.0, "-0"),
            (f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
            (-2.5e-7, "-2.5e-07"),
            (1e16, "1e+16"),
            (7.25, "7.25"),
            (0.12345, "0.1235"),
        ];
        for (x, want) in cases {
            assert_eq!(g4(x), want, "g4({x})");
        }
    }

    #[test]
    fn python_reprs() {
        assert_eq!(py_repr("optimal"), "'optimal'");
        assert_eq!(py_list(&["selector", "args"]), "['selector', 'args']");
    }
}
