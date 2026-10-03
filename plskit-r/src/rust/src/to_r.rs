//! `plskit_bind` values to R: a tagged record becomes a named
//! list with class `c(<r_class>, "plskit_result")`, a seed (`U64`) a
//! decimal string, an integer-keyed map a named double vector.

// Integers beyond R's 32-bit range leave as doubles; the
// precision loss above 2^53 is accepted.
#![allow(clippy::cast_precision_loss)]

use extendr_api::prelude::*;
use plskit_bind::{result_type, result_types, BindError, Outcome, Record, Value};

fn named_list(names: Vec<String>, values: Vec<Robj>) -> Robj {
    // Only fails when the lengths differ, which every caller rules out by
    // construction; a mismatch here is a seam bug, not a bad input, so it
    // panics (caught by `catch_unwind` in lib.rs) rather than silently
    // returning NULL.
    Robj::from(
        List::from_names_and_values(names, values)
            .expect("named_list: names and values must have the same length"),
    )
}

fn int_or_double(n: i64) -> Robj {
    match i32::try_from(n) {
        Ok(v) if v != i32::MIN => Robj::from(v),
        _ => Robj::from(n as f64),
    }
}

fn record(r: Record<'_>) -> Robj {
    let class = r.type_name().and_then(result_type).map(|t| t.r_class);
    let (names, values): (Vec<String>, Vec<Robj>) = r
        .into_fields()
        .into_iter()
        .map(|(k, v)| (k, value(v)))
        .unzip();
    let mut out = named_list(names, values);
    if let Some(c) = class {
        out.set_class([c, "plskit_result"])
            .expect("record: setting a class on a freshly built list cannot fail");
    }
    out
}

/// Convert one value.
pub(crate) fn value(v: Value<'_>) -> Robj {
    match v {
        Value::Null => Robj::from(()),
        Value::Bool(b) => Robj::from(b),
        Value::I64(n) => int_or_double(n),
        Value::U64(n) => Robj::from(n.to_string()),
        Value::F64(x) => Robj::from(x),
        Value::Str(s) => Robj::from(s.as_ref()),
        Value::Vec(v) => Robj::from(v.as_slice()),
        Value::Mat(m) => {
            let (nrow, ncol) = (m.nrows(), m.ncols());
            let mut out = Robj::from(m.col_major().as_ref());
            let dim = [
                i32::try_from(nrow).expect("mat: row count fits in R's 32-bit dim"),
                i32::try_from(ncol).expect("mat: column count fits in R's 32-bit dim"),
            ];
            out.set_attrib("dim", dim)
                .expect("mat: setting dim on a freshly built vector cannot fail");
            out
        }
        Value::IntMap(pairs) => {
            let mut out = Robj::from(pairs.iter().map(|p| p.1).collect::<Vec<f64>>());
            out.set_names(pairs.iter().map(|p| p.0.to_string()))
                .expect("int_map: setting names on a freshly built vector cannot fail");
            out
        }
        Value::IntVec(v) => {
            let small: Option<Vec<i32>> = v
                .iter()
                .map(|&n| i32::try_from(n).ok().filter(|&x| x != i32::MIN))
                .collect();
            match small {
                Some(ints) => Robj::from(ints),
                None => Robj::from(v.iter().map(|&n| n as f64).collect::<Vec<f64>>()),
            }
        }
        Value::BoolVec(v) => Robj::from(v),
        Value::Record(r) => record(r),
        Value::List(items) => {
            let values: Vec<Robj> = items.into_iter().map(value).collect();
            List::from_values(values).into()
        }
    }
}

/// `list(ok = TRUE, result = <value>, warnings = list(<named list>, ...))`.
pub(crate) fn outcome(o: Outcome) -> Robj {
    let warnings: Vec<Robj> = o.warnings.into_iter().map(record).collect();
    named_list(
        vec!["ok".into(), "result".into(), "warnings".into()],
        vec![
            Robj::from(true),
            value(o.result),
            List::from_values(warnings).into(),
        ],
    )
}

/// `list(ok = FALSE, code, message, details = <named list>)`.
pub(crate) fn error(e: BindError) -> Robj {
    named_list(
        vec![
            "ok".into(),
            "code".into(),
            "message".into(),
            "details".into(),
        ],
        vec![
            Robj::from(false),
            Robj::from(e.code),
            Robj::from(e.message.as_str()),
            record(e.details),
        ],
    )
}

/// The registry as nested named lists.
pub(crate) fn registry_list() -> Robj {
    let functions: Vec<Robj> = registry_fns();
    let types: Vec<Robj> = result_types()
        .iter()
        .map(|t| {
            let fields: Vec<Robj> = t
                .fields
                .iter()
                .map(|f| {
                    named_list(
                        vec!["name".into(), "type".into(), "nullable".into()],
                        vec![Robj::from(f.name), Robj::from(f.ty), Robj::from(f.nullable)],
                    )
                })
                .collect();
            named_list(
                vec!["name".into(), "r_class".into(), "fields".into()],
                vec![
                    Robj::from(t.name),
                    Robj::from(t.r_class),
                    List::from_values(fields).into(),
                ],
            )
        })
        .collect();
    named_list(
        vec![
            "engine_version".into(),
            "functions".into(),
            "result_types".into(),
        ],
        vec![
            Robj::from(plskit_bind::engine_version()),
            List::from_values(functions).into(),
            List::from_values(types).into(),
        ],
    )
}

fn registry_fns() -> Vec<Robj> {
    plskit_bind::registry()
        .iter()
        .map(|f| {
            let params: Vec<Robj> = f
                .params
                .iter()
                .map(|p| {
                    named_list(
                        vec!["name".into(), "kind".into()],
                        vec![Robj::from(p.name), Robj::from(p.kind.as_str())],
                    )
                })
                .collect();
            named_list(
                vec!["name".into(), "result_types".into(), "params".into()],
                vec![
                    Robj::from(f.name),
                    Robj::from(f.result_types.to_vec()),
                    List::from_values(params).into(),
                ],
            )
        })
        .collect()
}
