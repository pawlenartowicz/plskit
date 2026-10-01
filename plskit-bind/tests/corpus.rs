//! Drive every `testdata/` case through `plskit_bind::call` and compare the
//! returned records with the frozen outputs at the corpus tolerances
//! (`|a − e| ≤ atol + rtol·|e|`, scalars atol 1e-12, arrays atol 1e-10,
//! rtol 1e-14; integers and strings exact).
//!
//! Each case makes the call `plskit-py/tests/test_corpus.py` makes. A
//! fixture key maps onto the result by name: exact, then ASCII
//! case-insensitive (`y_std` is `Y_std`, `w_rot` is `W_rot`), then into
//! nested records (`ci`, `spec`), recursively. Two encodings cover non-array
//! fields: `{field}__keys` / `{field}__values` for integer-keyed maps, and
//! `{field}_{point|lower|upper|sd}` (`{field}_k{i}_{part}` for list items)
//! for `CIScalar`s. Every fixture key must resolve and match, and every
//! returned record must pass `check_record`.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use plskit_bind::{call, check_record, registry, MatF64, Record, Value, VecF64};
use serde_json::Value as Json;

const ATOL_SCALAR: f64 = 1e-12;
const ATOL_ARRAY: f64 = 1e-10;
const RTOL: f64 = 1e-14;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("testdata")
}

fn manifest_cases() -> Vec<Json> {
    let text =
        fs::read_to_string(corpus_dir().join("manifest.json")).expect("testdata/manifest.json");
    let m: Json = serde_json::from_str(&text).unwrap();
    m["cases"].as_array().unwrap().clone()
}

enum Stored {
    F(ndarray::ArrayD<f64>),
    I(ndarray::ArrayD<i64>),
    S(String),
}

/// Every entry of an npz, by dtype: f64 arrays, i64 counts / flags / map
/// keys, and strings stored as 1-D u8 (`plskit-testdata-gen/src/npz.rs`).
fn read_npz(path: &Path) -> HashMap<String, Stored> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(bytes)).unwrap();
    let names: Vec<String> = npz.names().unwrap();
    let mut out = HashMap::new();
    for n in names {
        let key = n.trim_end_matches(".npy").to_owned();
        if let Ok(a) = npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(&n) {
            out.insert(key, Stored::F(a));
        } else if let Ok(a) = npz.by_name::<ndarray::OwnedRepr<i64>, ndarray::IxDyn>(&n) {
            out.insert(key, Stored::I(a));
        } else if let Ok(a) = npz.by_name::<ndarray::OwnedRepr<u8>, ndarray::IxDyn>(&n) {
            out.insert(
                key,
                Stored::S(String::from_utf8(a.iter().copied().collect()).unwrap()),
            );
        } else {
            panic!("{}: entry {n} has an unsupported dtype", path.display());
        }
    }
    out
}

fn to_value(a: &ndarray::ArrayD<f64>) -> Value<'static> {
    match a.ndim() {
        1 => Value::Vec(VecF64::Owned(a.iter().copied().collect())),
        2 => {
            let (n, d) = (a.shape()[0], a.shape()[1]);
            let mut data = Vec::with_capacity(n * d);
            for j in 0..d {
                for i in 0..n {
                    data.push(a[[i, j]]);
                }
            }
            Value::Mat(MatF64::Owned {
                data,
                nrows: n,
                ncols: d,
            })
        }
        k => panic!("{k}-D input array"),
    }
}

fn json_to_value(v: &Json) -> Value<'static> {
    match v {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Number(n) => n
            .as_i64()
            .map_or_else(|| Value::F64(n.as_f64().unwrap()), Value::I64),
        Json::String(s) => Value::Str(Cow::Owned(s.clone())),
        Json::Object(m) => {
            let mut r = Record::new();
            for (k, val) in m {
                r.push(k, json_to_value(val)).unwrap();
            }
            Value::Record(r)
        }
        Json::Array(_) => panic!("array-valued kwargs are not used by the corpus"),
    }
}

/// Manifest kwargs that describe the generated data, not the call.
fn describes_data(function: &str, key: &str) -> bool {
    matches!(
        key,
        "d" | "n" | "n_new" | "n_train" | "seed_new" | "seed_train"
    ) || (key == "seed" && matches!(function, "spls1_fit" | "pls3_fit"))
}

fn weights(case: &Json, inputs: &HashMap<String, Stored>) -> Option<Value<'static>> {
    let descriptor = case["kwargs"].get("weights").and_then(Json::as_str) == Some("nonuniform");
    let array = inputs.contains_key("weights");
    assert_eq!(
        descriptor, array,
        "{}: weights descriptor and NPZ array disagree",
        case["name"]
    );
    match inputs.get("weights") {
        Some(Stored::F(a)) => Some(to_value(a)),
        _ => None,
    }
}

fn invoke(function: &str, fields: Vec<(&str, Value<'static>)>) -> Value<'static> {
    let mut r = Record::new();
    for (k, v) in fields {
        r.push(k, v).unwrap();
    }
    let out =
        call(function, r).unwrap_or_else(|e| panic!("{function}: {} ({})", e.message, e.code));
    if let Value::Record(rec) = &out.result {
        check_record(rec).unwrap_or_else(|e| panic!("{function}: {e}"));
    }
    out.result
}

fn into_record(v: Value<'static>) -> Record<'static> {
    match v {
        Value::Record(r) => r,
        other => panic!("expected a record, got {other:?}"),
    }
}

/// The record a case's fixture is compared against.
fn run_case(case: &Json, inputs: &HashMap<String, Stored>) -> Record<'static> {
    let function = case["function"].as_str().unwrap();
    let kw = case["kwargs"].as_object().unwrap();
    let arr = |name: &str| match inputs.get(name) {
        Some(Stored::F(a)) => to_value(a),
        _ => panic!("{}: input {name} missing", case["name"]),
    };
    let kwv = |key: &str| json_to_value(&kw[key]);
    match function {
        "pls1_predict" => {
            let fit = invoke(
                "pls1_fit",
                vec![
                    ("X", arr("X_train")),
                    ("y", arr("y_train")),
                    ("k", kwv("k")),
                ],
            );
            let y_pred = invoke(
                "pls1_predict",
                vec![("model", fit.clone()), ("X_new", arr("X_new"))],
            );
            let mut r = into_record(fit);
            r.push("y_pred", y_pred).unwrap();
            r
        }
        "rotate" => {
            let fit = into_record(invoke(
                "pls1_fit",
                vec![("X", arr("X")), ("y", arr("y")), ("k", kwv("k"))],
            ));
            let w = fit.get("W").unwrap().clone();
            into_record(invoke(
                "rotate",
                vec![("model_or_W", w), ("method", kwv("method"))],
            ))
        }
        "preprocess" => {
            let mut fields = vec![("X", arr("X")), ("Y", arr("y"))];
            if let Some(w) = weights(case, inputs) {
                fields.push(("weights", w));
            }
            into_record(invoke("preprocess", fields))
        }
        "pls3_transform" => {
            let model = invoke(
                "pls3_fit",
                vec![("X", arr("X")), ("Y", arr("Y")), ("k", kwv("k"))],
            );
            into_record(invoke(
                "pls3_transform",
                vec![
                    ("model", model),
                    ("X_new", arr("X_new")),
                    ("Y_new", arr("Y_new")),
                    ("which", kwv("which")),
                ],
            ))
        }
        _ => {
            let mut fields: Vec<(&str, Value<'static>)> = vec![("X", arr("X"))];
            if inputs.contains_key("y") {
                fields.push(("y", arr("y")));
            }
            if inputs.contains_key("Y") {
                fields.push(("Y", arr("Y")));
            }
            if let Some(w) = weights(case, inputs) {
                fields.push(("weights", w));
            }
            for (k, v) in kw {
                if k != "weights" && !describes_data(function, k) {
                    fields.push((k.as_str(), json_to_value(v)));
                }
            }
            into_record(invoke(function, fields))
        }
    }
}

enum Found<'r> {
    Val(&'r Value<'static>),
    MapKeys(&'r [(i64, f64)]),
    MapValues(&'r [(i64, f64)]),
}

fn lookup<'r>(rec: &'r Record<'static>, key: &str) -> Option<&'r Value<'static>> {
    if let Some(v) = rec.get(key) {
        return Some(v);
    }
    if let Some((_, v)) = rec.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)) {
        return Some(v);
    }
    rec.iter().find_map(|(_, v)| match v {
        Value::Record(inner) => lookup(inner, key),
        _ => None,
    })
}

fn resolve<'r>(rec: &'r Record<'static>, key: &str) -> Option<Found<'r>> {
    if let Some(base) = key.strip_suffix("__keys") {
        return match lookup(rec, base)? {
            Value::IntMap(m) => Some(Found::MapKeys(m)),
            _ => None,
        };
    }
    if let Some(base) = key.strip_suffix("__values") {
        return match lookup(rec, base)? {
            Value::IntMap(m) => Some(Found::MapValues(m)),
            _ => None,
        };
    }
    if let Some(v) = lookup(rec, key) {
        return Some(Found::Val(v));
    }
    let (base, part) = key.rsplit_once('_')?;
    if !matches!(part, "point" | "lower" | "upper" | "sd") {
        return None;
    }
    let ci = match lookup(rec, base) {
        Some(Value::Record(r)) => r,
        Some(_) => return None,
        None => {
            let (list, idx) = base.rsplit_once("_k")?;
            let i: usize = idx.parse().ok()?;
            match lookup(rec, list)? {
                Value::List(items) => match items.get(i)? {
                    Value::Record(r) => r,
                    _ => return None,
                },
                _ => return None,
            }
        }
    };
    ci.get(part).map(Found::Val)
}

#[allow(clippy::float_cmp)]
fn close(a: f64, e: f64, atol: f64) -> bool {
    a == e
        || (a.is_nan() && e.is_nan())
        || (e.is_finite() && (a - e).abs() <= atol + RTOL * e.abs())
}

fn floats_close(got: &[f64], want: &ndarray::ArrayD<f64>) -> Result<(), String> {
    let want: Vec<f64> = want.iter().copied().collect();
    if got.len() != want.len() {
        return Err(format!("length {} vs {}", got.len(), want.len()));
    }
    match got
        .iter()
        .zip(&want)
        .position(|(a, e)| !close(*a, *e, ATOL_ARRAY))
    {
        None => Ok(()),
        Some(i) => Err(format!("[{i}] {} vs {}", got[i], want[i])),
    }
}

fn ints_equal(got: &[i64], want: &ndarray::ArrayD<i64>) -> Result<(), String> {
    let want: Vec<i64> = want.iter().copied().collect();
    if got == want.as_slice() {
        Ok(())
    } else {
        Err(format!("{got:?} vs {want:?}"))
    }
}

fn scalar_i64(want: &ndarray::ArrayD<i64>) -> i64 {
    assert_eq!(want.ndim(), 0, "expected a 0-D integer fixture");
    *want.iter().next().unwrap()
}

fn compare(found: &Found<'_>, want: &Stored) -> Result<(), String> {
    match (found, want) {
        (Found::Val(Value::F64(a)), Stored::F(e)) if e.ndim() == 0 => {
            let e = *e.iter().next().unwrap();
            if close(*a, e, ATOL_SCALAR) {
                Ok(())
            } else {
                Err(format!("{a} vs {e}"))
            }
        }
        (Found::Val(Value::Vec(v)), Stored::F(e)) if e.ndim() == 1 => floats_close(v.as_slice(), e),
        (Found::Val(Value::Mat(m)), Stored::F(e)) if e.ndim() == 2 => {
            if (m.nrows(), m.ncols()) != (e.shape()[0], e.shape()[1]) {
                return Err(format!(
                    "shape {}x{} vs {:?}",
                    m.nrows(),
                    m.ncols(),
                    e.shape()
                ));
            }
            let mv = m.as_mat();
            for i in 0..m.nrows() {
                for j in 0..m.ncols() {
                    if !close(mv[(i, j)], e[[i, j]], ATOL_ARRAY) {
                        return Err(format!("[{i},{j}] {} vs {}", mv[(i, j)], e[[i, j]]));
                    }
                }
            }
            Ok(())
        }
        (Found::MapValues(m), Stored::F(e)) => {
            floats_close(&m.iter().map(|p| p.1).collect::<Vec<_>>(), e)
        }
        (Found::MapKeys(m), Stored::I(e)) => {
            ints_equal(&m.iter().map(|p| p.0).collect::<Vec<_>>(), e)
        }
        (Found::Val(Value::I64(a)), Stored::I(e)) => (*a == scalar_i64(e))
            .then_some(())
            .ok_or(format!("{a} vs {e}")),
        (Found::Val(Value::U64(a)), Stored::I(e)) => (i64::try_from(*a).ok()
            == Some(scalar_i64(e)))
        .then_some(())
        .ok_or(format!("{a} vs {e}")),
        (Found::Val(Value::Bool(b)), Stored::I(e)) => (i64::from(*b) == scalar_i64(e))
            .then_some(())
            .ok_or(format!("{b} vs {e}")),
        (Found::Val(Value::IntVec(v)), Stored::I(e)) => ints_equal(v, e),
        (Found::Val(Value::BoolVec(v)), Stored::I(e)) => {
            ints_equal(&v.iter().map(|&b| i64::from(b)).collect::<Vec<_>>(), e)
        }
        (Found::Val(Value::Str(s)), Stored::S(e)) => {
            (s == e).then_some(()).ok_or(format!("{s:?} vs {e:?}"))
        }
        (Found::Val(v), _) => Err(format!(
            "result holds a {}, fixture a different kind",
            v.kind_name()
        )),
        _ => Err("fixture kind does not match the map encoding".to_owned()),
    }
}

#[test]
fn every_manifest_function_is_registered() {
    let registered: BTreeSet<&str> = registry().iter().map(|f| f.name).collect();
    for case in manifest_cases() {
        let f = case["function"].as_str().unwrap();
        assert!(
            registered.contains(f),
            "manifest function {f} is not registered"
        );
    }
}

#[test]
fn every_corpus_case_matches_through_call() {
    faer::disable_global_parallelism();
    let cases = manifest_cases();
    assert!(!cases.is_empty(), "empty manifest");
    let mut failures = Vec::new();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let inputs = read_npz(&corpus_dir().join(case["inputs"].as_str().unwrap()));
        let expected = read_npz(&corpus_dir().join(case["outputs"].as_str().unwrap()));
        let actual = run_case(case, &inputs);
        let mut keys: Vec<&String> = expected.keys().collect();
        keys.sort();
        for key in keys {
            match resolve(&actual, key) {
                None => failures.push(format!("{name}.{key}: no matching result field")),
                Some(found) => {
                    if let Err(e) = compare(&found, &expected[key]) {
                        failures.push(format!("{name}.{key}: {e}"));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} corpus mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
