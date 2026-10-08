//! The Python seam of plskit: Python values to `plskit_bind::Value` and
//! back, plus the two functions `_api.py` calls. Everything else (argument
//! validation, method dispatch, result records, error codes) lives in
//! `plskit-bind`.
//!
//! Input conversion runs in two passes. `collect` walks the Python value
//! and keeps a read guard for every array; `convert` then borrows array
//! data from those guards. `X` and `X_new` reach the engine without a copy
//! when C- or Fortran-contiguous; any other matrix does so only when
//! Fortran-contiguous, and a C-ordered one is copied column-major. The
//! guards live until `call` returns.

// The pyo3::create_exception! macro generates a struct without doc comments;
// the workspace `missing_docs = "warn"` lint can't be applied to macro output.
#![allow(missing_docs)]

use std::borrow::Cow;
use std::panic::{self, AssertUnwindSafe};

use faer::MatRef;
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use plskit_bind::{BindError, MatF64, Record, Value, VecF64};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyString, PyTuple, PyType};

pyo3::create_exception!(_plskit, PlsKitException, PyException);

/// A `PlsKitException` with `.code` set: `_api.py` picks the public error
/// class by it, so no raise site may leave it empty.
fn coded_err(py: Python<'_>, code: &str, message: String) -> PyErr {
    let err = PlsKitException::new_err(message);
    match err.value(py).setattr("code", code) {
        Ok(()) => err,
        Err(failed) => failed,
    }
}

/// `invalid_argument` for a value the seam cannot convert. `path` names
/// the value: `args['n_perm']`, `model.T`.
fn bad(py: Python<'_>, path: &str, why: &str) -> PyErr {
    coded_err(py, "invalid_argument", format!("{path} {why}"))
}

/// A `plskit-bind` error as a `PlsKitException`: `.code`, plus one
/// attribute per entry of `details` (`reason` for `invalid_weights`;
/// `skipped`, `total`, `skip_rate`, `threshold` for
/// `resampling_degenerate`).
fn bind_err<'py>(py: Python<'py>, e: BindError, make_result: &Bound<'py, PyAny>) -> PyErr {
    let err = coded_err(py, e.code, e.message);
    for (key, v) in e.details.into_fields() {
        let set = to_py(py, v, make_result).and_then(|v| err.value(py).setattr(key.as_str(), v));
        if let Err(failed) = set {
            return failed;
        }
    }
    err
}

fn type_name(v: &Bound<'_, PyAny>) -> String {
    v.get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |n| n.to_string())
}

/// The value as a message quotes it: a string as `the string "x"` (as
/// plskit-bind's `describe` does), anything else by its Python repr. An int too long for
/// Python to print has no repr and is named by its type.
fn describe(v: &Bound<'_, PyAny>) -> String {
    if let Ok(s) = v.extract::<String>() {
        return format!("the string {s:?}");
    }
    v.repr().map_or_else(
        |_| format!("a value of type {}", type_name(v)),
        |r| r.to_string(),
    )
}

// ── Python to Value ────────────────────────────────────────────────────

/// Nesting allowed inside one argument. A result nests four deep; the cap
/// turns a container that holds itself into an error, not a stack overflow.
/// `_MAX_DEPTH` in `_api.py` mirrors it — change together.
const MAX_DEPTH: usize = 32;

/// numpy dtype kinds cast to float64: bool, signed and unsigned int, float.
/// Mirrors `_REAL_KINDS` in `_api.py` — change together.
const REAL_KINDS: &[u8] = b"biuf";

/// A Python value with every container already walked.
enum Node<'py> {
    /// A value that borrows no array data.
    Leaf(Value<'static>),
    Vec(PyReadonlyArray1<'py, f64>),
    Mat(PyReadonlyArray2<'py, f64>),
    Record {
        type_name: Option<String>,
        fields: Vec<(String, Node<'py>)>,
    },
    List(Vec<Node<'py>>),
}

fn integer(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<Value<'static>> {
    if let Ok(n) = obj.extract::<i64>() {
        Ok(Value::I64(n))
    } else if let Ok(n) = obj.extract::<u64>() {
        Ok(Value::U64(n))
    } else {
        // Past 128 bits the message gives the size: a digit string can run
        // to thousands of characters, and Python refuses to print one past
        // 4300 digits.
        let got = match obj
            .call_method0("bit_length")
            .and_then(|b| b.extract::<u64>())
        {
            Ok(bits) if bits > 128 => format!("an integer of {bits} bits"),
            _ => describe(obj),
        };
        Err(bad(
            obj.py(),
            path,
            &format!("is outside the 64-bit integer range [-2^63, 2^64), got {got}"),
        ))
    }
}

/// `None`, `bool`, `int`, `float`, `str` and the numpy scalar types that
/// stand for them; `None` for any other value. `bool` is tested before
/// `int` because it is an `int` subclass.
fn scalar(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<Option<Value<'static>>> {
    if obj.is_none() {
        return Ok(Some(Value::Null));
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Some(Value::Bool(b.is_true())));
    }
    if obj.is_instance_of::<PyInt>() {
        return integer(obj, path).map(Some);
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(Some(Value::F64(obj.extract()?)));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Some(Value::Str(Cow::Owned(s.to_str()?.to_owned()))));
    }
    let np = obj.py().import("numpy")?;
    if obj.is_instance(&np.getattr("bool_")?)? {
        return Ok(Some(Value::Bool(obj.is_truthy()?)));
    }
    if obj.is_instance(&np.getattr("integer")?)? {
        return integer(&obj.call_method0("__index__")?, path).map(Some);
    }
    if obj.is_instance(&np.getattr("floating")?)? {
        return Ok(Some(Value::F64(obj.extract()?)));
    }
    Ok(None)
}

fn unsupported(obj: &Bound<'_, PyAny>, path: &str) -> PyErr {
    bad(
        obj.py(),
        path,
        &format!("has unsupported type '{}'", type_name(obj)),
    )
}

/// An `ndarray`: a 0-d one is its item; a 1-D or 2-D one is read as
/// float64. An array of another real dtype is cast (a copy), so that
/// plskit-bind, which knows what the parameter takes, words the refusal
/// where no array fits; an array of any other dtype is refused here. A
/// float64 array must be aligned, because Rust may not read `f64` data
/// through a slice over misaligned memory (a byte-offset view of a buffer
/// can be contiguous and still misaligned).
fn array<'py>(arr: &Bound<'py, PyUntypedArray>, path: &str) -> PyResult<Node<'py>> {
    let py = arr.py();
    let ndim = arr.ndim();
    if ndim == 0 {
        let item = arr.call_method0("item")?;
        return scalar(&item, path)?
            .map(Node::Leaf)
            .ok_or_else(|| unsupported(&item, path));
    }
    if ndim > 2 {
        return Err(bad(
            py,
            path,
            &format!("must be an array of at most 2 dimensions, got {ndim}-D"),
        ));
    }
    let dtype = arr.dtype();
    let float64 = numpy::dtype::<f64>(py);
    let cast;
    let arr = if dtype.is_equiv_to(&float64) {
        arr
    } else if REAL_KINDS.contains(&dtype.kind()) {
        cast = arr
            .call_method1("astype", (float64,))?
            .cast_into::<PyUntypedArray>()?;
        &cast
    } else {
        return Err(bad(
            py,
            path,
            &format!("has unsupported array dtype {dtype}"),
        ));
    };
    if !arr.is_aligned() {
        return Err(bad(
            py,
            path,
            "must be an aligned array, got a misaligned one",
        ));
    }
    if ndim == 1 {
        Ok(Node::Vec(arr.cast::<PyArray1<f64>>()?.try_readonly()?))
    } else {
        Ok(Node::Mat(arr.cast::<PyArray2<f64>>()?.try_readonly()?))
    }
}

/// A mapping: string keys make an untagged record, integer keys an
/// integer-keyed map of numbers, and an empty mapping an empty record.
fn mapping<'py>(map: &Bound<'py, PyMapping>, path: &str, depth: usize) -> PyResult<Node<'py>> {
    let mut fields = Vec::new();
    let mut pairs = Vec::new();
    for item in map.items()?.iter() {
        let (k, v): (Bound<'py, PyAny>, Bound<'py, PyAny>) = item.extract()?;
        match scalar(&k, &format!("{path} key"))? {
            Some(Value::Str(key)) if pairs.is_empty() => {
                let node = collect(&v, &format!("{path}['{key}']"), depth + 1)?;
                fields.push((key.into_owned(), node));
            }
            Some(Value::I64(key)) if fields.is_empty() => {
                let child = format!("{path}[{key}]");
                let x = match scalar(&v, &child)? {
                    Some(Value::F64(x)) => x,
                    Some(Value::I64(_) | Value::U64(_)) => v.extract()?,
                    _ => {
                        return Err(bad(
                            v.py(),
                            &child,
                            &format!("must be a number, got {}", describe(&v)),
                        ))
                    }
                };
                pairs.push((key, x));
            }
            _ => {
                return Err(bad(
                    k.py(),
                    path,
                    &format!(
                        "must have all string keys or all integer keys, got the key {}",
                        describe(&k)
                    ),
                ))
            }
        }
    }
    Ok(if pairs.is_empty() {
        Node::Record {
            type_name: None,
            fields,
        }
    } else {
        Node::Leaf(Value::IntMap(pairs))
    })
}

/// First pass: one Python value as a [`Node`], by its runtime type.
fn collect<'py>(obj: &Bound<'py, PyAny>, path: &str, depth: usize) -> PyResult<Node<'py>> {
    let py = obj.py();
    if depth > MAX_DEPTH {
        return Err(bad(py, path, "is nested too deeply"));
    }
    if let Some(v) = scalar(obj, path)? {
        return Ok(Node::Leaf(v));
    }
    if let Ok(arr) = obj.cast::<PyUntypedArray>() {
        return array(arr, path);
    }
    if let Ok(map) = obj.cast::<PyMapping>() {
        return mapping(map, path, depth);
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let mut items = Vec::new();
        for (i, item) in obj.try_iter()?.enumerate() {
            items.push(collect(&item?, &format!("{path}[{i}]"), depth + 1)?);
        }
        // A list of integers (or an empty one) is an integer vector.
        let ints: Option<Vec<i64>> = items
            .iter()
            .map(|n| match n {
                Node::Leaf(Value::I64(k)) => Some(*k),
                _ => None,
            })
            .collect();
        return Ok(match ints {
            Some(ints) => Node::Leaf(Value::IntVec(ints)),
            None => Node::List(items),
        });
    }
    // A dataclass instance is a record tagged with its class name, fields
    // in declaration order.
    let dataclasses = py.import("dataclasses")?;
    if !obj.is_instance_of::<PyType>()
        && dataclasses
            .call_method1("is_dataclass", (obj,))?
            .is_truthy()?
    {
        let mut fields = Vec::new();
        for field in dataclasses.call_method1("fields", (obj,))?.try_iter()? {
            let name: String = field?.getattr("name")?.extract()?;
            let value = obj.getattr(name.as_str())?;
            let node = collect(&value, &format!("{path}.{name}"), depth + 1)?;
            fields.push((name, node));
        }
        return Ok(Node::Record {
            type_name: Some(type_name(obj)),
            fields,
        });
    }
    Err(unsupported(obj, path))
}

/// A 2-D array as a `MatF64`. With `any_layout` (`X` and `X_new`, which
/// every engine entry reads in either layout) a C-contiguous array is
/// viewed row-major in place. Otherwise only a Fortran-contiguous array is
/// read in place: column-major is the layout of the copy, so a view and a
/// copy of the same values give the same bits, which a row-major view of a
/// C-ordered `Y` does not.
fn matrix<'a>(
    arr: &'a PyReadonlyArray2<'_, f64>,
    any_layout: bool,
) -> Result<MatF64<'a>, BindError> {
    let (nrows, ncols) = (arr.shape()[0], arr.shape()[1]);
    match arr.as_slice() {
        Ok(s) if any_layout && arr.is_c_contiguous() => Ok(MatF64::Borrowed(
            MatRef::from_row_major_slice(s, nrows, ncols),
        )),
        Ok(s) if arr.is_fortran_contiguous() => MatF64::from_col_major(s, nrows, ncols),
        Ok(s) => Ok(MatF64::from_row_major(s, nrows, ncols)),
        // Not contiguous: the transposed view iterates column-major.
        Err(_) => Ok(MatF64::Owned {
            data: arr.as_array().t().iter().copied().collect(),
            nrows,
            ncols,
        }),
    }
}

/// Second pass: a [`Node`] as a `Value` that borrows from its guards.
fn convert<'a>(node: &'a Node<'_>, any_layout: bool) -> Result<Value<'a>, BindError> {
    Ok(match node {
        Node::Leaf(v) => v.clone(),
        Node::Vec(arr) => Value::Vec(match arr.as_slice() {
            Ok(s) => VecF64::Borrowed(s),
            Err(_) => VecF64::Owned(arr.as_array().iter().copied().collect()),
        }),
        Node::Mat(arr) => Value::Mat(matrix(arr, any_layout)?),
        Node::Record { type_name, fields } => {
            let mut rec = type_name.as_deref().map_or_else(Record::new, Record::typed);
            for (key, child) in fields {
                rec.push(key, convert(child, false)?)?;
            }
            Value::Record(rec)
        }
        Node::List(items) => Value::List(
            items
                .iter()
                .map(|item| convert(item, false))
                .collect::<Result<_, _>>()?,
        ),
    })
}

/// The record `plskit_bind::call` takes. `X` and `X_new` are the only
/// parameter names the seam knows: they select the matrix layout rule.
fn to_record<'a>(arguments: &'a [(String, Node<'_>)]) -> Result<Record<'a>, BindError> {
    let mut rec = Record::new();
    for (name, node) in arguments {
        rec.push(name, convert(node, name == "X" || name == "X_new")?)?;
    }
    Ok(rec)
}

// ── Value to Python ────────────────────────────────────────────────────

/// A matrix as a C-ordered 2-D array. A row-major buffer becomes the
/// array's own memory; any other matrix is copied row by row.
fn mat_to_py<'py>(py: Python<'py>, m: MatF64<'_>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = match m {
        MatF64::OwnedRowMajor { data, nrows, ncols } => {
            Array2::from_shape_vec((nrows, ncols), data)
                .map_err(|e| coded_err(py, "internal", format!("result matrix: {e}")))?
        }
        other => {
            let view = other.as_mat();
            Array2::from_shape_fn((view.nrows(), view.ncols()), |(i, j)| view[(i, j)])
        }
    };
    Ok(array.into_pyarray(py))
}

/// One value as a Python object. A tagged record goes through
/// `make_result(type_name, fields)` after its fields are converted, so
/// nested results are built innermost first.
fn to_py<'py>(
    py: Python<'py>,
    v: Value<'_>,
    make_result: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, b).to_owned().into_any(),
        Value::I64(n) => n.into_pyobject(py)?.into_any(),
        Value::U64(n) => n.into_pyobject(py)?.into_any(),
        Value::F64(x) => PyFloat::new(py, x).into_any(),
        Value::Str(s) => PyString::new(py, &s).into_any(),
        Value::Vec(VecF64::Owned(v)) => PyArray1::from_vec(py, v).into_any(),
        Value::Vec(VecF64::Borrowed(s)) => PyArray1::from_slice(py, s).into_any(),
        Value::Mat(m) => mat_to_py(py, m)?.into_any(),
        Value::IntMap(pairs) => {
            let dict = PyDict::new(py);
            for (k, x) in pairs {
                dict.set_item(k, x)?;
            }
            dict.into_any()
        }
        Value::IntVec(v) => PyList::new(py, v)?.into_any(),
        Value::BoolVec(v) => PyList::new(py, v)?.into_any(),
        Value::Record(r) => {
            let tag = r.type_name().map(str::to_owned);
            let fields = PyDict::new(py);
            for (key, value) in r.into_fields() {
                fields.set_item(key, to_py(py, value, make_result)?)?;
            }
            match tag {
                Some(tag) => make_result.call1((tag, fields))?,
                None => fields.into_any(),
            }
        }
        Value::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(to_py(py, item, make_result)?)?;
            }
            list.into_any()
        }
    })
}

// ── Exports ────────────────────────────────────────────────────────────

/// Run the public function `name`. `arguments` is a dict of its parameters
/// by name; `make_result(type_name, fields)` builds each result object.
/// Returns `(result, warnings)`, `warnings` a list of dicts that each hold
/// `kind` and `message`.
///
/// `plskit_bind::call` turns a panic under it into an `internal` error;
/// the guard here does the same for a panic in the conversion on either
/// side of it, which pyo3 would otherwise raise as its `PanicException`.
#[pyfunction]
fn call<'py>(
    py: Python<'py>,
    name: &str,
    arguments: &Bound<'py, PyDict>,
    make_result: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyList>)> {
    panic::catch_unwind(AssertUnwindSafe(|| run(py, name, arguments, make_result))).unwrap_or_else(
        |_| {
            Err(coded_err(
                py,
                "internal",
                format!("panic in the Python seam while running {name}"),
            ))
        },
    )
}

fn run<'py>(
    py: Python<'py>,
    name: &str,
    arguments: &Bound<'py, PyDict>,
    make_result: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyList>)> {
    let mut nodes = Vec::with_capacity(arguments.len());
    for (key, value) in arguments.iter() {
        let Ok(arg) = key.extract::<String>() else {
            return Err(coded_err(
                py,
                "invalid_argument",
                format!("argument names must be strings, got {}", describe(&key)),
            ));
        };
        let node = collect(&value, &arg, 0)?;
        nodes.push((arg, node));
    }
    let outcome = to_record(&nodes)
        .and_then(|record| plskit_bind::call(name, record))
        .map_err(|e| bind_err(py, e, make_result))?;
    let result = to_py(py, outcome.result, make_result)?;
    let warnings = PyList::empty(py);
    for w in outcome.warnings {
        warnings.append(to_py(py, Value::Record(w), make_result)?)?;
    }
    Ok((result, warnings))
}

/// The function registry and the result types as JSON.
#[pyfunction]
fn registry_json() -> String {
    plskit_bind::registry_json()
}

// The extension module is `plskit._plskit` (pyproject.toml `module-name`);
// `name` sets it, so the Rust function needs no leading underscore.
#[pymodule(name = "_plskit")]
fn plskit_ext(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PlsKitException", py.get_type::<PlsKitException>())?;
    m.add_function(wrap_pyfunction!(call, m)?)?;
    m.add_function(wrap_pyfunction!(registry_json, m)?)?;
    Ok(())
}
