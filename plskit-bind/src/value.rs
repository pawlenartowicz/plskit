//! The value tree every wrapper marshals into and out of `plskit-bind`.
//!
//! Inputs borrow host memory (an R `REALSXP`, a Julia `Matrix{Float64}`);
//! outputs are owned. Owned matrices are stored column-major and
//! contiguous rather than as `faer::Mat`: faer pads an owned matrix's
//! column stride, and the C ABI hands the host one pointer per array.

use std::borrow::Cow;

use faer::{Col, ColRef, Mat, MatRef};

use crate::error::BindError;

/// A 1-D `f64` array.
#[derive(Debug, Clone)]
pub enum VecF64<'a> {
    /// Host memory, valid for the duration of the call.
    Borrowed(&'a [f64]),
    /// Rust-owned elements.
    Owned(Vec<f64>),
}

impl VecF64<'_> {
    /// The elements.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(v) => v,
        }
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// True when there are no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// A faer view of the elements, without a copy.
    #[must_use]
    pub fn as_col(&self) -> ColRef<'_, f64> {
        ColRef::from_slice(self.as_slice())
    }

    /// A copy that no longer points into host memory.
    #[must_use]
    pub fn into_owned(self) -> VecF64<'static> {
        match self {
            Self::Borrowed(s) => VecF64::Owned(s.to_vec()),
            Self::Owned(v) => VecF64::Owned(v),
        }
    }

    /// Copy a faer column out.
    #[must_use]
    pub fn from_col(c: &Col<f64>) -> VecF64<'static> {
        VecF64::Owned((0..c.nrows()).map(|i| c[i]).collect())
    }
}

/// A 2-D `f64` array.
#[derive(Debug, Clone)]
pub enum MatF64<'a> {
    /// Host memory, valid for the duration of the call.
    Borrowed(MatRef<'a, f64>),
    /// Rust-owned elements, column-major and contiguous: element `(i, j)`
    /// is `data[i + j * nrows]`.
    ///
    /// Invariant: `data.len() == nrows * ncols`. Nothing enforces this on
    /// construction, since the fields are public; `as_mat()` panics if it
    /// does not hold. A seam building a host matrix should use
    /// [`MatF64::from_col_major`] (which returns a `Borrowed`, checked)
    /// rather than constructing `Owned` directly from host data.
    Owned {
        /// The elements.
        data: Vec<f64>,
        /// Row count.
        nrows: usize,
        /// Column count.
        ncols: usize,
    },
}

impl<'a> MatF64<'a> {
    /// Borrow a column-major host buffer of `nrows * ncols` elements.
    pub fn from_col_major(data: &'a [f64], nrows: usize, ncols: usize) -> Result<Self, BindError> {
        if nrows.checked_mul(ncols) != Some(data.len()) {
            return Err(BindError::invalid_argument(format!(
                "matrix buffer holds {} elements, expected {nrows} x {ncols}",
                data.len()
            )));
        }
        Ok(Self::Borrowed(MatRef::from_column_major_slice(
            data, nrows, ncols,
        )))
    }
}

impl MatF64<'_> {
    /// Row count.
    #[must_use]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(m) => m.nrows(),
            Self::Owned { nrows, .. } => *nrows,
        }
    }

    /// Column count.
    #[must_use]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(m) => m.ncols(),
            Self::Owned { ncols, .. } => *ncols,
        }
    }

    /// A faer view, without a copy.
    #[must_use]
    pub fn as_mat(&self) -> MatRef<'_, f64> {
        match self {
            Self::Borrowed(m) => *m,
            Self::Owned { data, nrows, ncols } => {
                MatRef::from_column_major_slice(data, *nrows, *ncols)
            }
        }
    }

    /// The elements column-major; borrowed when already stored that way.
    #[must_use]
    pub fn col_major(&self) -> Cow<'_, [f64]> {
        match self {
            Self::Owned { data, .. } => Cow::Borrowed(data),
            Self::Borrowed(m) => {
                let mut v = Vec::with_capacity(m.nrows() * m.ncols());
                for j in 0..m.ncols() {
                    for i in 0..m.nrows() {
                        v.push(m[(i, j)]);
                    }
                }
                Cow::Owned(v)
            }
        }
    }

    /// A copy that no longer points into host memory.
    #[must_use]
    pub fn into_owned(self) -> MatF64<'static> {
        let (nrows, ncols) = (self.nrows(), self.ncols());
        match self {
            Self::Owned { data, .. } => MatF64::Owned { data, nrows, ncols },
            b @ Self::Borrowed(_) => MatF64::Owned {
                data: b.col_major().into_owned(),
                nrows,
                ncols,
            },
        }
    }

    /// Copy a faer matrix out, column-major.
    #[must_use]
    pub fn from_mat(m: &Mat<f64>) -> MatF64<'static> {
        let (nrows, ncols) = (m.nrows(), m.ncols());
        let mut data = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(m[(i, j)]);
            }
        }
        MatF64::Owned { data, nrows, ncols }
    }

    /// Copy a row-major buffer (`flat[i * ncols + j]`) into column-major storage.
    #[must_use]
    pub fn from_row_major(flat: &[f64], nrows: usize, ncols: usize) -> MatF64<'static> {
        let mut data = vec![0.0; nrows * ncols];
        for i in 0..nrows {
            for j in 0..ncols {
                data[i + j * nrows] = flat[i * ncols + j];
            }
        }
        MatF64::Owned { data, nrows, ncols }
    }
}

/// An ordered list of named values, optionally tagged with a result type
/// name (`"PLS1Result"`, `"CIScalar"`, ...).
#[derive(Debug, Clone, Default)]
pub struct Record<'a> {
    type_name: Option<String>,
    fields: Vec<(String, Value<'a>)>,
}

impl<'a> Record<'a> {
    /// An untagged, empty record.
    #[must_use]
    pub fn new() -> Self {
        Self {
            type_name: None,
            fields: Vec::new(),
        }
    }

    /// An empty record tagged with a result type name.
    #[must_use]
    pub fn typed(type_name: &str) -> Self {
        Self {
            type_name: Some(type_name.to_owned()),
            fields: Vec::new(),
        }
    }

    /// The result type name, or `None` for an untagged record.
    #[must_use]
    pub fn type_name(&self) -> Option<&str> {
        self.type_name.as_deref()
    }

    /// Tag (or re-tag) the record.
    pub fn set_type_name(&mut self, type_name: &str) {
        self.type_name = Some(type_name.to_owned());
    }

    /// Builder for records whose keys are fixed in code (never duplicated).
    #[must_use]
    pub fn field(mut self, key: &str, value: Value<'a>) -> Self {
        debug_assert!(self.get(key).is_none(), "duplicate field {key}");
        self.fields.push((key.to_owned(), value));
        self
    }

    /// Append a field; a key that is already present raises `invalid_argument`.
    pub fn push(&mut self, key: &str, value: Value<'a>) -> Result<(), BindError> {
        if self.get(key).is_some() {
            return Err(BindError::invalid_argument(format!(
                "duplicate key '{key}'"
            )));
        }
        self.fields.push((key.to_owned(), value));
        Ok(())
    }

    /// Replace a field's value in place, or append it when absent.
    pub fn set(&mut self, key: &str, value: Value<'a>) {
        match self.fields.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => self.fields.push((key.to_owned(), value)),
        }
    }

    /// A field's value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Field names, in order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.fields.iter().map(|(k, _)| k.as_str())
    }

    /// `(name, value)` pairs, in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value<'a>)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of fields.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// True when there are no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The fields, consuming the record.
    #[must_use]
    pub fn into_fields(self) -> Vec<(String, Value<'a>)> {
        self.fields
    }

    /// A deep copy that no longer points into host memory.
    #[must_use]
    pub fn into_owned(self) -> Record<'static> {
        Record {
            type_name: self.type_name,
            fields: self
                .fields
                .into_iter()
                .map(|(k, v)| (k, v.into_owned()))
                .collect(),
        }
    }
}

/// One value crossing the wrapper boundary.
#[derive(Debug, Clone)]
pub enum Value<'a> {
    /// `None` / `NULL` / `nothing`.
    Null,
    /// A boolean.
    Bool(bool),
    /// A signed integer (counts leave bind as this).
    I64(i64),
    /// An unsigned 64-bit integer (seeds leave bind as this).
    U64(u64),
    /// A float.
    F64(f64),
    /// A string.
    Str(Cow<'a, str>),
    /// A 1-D float array.
    Vec(VecF64<'a>),
    /// A 2-D float array.
    Mat(MatF64<'a>),
    /// `dict[int, float]` (`cv_scores`, `cv_scores_se`, `bic_scores`).
    IntMap(Vec<(i64, f64)>),
    /// `list[int]` / an integer array (`keep_grid`, `PLS3Result.n_iter`).
    IntVec(Vec<i64>),
    /// A bool array (`PLS3Result.converged`).
    BoolVec(Vec<bool>),
    /// A named record (a result, a nested result, an `args` dict).
    Record(Record<'a>),
    /// An ordered list (`list[CIScalar]`, warnings).
    List(Vec<Value<'a>>),
}

impl Value<'_> {
    /// The C ABI kind code: 0 Null, 1 Bool, 2 I64, 3 U64, 4 F64, 5 Str,
    /// 6 Vec, 7 Mat, 8 `IntMap`, 9 `IntVec`, 10 `BoolVec`, 11 Record, 12 List.
    #[must_use]
    pub fn kind_code(&self) -> u8 {
        match self {
            Self::Null => 0,
            Self::Bool(_) => 1,
            Self::I64(_) => 2,
            Self::U64(_) => 3,
            Self::F64(_) => 4,
            Self::Str(_) => 5,
            Self::Vec(_) => 6,
            Self::Mat(_) => 7,
            Self::IntMap(_) => 8,
            Self::IntVec(_) => 9,
            Self::BoolVec(_) => 10,
            Self::Record(_) => 11,
            Self::List(_) => 12,
        }
    }

    /// A short name of the variant, for error messages.
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "bool",
            Self::I64(_) | Self::U64(_) => "integer",
            Self::F64(_) => "number",
            Self::Str(_) => "string",
            Self::Vec(_) => "vector",
            Self::Mat(_) => "matrix",
            Self::IntMap(_) => "integer-keyed map",
            Self::IntVec(_) => "integer vector",
            Self::BoolVec(_) => "logical vector",
            Self::Record(_) => "record",
            Self::List(_) => "list",
        }
    }

    /// True for `Null`.
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// A deep copy that no longer points into host memory.
    #[must_use]
    pub fn into_owned(self) -> Value<'static> {
        match self {
            Self::Null => Value::Null,
            Self::Bool(b) => Value::Bool(b),
            Self::I64(n) => Value::I64(n),
            Self::U64(n) => Value::U64(n),
            Self::F64(x) => Value::F64(x),
            Self::Str(s) => Value::Str(Cow::Owned(s.into_owned())),
            Self::Vec(v) => Value::Vec(v.into_owned()),
            Self::Mat(m) => Value::Mat(m.into_owned()),
            Self::IntMap(m) => Value::IntMap(m),
            Self::IntVec(v) => Value::IntVec(v),
            Self::BoolVec(v) => Value::BoolVec(v),
            Self::Record(r) => Value::Record(r.into_owned()),
            Self::List(l) => Value::List(l.into_iter().map(Value::into_owned).collect()),
        }
    }
}

impl Value<'static> {
    /// An owned string value.
    #[must_use]
    pub fn text(s: &str) -> Self {
        Value::Str(Cow::Owned(s.to_owned()))
    }
}

/// What a successful [`call`](crate::call) returns.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// The result: a tagged record, or a plain array (`pls1_predict`).
    pub result: Value<'static>,
    /// Structured warnings, each with a pre-rendered `message`.
    pub warnings: Vec<Record<'static>>,
}

impl Outcome {
    /// An outcome with no warnings.
    #[must_use]
    pub fn new(result: Value<'static>) -> Self {
        Self {
            result,
            warnings: Vec::new(),
        }
    }

    /// `{ result, warnings }`, the shape the C ABI hands out.
    #[must_use]
    pub fn into_record(self) -> Record<'static> {
        Record::new().field("result", self.result).field(
            "warnings",
            Value::List(self.warnings.into_iter().map(Value::Record).collect()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_codes_follow_declaration_order() {
        let vals = [
            Value::Null,
            Value::Bool(true),
            Value::I64(1),
            Value::U64(1),
            Value::F64(1.0),
            Value::text("a"),
            Value::Vec(VecF64::Owned(vec![1.0])),
            Value::Mat(MatF64::Owned {
                data: vec![1.0],
                nrows: 1,
                ncols: 1,
            }),
            Value::IntMap(vec![(1, 1.0)]),
            Value::IntVec(vec![1]),
            Value::BoolVec(vec![true]),
            Value::Record(Record::new()),
            Value::List(vec![]),
        ];
        let codes: Vec<u8> = vals.iter().map(Value::kind_code).collect();
        assert_eq!(codes, (0u8..=12).collect::<Vec<_>>());
    }

    #[test]
    fn push_rejects_a_duplicate_key() {
        let mut r = Record::new();
        r.push("a", Value::I64(1)).unwrap();
        let e = r.push("a", Value::I64(2)).unwrap_err();
        assert_eq!(e.code, "invalid_argument");
        assert!(e.message.contains("'a'"), "{}", e.message);
    }

    #[test]
    fn set_replaces_in_place_and_appends_new_keys() {
        let mut r = Record::new()
            .field("a", Value::I64(1))
            .field("b", Value::I64(2));
        r.set("a", Value::I64(9));
        r.set("c", Value::I64(3));
        let keys: Vec<&str> = r.keys().collect();
        assert_eq!(keys, ["a", "b", "c"]);
        assert!(matches!(r.get("a"), Some(Value::I64(9))));
    }

    #[test]
    fn into_owned_detaches_borrowed_arrays() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let owned = {
            let m = MatF64::from_col_major(&data, 2, 3).unwrap();
            Value::Record(Record::new().field("M", Value::Mat(m))).into_owned()
        };
        let Value::Record(r) = owned else { panic!() };
        let Some(Value::Mat(MatF64::Owned {
            data: d,
            nrows: 2,
            ncols: 3,
        })) = r.get("M")
        else {
            panic!("expected an owned 2x3 matrix")
        };
        assert_eq!(d, &data);
    }

    #[test]
    fn from_mat_is_column_major_and_contiguous() {
        let m = Mat::<f64>::from_fn(2, 3, |i, j| (10 * i + j) as f64);
        let MatF64::Owned { data, nrows, ncols } = MatF64::from_mat(&m) else {
            panic!()
        };
        assert_eq!((nrows, ncols), (2, 3));
        assert_eq!(data, vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
    }

    #[test]
    fn from_row_major_transposes_the_layout() {
        // 2 x 3, rows [0 1 2] and [10 11 12]
        let flat = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
        let m = MatF64::from_row_major(&flat, 2, 3);
        assert_eq!(m.as_mat()[(1, 2)], 12.0);
        assert_eq!(m.col_major().as_ref(), &[0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
    }

    #[test]
    fn from_col_major_checks_the_length() {
        let e = MatF64::from_col_major(&[1.0, 2.0, 3.0], 2, 2).unwrap_err();
        assert_eq!(e.code, "invalid_argument");
    }

    #[test]
    fn outcome_record_has_result_and_warnings() {
        let o = Outcome {
            result: Value::F64(1.0),
            warnings: vec![Record::new().field("kind", Value::text("rerouted"))],
        };
        let r = o.into_record();
        assert_eq!(r.keys().collect::<Vec<_>>(), ["result", "warnings"]);
        assert!(matches!(r.get("warnings"), Some(Value::List(l)) if l.len() == 1));
    }
}
