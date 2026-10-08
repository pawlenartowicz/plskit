//! The value tree every wrapper marshals into and out of `plskit-bind`.
//!
//! Inputs borrow host memory (a numpy array in Python, an R `REALSXP`);
//! outputs are owned. Owned matrices are stored contiguously, column-major
//! or row-major, rather than as `faer::Mat`: faer pads an owned matrix's
//! column stride, and a wrapper copies each array out as one contiguous
//! slice (or takes the buffer whole).

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
    /// Rust-owned elements, row-major and contiguous: element `(i, j)` is
    /// `data[i * ncols + j]`. A wrapper that wants a C-ordered array can
    /// take `data` whole by matching on this variant; `col_major()` still
    /// returns column-major data.
    ///
    /// Same length invariant as `Owned`.
    OwnedRowMajor {
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
            Self::Owned { nrows, .. } | Self::OwnedRowMajor { nrows, .. } => *nrows,
        }
    }

    /// Column count.
    #[must_use]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(m) => m.ncols(),
            Self::Owned { ncols, .. } | Self::OwnedRowMajor { ncols, .. } => *ncols,
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
            Self::OwnedRowMajor { data, nrows, ncols } => {
                MatRef::from_row_major_slice(data, *nrows, *ncols)
            }
        }
    }

    /// The elements column-major; borrowed when already stored that way.
    #[must_use]
    pub fn col_major(&self) -> Cow<'_, [f64]> {
        match self {
            Self::Owned { data, .. } => Cow::Borrowed(data),
            Self::Borrowed(_) | Self::OwnedRowMajor { .. } => {
                let m = self.as_mat();
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
            Self::OwnedRowMajor { data, .. } => MatF64::OwnedRowMajor { data, nrows, ncols },
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_rejects_a_duplicate_key() {
        let mut r = Record::new();
        r.push("a", Value::I64(1)).unwrap();
        let e = r.push("a", Value::I64(2)).unwrap_err();
        assert_eq!(e.code, "invalid_argument");
        assert!(e.message.contains("'a'"), "{}", e.message);
    }

    #[test]
    fn from_col_major_checks_the_length() {
        let e = MatF64::from_col_major(&[1.0, 2.0, 3.0], 2, 2).unwrap_err();
        assert_eq!(e.code, "invalid_argument");
    }

    #[test]
    fn row_major_owned_reads_like_the_same_column_major_matrix() {
        // [[1, 2, 3], [4, 5, 6]]
        let rm = MatF64::OwnedRowMajor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            nrows: 2,
            ncols: 3,
        };
        let cm = MatF64::Owned {
            data: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            nrows: 2,
            ncols: 3,
        };
        assert_eq!((rm.nrows(), rm.ncols()), (2, 3));
        assert_eq!(rm.col_major(), cm.col_major());
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(rm.as_mat()[(i, j)], cm.as_mat()[(i, j)]);
            }
        }
        let owned = rm.into_owned();
        assert_eq!(owned.col_major(), cm.col_major());
    }
}
