//! Typed extraction of a call's arguments, with a record of which ones
//! the function read.

use crate::coerce;
use crate::error::BindError;
use crate::value::{MatF64, Record, Value, VecF64};

/// A call's arguments. [`crate::call`] has already rejected undeclared
/// keys, rejected null required arguments and filled absent optional
/// arguments with their defaults.
pub(crate) struct Inputs<'a> {
    fn_name: &'static str,
    fields: Vec<(String, Value<'a>)>,
    read: Vec<bool>,
}

fn named(name: &str, why: &str) -> BindError {
    BindError::invalid_argument(format!("{name} {why}"))
}

impl<'a> Inputs<'a> {
    pub(crate) fn new(fn_name: &'static str, record: Record<'a>) -> Self {
        let fields = record.into_fields();
        let read = vec![false; fields.len()];
        Self {
            fn_name,
            fields,
            read,
        }
    }

    /// Move a value out (`Null` when absent) and mark it read.
    pub(crate) fn take(&mut self, name: &str) -> Value<'a> {
        match self.fields.iter().position(|(k, _)| k == name) {
            Some(i) => {
                self.read[i] = true;
                std::mem::replace(&mut self.fields[i].1, Value::Null)
            }
            None => Value::Null,
        }
    }

    pub(crate) fn mat(&mut self, name: &str) -> Result<MatF64<'a>, BindError> {
        coerce::to_mat(self.take(name)).map_err(|w| named(name, &w))
    }

    pub(crate) fn opt_mat(&mut self, name: &str) -> Result<Option<MatF64<'a>>, BindError> {
        match self.take(name) {
            Value::Null => Ok(None),
            v => coerce::to_mat(v).map(Some).map_err(|w| named(name, &w)),
        }
    }

    pub(crate) fn vec(&mut self, name: &str) -> Result<VecF64<'a>, BindError> {
        coerce::to_vec(self.take(name)).map_err(|w| named(name, &w))
    }

    pub(crate) fn opt_vec(&mut self, name: &str) -> Result<Option<VecF64<'a>>, BindError> {
        match self.take(name) {
            Value::Null => Ok(None),
            v => coerce::to_vec(v).map(Some).map_err(|w| named(name, &w)),
        }
    }

    pub(crate) fn usize(&mut self, name: &str) -> Result<usize, BindError> {
        coerce::to_usize(&self.take(name)).map_err(|w| named(name, &w))
    }

    pub(crate) fn opt_usize(&mut self, name: &str) -> Result<Option<usize>, BindError> {
        match self.take(name) {
            Value::Null => Ok(None),
            v => coerce::to_usize(&v).map(Some).map_err(|w| named(name, &w)),
        }
    }

    pub(crate) fn opt_f64(&mut self, name: &str) -> Result<Option<f64>, BindError> {
        match self.take(name) {
            Value::Null => Ok(None),
            v => coerce::to_f64(&v).map(Some).map_err(|w| named(name, &w)),
        }
    }

    pub(crate) fn bool(&mut self, name: &str) -> Result<bool, BindError> {
        coerce::to_bool(&self.take(name)).map_err(|w| named(name, &w))
    }

    pub(crate) fn string(&mut self, name: &str) -> Result<String, BindError> {
        let v = self.take(name);
        coerce::to_str(&v)
            .map(str::to_owned)
            .map_err(|w| named(name, &w))
    }

    pub(crate) fn opt_string(&mut self, name: &str) -> Result<Option<String>, BindError> {
        match self.take(name) {
            Value::Null => Ok(None),
            v => coerce::to_str(&v)
                .map(|s| Some(s.to_owned()))
                .map_err(|w| named(name, &w)),
        }
    }

    pub(crate) fn seed(&mut self) -> Result<Option<u64>, BindError> {
        match self.take("seed") {
            Value::Null => Ok(None),
            v => coerce::to_seed(&v).map(Some).map_err(|w| named("seed", &w)),
        }
    }

    pub(crate) fn opt_record(&mut self, name: &str) -> Result<Option<Record<'a>>, BindError> {
        match self.take(name) {
            Value::Null => Ok(None),
            Value::Record(r) => Ok(Some(r)),
            other => Err(named(
                name,
                &format!(
                    "must be a record of named values (a dict / named list), got {}",
                    coerce::describe(&other)
                ),
            )),
        }
    }

    /// Fail on a non-null argument the function never read. Every `run`
    /// reads every declared parameter, so this fires only on a bind bug.
    pub(crate) fn finish(&self) -> Result<(), BindError> {
        for ((k, v), read) in self.fields.iter().zip(&self.read) {
            if !read && !v.is_null() {
                return Err(BindError::invalid_argument(format!(
                    "{}() does not use argument '{k}' in this call",
                    self.fn_name
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(fields: &[(&str, Value<'static>)]) -> Inputs<'static> {
        let mut r = Record::new();
        for (k, v) in fields {
            r.push(k, v.clone()).unwrap();
        }
        Inputs::new("f", r)
    }

    #[test]
    fn finish_flags_an_unread_non_null_argument() {
        let mut inp = inputs(&[
            ("a", Value::I64(1)),
            ("b", Value::I64(2)),
            ("c", Value::Null),
        ]);
        inp.usize("a").unwrap();
        let e = inp.finish().unwrap_err();
        assert_eq!(e.code, "invalid_argument");
        assert!(e.message.contains("'b'"), "{}", e.message);
    }

    #[test]
    fn errors_name_the_argument() {
        let mut inp = inputs(&[("k", Value::F64(2.5))]);
        let e = inp.usize("k").unwrap_err();
        assert_eq!(e.message, "k must be a non-negative whole number, got 2.5");
    }

    #[test]
    fn seed_null_is_none_and_string_is_parsed() {
        let mut inp = inputs(&[("seed", Value::Null)]);
        assert_eq!(inp.seed().unwrap(), None);
        let mut inp = inputs(&[("seed", Value::text("18446744073709551615"))]);
        assert_eq!(inp.seed().unwrap(), Some(u64::MAX));
    }

    #[test]
    fn length_one_vector_argument_widens() {
        let mut inp = inputs(&[("y", Value::F64(1.5))]);
        assert_eq!(inp.vec("y").unwrap().as_slice(), &[1.5]);
    }

    #[test]
    fn opt_record_rejects_a_non_record() {
        let mut inp = inputs(&[("args", Value::I64(1))]);
        assert_eq!(inp.opt_record("args").unwrap_err().code, "invalid_argument");
    }
}
