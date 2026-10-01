//! Helpers shared by the integration tests.
#![allow(dead_code, clippy::cast_precision_loss)]

use plskit_bind::{call, BindError, MatF64, Outcome, Record, Value, VecF64};

/// Deterministic pseudo-random numbers in [-1, 1) (xorshift64*), so the
/// tests need no rand dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let u = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (u as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// `n x p` design and `y = 4 (x0 + x1) + noise`, column-major.
pub fn data(n: usize, p: usize, seed: u64) -> (Value<'static>, Value<'static>) {
    let mut rng = Rng::new(seed);
    let x: Vec<f64> = (0..n * p).map(|_| rng.next()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 4.0 * (x[i] + x[i + n]) + 0.5 * rng.next())
        .collect();
    (
        Value::Mat(MatF64::Owned {
            data: x,
            nrows: n,
            ncols: p,
        }),
        Value::Vec(VecF64::Owned(y)),
    )
}

/// `n x q` second block for PLS3, sharing signal with `x0`.
pub fn block_y(n: usize, q: usize, x: &Value<'_>, seed: u64) -> Value<'static> {
    let Value::Mat(xm) = x else {
        panic!("x must be a matrix")
    };
    let xd = xm.col_major();
    let mut rng = Rng::new(seed);
    let data: Vec<f64> = (0..n * q)
        .map(|idx| {
            let i = idx % n;
            2.0 * xd[i] + rng.next()
        })
        .collect();
    Value::Mat(MatF64::Owned {
        data,
        nrows: n,
        ncols: q,
    })
}

pub fn rec(fields: Vec<(&str, Value<'static>)>) -> Record<'static> {
    let mut r = Record::new();
    for (k, v) in fields {
        r.push(k, v).unwrap();
    }
    r
}

pub fn ok(name: &str, fields: Vec<(&str, Value<'static>)>) -> Outcome {
    call(name, rec(fields)).unwrap_or_else(|e| panic!("{name}: {} ({})", e.message, e.code))
}

pub fn err(name: &str, fields: Vec<(&str, Value<'static>)>) -> BindError {
    match call(name, rec(fields)) {
        Ok(_) => panic!("{name}: expected an error"),
        Err(e) => e,
    }
}

pub fn record(o: &Outcome) -> &Record<'static> {
    match &o.result {
        Value::Record(r) => r,
        other => panic!("expected a record, got {other:?}"),
    }
}

pub fn field<'r>(r: &'r Record<'static>, key: &str) -> &'r Value<'static> {
    r.get(key).unwrap_or_else(|| panic!("missing field {key}"))
}

pub fn f(r: &Record<'static>, key: &str) -> f64 {
    match field(r, key) {
        Value::F64(x) => *x,
        other => panic!("{key}: expected f64, got {other:?}"),
    }
}

pub fn i(r: &Record<'static>, key: &str) -> i64 {
    match field(r, key) {
        Value::I64(n) => *n,
        other => panic!("{key}: expected i64, got {other:?}"),
    }
}

pub fn s<'r>(r: &'r Record<'static>, key: &str) -> &'r str {
    match field(r, key) {
        Value::Str(x) => x,
        other => panic!("{key}: expected str, got {other:?}"),
    }
}

pub fn floats(v: &Value<'_>) -> Vec<f64> {
    match v {
        Value::Vec(x) => x.as_slice().to_vec(),
        Value::Mat(m) => m.col_major().into_owned(),
        other => panic!("expected an array, got {other:?}"),
    }
}

/// Bitwise deep equality (NaN equals NaN), ignoring Borrowed vs Owned.
pub fn same(a: &Value<'_>, b: &Value<'_>) -> bool {
    let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::I64(x), Value::I64(y)) => x == y,
        (Value::U64(x), Value::U64(y)) => x == y,
        (Value::F64(x), Value::F64(y)) => x.to_bits() == y.to_bits(),
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Vec(x), Value::Vec(y)) => bits(x.as_slice()) == bits(y.as_slice()),
        (Value::Mat(x), Value::Mat(y)) => {
            (x.nrows(), x.ncols()) == (y.nrows(), y.ncols())
                && bits(&x.col_major()) == bits(&y.col_major())
        }
        (Value::IntMap(x), Value::IntMap(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|(p, q)| p.0 == q.0 && p.1.to_bits() == q.1.to_bits())
        }
        (Value::IntVec(x), Value::IntVec(y)) => x == y,
        (Value::BoolVec(x), Value::BoolVec(y)) => x == y,
        (Value::Record(x), Value::Record(y)) => {
            x.type_name() == y.type_name()
                && x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|((k1, v1), (k2, v2))| k1 == k2 && same(v1, v2))
        }
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
        }
        _ => false,
    }
}
