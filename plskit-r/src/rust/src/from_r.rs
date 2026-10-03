//! R values to `plskit_bind::Value`.
//!
//! Conversion runs in two passes. `collect` walks the R object and keeps
//! an owned `Robj` handle for every node; `convert` then borrows numeric
//! data from those handles, so a double vector or matrix reaches the
//! engine without a copy. Host data enters bind only through
//! `Record::push` (duplicate names are rejected there) and matrices only
//! through `MatF64::from_col_major` (which checks the length).

use std::borrow::Cow;

use extendr_api::prelude::*;
use plskit_bind::{registry, result_types, BindError, MatF64, ParamKind, Record, Value, VecF64};

/// An R object with every list element already visited.
enum Node {
    Atom(Robj),
    List { robj: Robj, items: Vec<Node> },
}

impl Node {
    fn collect(x: Robj) -> Node {
        match x.as_list() {
            Some(list) if x.rtype() == Rtype::List => {
                let items = list.values().map(Node::collect).collect();
                Node::List { robj: x, items }
            }
            _ => Node::Atom(x),
        }
    }
}

/// Where a value sits; decides how an R value is read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ctx {
    /// A data or option argument: names are ignored.
    Data,
    /// An `args`-kind argument, or a list nested in one (`find_k_args$args`):
    /// a list is a record, even an empty one.
    Args,
    /// Inside a passed-back result: a numeric vector whose names are all
    /// integers is an integer-keyed map.
    Model,
}

/// A call's arguments, collected from the named list the R stub builds.
pub(crate) struct Inputs {
    names: Vec<String>,
    nodes: Vec<Node>,
}

impl Inputs {
    pub(crate) fn collect(inputs: &List) -> Result<Self, BindError> {
        let names: Vec<String> = match inputs.names() {
            Some(n) => n.map(str::to_owned).collect(),
            None if inputs.is_empty() => Vec::new(),
            None => return Err(BindError::internal("inputs must be a named list")),
        };
        let nodes = inputs.values().map(Node::collect).collect();
        Ok(Self { names, nodes })
    }

    /// The record `plskit_bind::call` takes. Each argument is read by the
    /// kind its parameter declares.
    pub(crate) fn to_record(&self, fn_name: &str) -> Result<Record<'_>, BindError> {
        let spec = registry().iter().find(|f| f.name == fn_name);
        let mut rec = Record::new();
        for (key, node) in self.names.iter().zip(&self.nodes) {
            let kind = spec
                .and_then(|s| s.params.iter().find(|p| p.name == key))
                .map(|p| p.kind);
            let ctx = match kind {
                Some(ParamKind::Args) => Ctx::Args,
                Some(ParamKind::Model | ParamKind::ModelOrMat) => Ctx::Model,
                _ => Ctx::Data,
            };
            rec.push(key, convert(node, ctx, key)?)
                .map_err(|e| bad(key, &e.message))?;
        }
        Ok(rec)
    }
}

fn bad(path: &str, why: &str) -> BindError {
    BindError::invalid_argument(format!("{path}: {why}"))
}

fn convert<'a>(node: &'a Node, ctx: Ctx, path: &str) -> Result<Value<'a>, BindError> {
    match node {
        Node::List { robj, items } => list(robj, items, ctx, path),
        Node::Atom(x) => atom(x, ctx, path),
    }
}

fn list<'a>(robj: &Robj, items: &'a [Node], ctx: Ctx, path: &str) -> Result<Value<'a>, BindError> {
    // The first class in the vector that maps to a known plskit result
    // type wins (a user class prepended with `class(fit) <- c("my_fit",
    // class(fit))` does not break round-tripping). If none of the classes
    // map to a known type, the list is rejected, reporting the class the
    // caller is most likely to recognize: the first one.
    let tag = if let Some(classes) = robj.class() {
        let classes: Vec<&str> = classes.collect();
        let found = classes
            .iter()
            .find_map(|c| result_types().iter().find(|t| &t.r_class == c));
        let Some(t) = found else {
            let first = classes.first().copied().unwrap_or("");
            return Err(bad(
                path,
                &format!("a list of class '{first}' is not a plskit result"),
            ));
        };
        Some(t.name)
    } else {
        None
    };
    let names: Option<Vec<&str>> = robj.names().map(Iterator::collect);
    match names {
        Some(names) => {
            let mut rec = tag.map_or_else(Record::new, Record::typed);
            for (name, item) in names.iter().zip(items) {
                if name.is_empty() {
                    return Err(bad(path, "every element of the list must be named"));
                }
                let child = format!("{path}${name}");
                rec.push(name, convert(item, ctx, &child)?)
                    .map_err(|e| bad(&child, &e.message))?;
            }
            Ok(Value::Record(rec))
        }
        None if items.is_empty() && (ctx == Ctx::Args || tag.is_some()) => {
            Ok(Value::Record(tag.map_or_else(Record::new, Record::typed)))
        }
        None if tag.is_some() => Err(bad(path, "a plskit result must be a named list")),
        None => items
            .iter()
            .enumerate()
            .map(|(i, item)| convert(item, ctx, &format!("{path}[[{}]]", i + 1)))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::List),
    }
}

fn atom<'a>(x: &'a Robj, ctx: Ctx, path: &str) -> Result<Value<'a>, BindError> {
    match x.rtype() {
        Rtype::Null => Ok(Value::Null),
        Rtype::Doubles => doubles(x, ctx, path),
        Rtype::Integers => integers(x, path),
        Rtype::Logicals => logicals(x, path),
        Rtype::Strings => string(x, path),
        other => Err(bad(
            path,
            &format!("R values of type {other:?} are not supported"),
        )),
    }
}

fn dims(x: &Robj) -> Option<Vec<usize>> {
    let d = x.get_attrib("dim")?.as_integer_vector()?;
    Some(
        d.into_iter()
            .map(|v| usize::try_from(v).unwrap_or(0))
            .collect(),
    )
}

/// Integer keys when every name parses as one, and does so canonically:
/// the name must equal the parsed key's own decimal form, so `"+1"` and
/// `"01"` are not integer names (only `"1"` is `1`'s name). A name that
/// fails this makes the whole vector not an int map (every name must
/// be an integer). Two names that do both round-trip to the same key (a
/// literal duplicate name, e.g. `c("3", "3")`) raise `invalid_argument`
/// instead of silently overwriting one with the other.
fn int_names(x: &Robj, path: &str) -> Result<Option<Vec<i64>>, BindError> {
    let Some(names) = x.names() else {
        return Ok(None);
    };
    let mut keys = Vec::new();
    for n in names {
        match n.parse::<i64>() {
            Ok(k) if k.to_string() == n => keys.push(k),
            _ => return Ok(None),
        }
    }
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(bad(path, "duplicate integer keys in a model field"));
    }
    Ok(Some(keys))
}

fn doubles<'a>(x: &'a Robj, ctx: Ctx, path: &str) -> Result<Value<'a>, BindError> {
    let data = x
        .as_real_slice()
        .ok_or_else(|| BindError::internal(format!("{path}: double vector without data")))?;
    if let Some(d) = dims(x) {
        match d.as_slice() {
            // A 1-D array (`array(y)`): the same shape as a plain vector.
            [_n] => {}
            [nrow, ncol] => return MatF64::from_col_major(data, *nrow, *ncol).map(Value::Mat),
            _ => {
                return Err(bad(
                    path,
                    &format!("a {}-dimensional array is not supported", d.len()),
                ))
            }
        }
    }
    if ctx == Ctx::Model {
        if let Some(keys) = int_names(x, path)? {
            return Ok(Value::IntMap(
                keys.into_iter().zip(data.iter().copied()).collect(),
            ));
        }
    }
    if data.len() == 1 {
        Ok(Value::F64(data[0]))
    } else {
        Ok(Value::Vec(VecF64::Borrowed(data)))
    }
}

fn integers(x: &Robj, path: &str) -> Result<Value<'static>, BindError> {
    if x.inherits("factor") {
        return Err(bad(
            path,
            "a factor is not supported; convert it to numeric first",
        ));
    }
    if let Some(d) = dims(x) {
        match d.as_slice() {
            // A 1-D array (`array(y)`): the same shape as a plain vector.
            [_n] => {}
            _ => return Err(bad(path, "an integer matrix must be converted to double")),
        }
    }
    let data = x
        .as_integer_slice()
        .ok_or_else(|| BindError::internal(format!("{path}: integer vector without data")))?;
    let na = i32::MIN;
    if let [v] = data {
        return Ok(if *v == na {
            Value::F64(f64::NAN)
        } else {
            Value::I64(i64::from(*v))
        });
    }
    if data.contains(&na) {
        let widened = data
            .iter()
            .map(|&v| if v == na { f64::NAN } else { f64::from(v) })
            .collect();
        return Ok(Value::Vec(VecF64::Owned(widened)));
    }
    Ok(Value::IntVec(data.iter().map(|&v| i64::from(v)).collect()))
}

fn logicals(x: &Robj, path: &str) -> Result<Value<'static>, BindError> {
    if let Some(d) = dims(x) {
        match d.as_slice() {
            // A 1-D array (`array(y)`): the same shape as a plain vector.
            [_n] => {}
            _ => return Err(bad(path, "a logical matrix is not supported")),
        }
    }
    let data = x
        .as_logical_slice()
        .ok_or_else(|| BindError::internal(format!("{path}: logical vector without data")))?;
    if data.iter().any(CanBeNA::is_na) {
        return Err(bad(path, "NA is not a valid logical value"));
    }
    Ok(match data {
        [b] => Value::Bool(b.is_true()),
        _ => Value::BoolVec(data.iter().map(Rbool::is_true).collect()),
    })
}

fn string<'a>(x: &'a Robj, path: &str) -> Result<Value<'a>, BindError> {
    if x.len() != 1 {
        return Err(bad(
            path,
            &format!("a character vector must have length 1, got {}", x.len()),
        ));
    }
    if x.is_na() {
        return Err(bad(path, "NA is not a valid string"));
    }
    let s = x
        .as_str()
        .ok_or_else(|| BindError::internal(format!("{path}: string without data")))?;
    Ok(Value::Str(Cow::Borrowed(s)))
}
