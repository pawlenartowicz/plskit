//! The `split_nb` reroute warning, rendered once so every language
//! prints the sentence `_api.py`'s `_warn_if_rerouted` prints.

use crate::convert::{count, opt};
use crate::fmt::g4;
use crate::value::{Record, Value};

/// A warning record when the auto-gate ran something other than what was
/// asked; `None` when `requested == actual`.
pub(crate) fn rerouted(
    requested: Option<&str>,
    actual: Option<&str>,
    n_perm: Option<usize>,
    stable_rank: Option<f64>,
    n_eff: Option<f64>,
) -> Option<Record<'static>> {
    if requested == actual {
        return None;
    }
    let requested = requested.unwrap_or("None");
    let actual = actual.unwrap_or("None");
    let mut saw = Vec::new();
    if let Some(sr) = stable_rank {
        saw.push(format!("stable rank of the standardized X = {}", g4(sr)));
    }
    if let Some(ne) = n_eff {
        saw.push(format!("n_eff = {}", g4(ne)));
    }
    let seen = if saw.is_empty() {
        String::new()
    } else {
        format!(" ({})", saw.join("; "))
    };
    let n_perm_text = n_perm.map_or_else(|| "None".to_owned(), |n| n.to_string());
    let message = format!(
        "'{requested}' was rerouted to '{actual}': the {requested} auto-gate flagged this \
         design{seen}. The fallback runs n_perm={n_perm_text} permutations, so it costs more \
         than {requested}. Pass args={{'force': True}} to run {requested} anyway."
    );
    Some(
        Record::new()
            .field("kind", Value::text("rerouted"))
            .field("requested", Value::text(requested))
            .field("actual", Value::text(actual))
            .field("n_perm", opt(n_perm, count))
            .field("stable_rank", opt(stable_rank, Value::F64))
            .field("n_eff", opt(n_eff, Value::F64))
            .field("message", Value::text(&message)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_when_the_method_ran_as_asked() {
        assert!(rerouted(
            Some("split_nb"),
            Some("split_nb"),
            None,
            Some(2.5),
            Some(60.0)
        )
        .is_none());
        assert!(rerouted(None, None, None, None, Some(60.0)).is_none());
    }

    #[test]
    fn sentence_matches_the_python_warning() {
        // Rendered by the f-string in `_warn_if_rerouted` (_api.py).
        let w = rerouted(
            Some("split_nb"),
            Some("split_exact"),
            Some(1000),
            Some(2.5),
            Some(60.0),
        )
        .unwrap();
        let Some(Value::Str(msg)) = w.get("message") else {
            panic!()
        };
        assert_eq!(
            msg,
            "'split_nb' was rerouted to 'split_exact': the split_nb auto-gate flagged this design \
             (stable rank of the standardized X = 2.5; n_eff = 60). The fallback runs n_perm=1000 \
             permutations, so it costs more than split_nb. Pass args={'force': True} to run \
             split_nb anyway."
        );
        let keys: Vec<&str> = w.keys().collect();
        assert_eq!(
            keys,
            [
                "kind",
                "requested",
                "actual",
                "n_perm",
                "stable_rank",
                "n_eff",
                "message"
            ]
        );
    }
}
