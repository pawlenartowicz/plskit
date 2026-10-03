//! Registry tests: every declared result type exists, and every function
//! rejects an unknown argument.

use plskit_bind::{call, registry, result_type, Record, Value};

#[test]
fn every_declared_result_type_exists() {
    for f in registry() {
        for t in f.result_types {
            assert!(
                result_type(t).is_some(),
                "{}: unknown result type {t}",
                f.name
            );
        }
    }
}

#[test]
fn every_function_rejects_an_unknown_argument() {
    for f in registry() {
        let mut r = Record::new();
        r.push("definitely_not_a_param", Value::I64(1)).unwrap();
        let e = call(f.name, r).unwrap_err();
        assert_eq!(e.code, "invalid_argument", "{}", f.name);
        assert!(
            e.message.contains("'definitely_not_a_param'"),
            "{}: {}",
            f.name,
            e.message
        );
    }
}
