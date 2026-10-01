//! Registry shape tests: function order, result types, unknown-argument
//! rejection, `registry_json` contents, and result field order vs docs.

use std::fs;
use std::path::PathBuf;

use plskit_bind::{call, registry, registry_json, result_type, result_types, Record, Value};

const API_ORDER: [&str; 20] = [
    "preprocess",
    "pls1_fit",
    "pls1_predict",
    "pls1_find_k_optimal",
    "pls1_find_k_sequence",
    "spls1_fit",
    "spls1_find_keep_optimal",
    "spls1_find_k_optimal",
    "spls1_find_k_sequence",
    "pls3_fit",
    "plssvd_fit",
    "pls3_transform",
    "plssvd_transform",
    "spls3_fit",
    "pls1_confirmatory_test",
    "split_nb_gate",
    "pls1_perm_null",
    "pls3_confirmatory_test",
    "rotate",
    "pls1_rotation_stability",
];

#[test]
fn twenty_functions_in_api_md_order() {
    let names: Vec<&str> = registry().iter().map(|f| f.name).collect();
    assert_eq!(names, API_ORDER);
}

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
    assert_eq!(result_types().len(), 15);
}

#[test]
fn every_function_rejects_an_unknown_argument() {
    for f in registry() {
        let mut r = Record::new();
        r.push("definitely_not_a_param", Value::I64(1)).unwrap();
        let e = call(f.name, r).unwrap_err();
        assert_eq!(e.code, "invalid_argument", "{}", f.name);
    }
}

#[test]
fn registry_json_parses_and_carries_the_signature_details() {
    let v: serde_json::Value = serde_json::from_str(&registry_json()).expect("valid JSON");
    assert_eq!(v["engine_version"], plskit::version());
    assert_eq!(v["functions"].as_array().unwrap().len(), 20);
    assert_eq!(v["error_codes"].as_array().unwrap().len(), 17);
    let ct = v["functions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == "pls1_confirmatory_test")
        .unwrap();
    let test_method = ct["params"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "test_method")
        .unwrap();
    assert_eq!(test_method["required"], true);
    assert_eq!(test_method["keyword_only"], true);
    let k = ct["params"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "k")
        .unwrap();
    assert_eq!(k["default"], 1);
    assert_eq!(k["keyword_only"], false);
    let types = v["result_types"].as_array().unwrap();
    assert_eq!(types.len(), 15);
    assert_eq!(
        types.iter().find(|t| t["name"] == "CIScalar").unwrap()["r_class"],
        "ci_scalar"
    );
}

/// Parses `## Type` section headers (backtick-quoted name) and their
/// `| field |` table rows (backtick-quoted name), first mention wins.
fn documented_fields() -> Vec<(String, Vec<String>)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../_docs/python/results.md");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let ident = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    let mut current: Option<(String, Vec<String>)> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("## ") {
            if let Some(done) = current.take() {
                out.push(done);
            }
            if let Some(name) = rest.strip_prefix('`').and_then(|r| r.split('`').next()) {
                if ident(name) {
                    current = Some((name.to_owned(), Vec::new()));
                }
            }
            continue;
        }
        if let Some((_, fields)) = current.as_mut() {
            if let Some(rest) = line.strip_prefix("| `") {
                let name = rest.split('`').next().unwrap_or("");
                if ident(name) && !fields.iter().any(|f| f == name) {
                    fields.push(name.to_owned());
                }
            }
        }
    }
    if let Some(done) = current.take() {
        out.push(done);
    }
    out
}

#[test]
fn result_fields_follow_results_md_order() {
    let docs = documented_fields();
    let documented: Vec<&str> = docs.iter().map(|(n, _)| n.as_str()).collect();
    for t in result_types() {
        let (_, fields) = docs.iter().find(|(n, _)| n == t.name).unwrap_or_else(|| {
            panic!(
                "{} has no section in results.md (documented: {documented:?})",
                t.name
            )
        });
        let ours: Vec<&str> = t.fields.iter().map(|f| f.name).collect();
        assert_eq!(
            ours, *fields,
            "{} field order differs from results.md",
            t.name
        );
    }
}
