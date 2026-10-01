//! [`plskit_testdata_gen::settle`]: a regeneration keeps a committed fixture
//! only when the fresh one matches it under the corpus tolerance.

use ndarray::{array, ArrayD};
use plskit_testdata_gen::npz::NpzWriter;
use plskit_testdata_gen::settle::{equivalent, settle_cases, settle_file, Outcome, Tolerance};
use std::path::Path;
use tempfile::tempdir;

/// One fixture: a 0-D scalar, a 1-D array, an `i64` count and a string.
struct Fx {
    scalar: f64,
    arr: [f64; 3],
    count: i64,
    method: &'static str,
}

const BASE: Fx = Fx {
    scalar: 0.5,
    arr: [1.0, f64::NAN, f64::INFINITY],
    count: 7,
    method: "split_exact",
};

fn write(path: &Path, fx: &Fx) {
    let mut w = NpzWriter::create(path).unwrap();
    w.add_f64("pvalue", &ArrayD::from_elem(vec![], fx.scalar))
        .unwrap();
    w.add_f64("coef", &array![fx.arr[0], fx.arr[1], fx.arr[2]].into_dyn())
        .unwrap();
    w.add_i64("k_used", &ArrayD::from_elem(vec![], fx.count))
        .unwrap();
    w.add_string("method", fx.method).unwrap();
    w.finish().unwrap();
}

/// A committed scalar the size of the score statistic (`‖X'y‖²` ≈ 6126),
/// where one ulp (9.1e-13) nearly fills `atol_scalar`.
const LARGE: Fx = Fx {
    scalar: 6_125.560_061_283_051,
    ..BASE
};

/// A committed array entry of `1e6`, whose ulp (1.2e-10) exceeds `atol_array`.
const LARGE_ARRAY: Fx = Fx {
    arr: [1e6, f64::NAN, f64::INFINITY],
    ..BASE
};

/// `x` moved up by `n` ulps (`x` positive and finite).
fn ulps_up(x: f64, n: u64) -> f64 {
    f64::from_bits(x.to_bits() + n)
}

#[test]
fn equivalence_follows_the_corpus_tolerance() {
    let rows: [(&str, Fx, bool); 9] = [
        (
            "scalar inside 1e-12",
            Fx {
                scalar: 0.5 + 5e-13,
                ..BASE
            },
            true,
        ),
        (
            "scalar beyond 1e-12",
            Fx {
                scalar: 0.5 + 5e-12,
                ..BASE
            },
            false,
        ),
        (
            "array inside 1e-10",
            Fx {
                arr: [1.0 + 5e-11, f64::NAN, f64::INFINITY],
                ..BASE
            },
            true,
        ),
        (
            "array beyond 1e-10",
            Fx {
                arr: [1.0 + 5e-10, f64::NAN, f64::INFINITY],
                ..BASE
            },
            false,
        ),
        (
            "NaN mask moved",
            Fx {
                arr: [1.0, 0.0, f64::INFINITY],
                ..BASE
            },
            false,
        ),
        // An infinity is equal only to itself: `rtol·|inf|` must not widen
        // the tolerance to infinity.
        (
            "finite entry became +inf",
            Fx {
                arr: [f64::INFINITY, f64::NAN, f64::INFINITY],
                ..BASE
            },
            false,
        ),
        (
            "+inf became -inf",
            Fx {
                arr: [1.0, f64::NAN, f64::NEG_INFINITY],
                ..BASE
            },
            false,
        ),
        ("integer differs", Fx { count: 6, ..BASE }, false),
        (
            "string differs",
            Fx {
                method: "split_nb",
                ..BASE
            },
            false,
        ),
    ];
    check_equivalence(
        rows.into_iter()
            .map(|(label, fx, expect)| (label, BASE, fx, expect)),
    );
}

/// The relative term: `atol + rtol·|e|` absorbs a few ulps of cross-host
/// drift in a large value (2 ulps of 6126 exceed `atol_scalar` alone), and
/// still rejects a real move.
#[test]
fn equivalence_scales_with_the_committed_value() {
    let relative: [(&str, Fx, Fx, bool); 3] = [
        (
            "large scalar 2 ulps off",
            LARGE,
            Fx {
                scalar: ulps_up(LARGE.scalar, 2),
                ..LARGE
            },
            true,
        ),
        (
            "large scalar beyond atol + rtol·|e|",
            LARGE,
            Fx {
                scalar: LARGE.scalar + 1e-10,
                ..LARGE
            },
            false,
        ),
        (
            "large array entry inside atol + rtol·|e|",
            LARGE_ARRAY,
            Fx {
                arr: [1e6 + 5e-9, f64::NAN, f64::INFINITY],
                ..LARGE_ARRAY
            },
            true,
        ),
    ];
    check_equivalence(relative);
}

/// `equivalent` under the default tolerance on each (label, committed,
/// staged, expected) row.
fn check_equivalence<'a>(rows: impl IntoIterator<Item = (&'a str, Fx, Fx, bool)>) {
    let dir = tempdir().unwrap();
    let committed = dir.path().join("committed.npz");
    let staged = dir.path().join("staged.npz");
    for (label, base, fx, expect) in rows {
        write(&committed, &base);
        write(&staged, &fx);
        assert_eq!(
            equivalent(&committed, &staged, Tolerance::DEFAULT).unwrap(),
            expect,
            "{label}"
        );
    }
}

#[test]
fn settle_file_writes_only_new_or_moved_fixtures() {
    let staged_root = tempdir().unwrap();
    let committed_root = tempdir().unwrap();
    let rel = "outputs/f/case.npz";
    std::fs::create_dir_all(staged_root.path().join("outputs/f")).unwrap();
    let staged = staged_root.path().join(rel);
    let committed = committed_root.path().join(rel);
    let settle = || {
        settle_file(
            staged_root.path(),
            committed_root.path(),
            rel,
            Tolerance::DEFAULT,
        )
        .unwrap()
    };

    write(&staged, &BASE);
    assert_eq!(settle(), Outcome::Added);
    assert_eq!(
        std::fs::read(&committed).unwrap(),
        std::fs::read(&staged).unwrap()
    );

    // Drift inside tolerance: the committed bytes stay.
    let before = std::fs::read(&committed).unwrap();
    write(
        &staged,
        &Fx {
            scalar: 0.5 + 1e-13,
            ..BASE
        },
    );
    assert_eq!(settle(), Outcome::Kept);
    assert_eq!(std::fs::read(&committed).unwrap(), before);

    write(
        &staged,
        &Fx {
            scalar: 0.75,
            ..BASE
        },
    );
    assert_eq!(settle(), Outcome::Replaced);
    assert_eq!(
        std::fs::read(&committed).unwrap(),
        std::fs::read(&staged).unwrap()
    );
}

/// An input settles with its case's output: a within-tolerance input goes in
/// when its output moved, including for a case whose own output was kept.
#[test]
fn settle_cases_moves_an_input_with_its_output() {
    let staged_root = tempdir().unwrap();
    let committed_root = tempdir().unwrap();
    let (s, c) = (staged_root.path(), committed_root.path());
    for root in [s, c] {
        std::fs::create_dir_all(root.join("inputs")).unwrap();
        std::fs::create_dir_all(root.join("outputs")).unwrap();
    }
    let drifted = Fx {
        scalar: 0.5 + 1e-13,
        ..BASE
    };
    let moved = Fx {
        scalar: 0.75,
        ..BASE
    };
    // (path, committed, staged): both inputs drift inside tolerance.
    let files = [
        ("inputs/shared.npz", &BASE, &drifted),
        ("inputs/quiet.npz", &BASE, &drifted),
        ("outputs/moved.npz", &BASE, &moved),
        ("outputs/kept.npz", &BASE, &drifted),
        ("outputs/quiet.npz", &BASE, &drifted),
    ];
    for (rel, committed, staged) in files {
        write(&c.join(rel), committed);
        write(&s.join(rel), staged);
    }
    let tol = Tolerance::DEFAULT;
    let cases = [
        ("inputs/shared.npz", "outputs/moved.npz", tol),
        ("inputs/shared.npz", "outputs/kept.npz", tol),
        ("inputs/quiet.npz", "outputs/quiet.npz", tol),
    ];
    let mut got = settle_cases(s, c, &cases).unwrap();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    let want = [
        ("inputs/quiet.npz", Outcome::Kept),
        ("inputs/shared.npz", Outcome::Replaced),
        ("outputs/kept.npz", Outcome::Kept),
        ("outputs/moved.npz", Outcome::Replaced),
        ("outputs/quiet.npz", Outcome::Kept),
    ];
    let got: Vec<(&str, Outcome)> = got.iter().map(|(r, o)| (r.as_str(), *o)).collect();
    assert_eq!(got, want);
    assert_eq!(
        std::fs::read(c.join("inputs/shared.npz")).unwrap(),
        std::fs::read(s.join("inputs/shared.npz")).unwrap()
    );
}

/// A path that two cases write, or that is both an input and an output, is
/// refused before any file is settled.
#[test]
fn settle_cases_refuses_overlapping_paths() {
    let root = tempdir().unwrap();
    let tol = Tolerance::DEFAULT;
    let shared_output = [
        ("inputs/a.npz", "outputs/x.npz", tol),
        ("inputs/b.npz", "outputs/x.npz", tol),
    ];
    let input_is_output = [
        ("inputs/a.npz", "outputs/x.npz", tol),
        ("outputs/x.npz", "outputs/y.npz", tol),
    ];
    for cases in [&shared_output, &input_is_output] {
        assert!(settle_cases(root.path(), root.path(), cases).is_err());
    }
}
