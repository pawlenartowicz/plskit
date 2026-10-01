//! [`plskit_testdata_gen::npz`]: the writer's bytes do not depend on the clock.

use ndarray::array;
use plskit_testdata_gen::npz::{sha256_of_file, NpzWriter};
use std::thread::sleep;
use std::time::Duration;
use tempfile::tempdir;

/// The DOS timestamps inside a `.npz` have two-second granularity, so a writer
/// that stamped the wall clock would produce different bytes for these two
/// files. The sleep guarantees the writes straddle a boundary.
#[test]
fn npz_bytes_are_identical_across_a_two_second_boundary() {
    let dir = tempdir().unwrap();
    let write = |name: &str| {
        let path = dir.path().join(name);
        let mut w = NpzWriter::create(&path).unwrap();
        w.add_f64("X", &array![[1.0, 2.0], [3.0, 4.0]].into_dyn())
            .unwrap();
        w.finish().unwrap();
        (
            std::fs::read(&path).unwrap(),
            sha256_of_file(&path).unwrap(),
        )
    };
    let (bytes_a, hash_a) = write("a.npz");
    sleep(Duration::from_millis(2100));
    let (bytes_b, hash_b) = write("b.npz");
    assert_eq!(bytes_a, bytes_b);
    assert_eq!(hash_a, hash_b);
}
