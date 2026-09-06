//! Integration tests for [`plskit_testdata_gen::npz`]: write and hash an NPZ file.
//!
//! TODO: A true roundtrip test (write → read back) requires an `NpzReader`;
//! that is future scope once a suitable reader dependency is available.

use ndarray::array;
use plskit_testdata_gen::npz::{sha256_of_file, NpzWriter};
use std::thread::sleep;
use std::time::Duration;
use tempfile::tempdir;

#[test]
fn write_and_hash_npz() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("a.npz");
    let mut w = NpzWriter::create(&path).unwrap();
    w.add_f64("X", &array![[1.0, 2.0], [3.0, 4.0]].into_dyn())
        .unwrap();
    w.add_f64("y", &array![5.0_f64, 6.0].into_dyn()).unwrap();
    w.finish().unwrap();
    let h = sha256_of_file(&path).unwrap();
    assert_eq!(h.len(), 64);
}

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
        (std::fs::read(&path).unwrap(), sha256_of_file(&path).unwrap())
    };
    let (bytes_a, hash_a) = write("a.npz");
    sleep(Duration::from_millis(2100));
    let (bytes_b, hash_b) = write("b.npz");
    assert_eq!(bytes_a, bytes_b);
    assert_eq!(hash_a, hash_b);
}
