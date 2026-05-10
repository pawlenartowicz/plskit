//! Integration tests for [`plskit_testdata_gen::npz`]: write and hash an NPZ file.
//!
//! TODO: A true roundtrip test (write → read back) requires an `NpzReader`;
//! that is future scope once a suitable reader dependency is available.

use ndarray::array;
use plskit_testdata_gen::npz::{sha256_of_file, NpzWriter};
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
