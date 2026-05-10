//! NPZ writer wrapper and content-hash helper.

use anyhow::Result;
use ndarray::ArrayD;
use ndarray_npy::NpzWriter as InnerWriter;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufWriter, Read};
use std::path::Path;

/// Thin wrapper around [`ndarray_npy::NpzWriter`] for writing testdata `.npz` files.
pub struct NpzWriter {
    inner: InnerWriter<BufWriter<File>>,
}

impl NpzWriter {
    /// Create (or overwrite) the `.npz` file at `path`.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created.
    pub fn create(path: &Path) -> Result<Self> {
        let f = File::create(path)?;
        Ok(Self {
            inner: InnerWriter::new(BufWriter::new(f)),
        })
    }

    /// Append a named `f64` array.
    ///
    /// # Errors
    /// Returns an error if serialisation or ZIP writing fails.
    pub fn add_f64(&mut self, name: &str, arr: &ArrayD<f64>) -> Result<()> {
        self.inner.add_array(name, arr)?;
        Ok(())
    }

    /// Append a named `i64` array.
    ///
    /// # Errors
    /// Returns an error if serialisation or ZIP writing fails.
    pub fn add_i64(&mut self, name: &str, arr: &ArrayD<i64>) -> Result<()> {
        self.inner.add_array(name, arr)?;
        Ok(())
    }

    /// Append a named byte array holding the UTF-8 encoding of `value`.
    ///
    /// Numpy will see a 1-D `uint8` array. Wrapper-side readers treat it as
    /// raw bytes. No attempt is made to emit a numpy unicode dtype.
    ///
    /// # Errors
    /// Returns an error if serialisation or ZIP writing fails.
    pub fn add_string(&mut self, name: &str, value: &str) -> Result<()> {
        let bytes = ndarray::Array::from(value.as_bytes().to_vec()).into_dyn();
        self.inner.add_array(name, &bytes)?;
        Ok(())
    }

    /// Flush, finalise the ZIP central directory, and close the file.
    ///
    /// # Errors
    /// Returns an error if flushing or closing the underlying file fails.
    pub fn finish(self) -> Result<()> {
        self.inner.finish()?;
        Ok(())
    }
}

/// Return the lowercase hex-encoded SHA-256 digest of the file at `path`.
///
/// # Errors
/// Returns an error if the file cannot be opened or read.
pub fn sha256_of_file(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut buf = vec![0_u8; 64 * 1024];
    let mut hasher = Sha256::new();
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        write!(&mut hex, "{b:02x}").expect("writing to String never fails");
    }
    Ok(hex)
}
