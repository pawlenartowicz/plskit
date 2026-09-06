//! NPZ writer wrapper and content-hash helper.

use anyhow::Result;
use ndarray::ArrayD;
use ndarray_npy::NpzWriter as InnerWriter;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufWriter, Read};
use std::path::Path;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, DateTime};

/// Thin wrapper around [`ndarray_npy::NpzWriter`] for writing testdata `.npz` files.
pub struct NpzWriter {
    inner: InnerWriter<BufWriter<File>>,
}

impl NpzWriter {
    /// Create (or overwrite) the `.npz` file at `path`.
    ///
    /// The written bytes are reproducible across runs: the manifest records a
    /// SHA-256 of the whole `.npz` container, so anything varying run-to-run
    /// would make the hash useless as a change detector.
    ///
    /// `.npz` is a ZIP, and each entry carries a DOS last-modified stamp.
    /// `SimpleFileOptions::default()` fills that stamp from the wall clock when
    /// the `zip` crate is compiled with its `time` feature — which happens here
    /// through Cargo feature unification, since another crate in the workspace
    /// depends on `zip` with default features. DOS stamps have two-second
    /// granularity, so two writes seconds apart produce different bytes. Pin the
    /// stamp to the ZIP epoch (1980-01-01), which is also what the `zip` crate
    /// writes when its `time` feature is off, so hashes match either build.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created.
    pub fn create(path: &Path) -> Result<Self> {
        let f = File::create(path)?;
        let options = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Stored)
            .last_modified_time(DateTime::default());
        Ok(Self {
            inner: InnerWriter::new_with_options(BufWriter::new(f), options),
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
