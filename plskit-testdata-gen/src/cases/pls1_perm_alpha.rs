//! `pls1_perm_alpha` fixture cases.

use crate::cases::{
    CasePaths, faer_mat_to_array, i64_vec, manifest_case, ndarray_to_faer_mat, scalar_i64,
    synth_data,
};
use crate::manifest::Case;
use crate::npz::NpzWriter;
use anyhow::Result;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};
use plskit::pls1_perm_alpha;
use std::path::Path;

/// Case: `pls1_perm_alpha` at `k = 1` on the Gram matrix of the standardized
/// `synth_data(40, 60, 2, 4.0, 42)` design. `p = 60` keeps `alpha` near
/// `1e-2`, where the array tolerance binds. `Y` holds four outcomes: `y`,
/// `y` with its rows reversed, an all-zero column (the zero model) and `y`
/// rotated by 7 rows. No column is unresolved: a gate decision at a band
/// edge can differ across hosts.
///
/// Inputs: `gram` `(40, 40)`, `Y` `(40, 4)`.
/// Outputs: `alpha`, `k_used`, `resolved` (0/1), `k`, `n_features`.
///
/// # Errors
/// Returns an error if the fixture files cannot be written, if
/// `pls1_perm_alpha` fails, or if the columns do not resolve as described.
#[allow(clippy::many_single_char_names)]
pub fn n40_p60_m4_k1(root: &Path) -> Result<Case> {
    let name = "pls1_perm_alpha_n40_p60_m4_k1";
    let function = "pls1_perm_alpha";
    let paths = CasePaths::build(root, function, name)?;
    let (n, p) = (40_usize, 60_usize);
    let (x, y) = synth_data(n, p, 2, 4.0, 42);
    let (xs, _, _) = plskit::linalg::standardize(ndarray_to_faer_mat(&x).as_ref());
    let mut gram = Mat::<f64>::zeros(n, n);
    matmul(
        gram.as_mut(),
        Accum::Replace,
        xs.as_ref(),
        xs.as_ref().transpose(),
        1.0,
        Par::Seq,
    );
    let outcomes = Mat::<f64>::from_fn(n, 4, |i, j| match j {
        0 => y[i],
        1 => y[n - 1 - i],
        2 => 0.0,
        _ => y[(i + 7) % n],
    });
    {
        let mut w = NpzWriter::create(&paths.abs_inputs)?;
        w.add_f64("gram", &faer_mat_to_array(&gram))?;
        w.add_f64("Y", &faer_mat_to_array(&outcomes))?;
        w.finish()?;
    }

    let r = pls1_perm_alpha(gram.as_ref(), outcomes.as_ref(), 1, p)?;
    anyhow::ensure!(
        r.resolved == [true; 4] && r.k_used == [1, 1, 0, 1],
        "{name}: expected three resolved columns and one zero model, got resolved {:?}, k_used {:?}",
        r.resolved,
        r.k_used
    );
    {
        let mut w = NpzWriter::create(&paths.abs_outputs)?;
        w.add_f64("alpha", &faer_mat_to_array(&r.alpha))?;
        let k_used = r
            .k_used
            .iter()
            .map(|&v| i64::try_from(v))
            .collect::<Result<Vec<_>, _>>()?;
        w.add_i64("k_used", &i64_vec(k_used))?;
        w.add_i64(
            "resolved",
            &i64_vec(r.resolved.iter().map(|&b| i64::from(b)).collect()),
        )?;
        w.add_i64("k", &scalar_i64(i64::try_from(r.k)?))?;
        w.add_i64("n_features", &scalar_i64(i64::try_from(r.n_features)?))?;
        w.finish()?;
    }

    manifest_case(
        root,
        name,
        function,
        paths.rel_inputs,
        paths.rel_outputs,
        serde_json::json!({ "k": 1, "n_features": p }),
    )
}
