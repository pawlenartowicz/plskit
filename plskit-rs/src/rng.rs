//! RNG path for plskit. ChaCha8Rng + pre-computed child seeds.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::{PlsKitError, PlsKitResult};

/// Type alias for the canonical RNG used throughout plskit.
pub type Rng = ChaCha8Rng;

/// Resolve `Option<u64>` to a concrete seed; on None, draw 64 bits from OS entropy.
/// Returns `(seed_used, rng_seeded_from_it)`.
///
/// # Errors
///
/// Returns `PlsKitError::Internal` if the OS cannot provide entropy (getrandom failure —
/// extremely rare; indicates a severely misconfigured or sandboxed environment).
pub(crate) fn resolve_seed(seed: Option<u64>) -> PlsKitResult<(u64, Rng)> {
    let s = match seed {
        Some(v) => v,
        None => draw_os_seed()?,
    };
    Ok((s, ChaCha8Rng::seed_from_u64(s)))
}

/// Draw 64 bits of OS entropy via getrandom. Used when caller passes seed=None.
fn draw_os_seed() -> PlsKitResult<u64> {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf)
        .map_err(|e| PlsKitError::Internal(format!("OS entropy unavailable: {e}")))?;
    Ok(u64::from_le_bytes(buf))
}

/// Pre-compute `n_iterations` child seeds sequentially from `parent`.
/// Storing them in a Vec ensures byte-parity across thread counts:
/// iteration `i` always uses `child_seeds[i]` regardless of which Rayon
/// worker picks it up.
#[must_use]
pub fn child_seeds(parent: &mut Rng, n_iterations: usize) -> Vec<u64> {
    use rand::Rng;
    (0..n_iterations).map(|_| parent.next_u64()).collect()
}

/// Re-seed a fresh `ChaCha8Rng` from one child seed. Use inside Rayon workers.
#[must_use]
pub fn child_rng(seed: u64) -> Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_seed_none_draws_fresh_entropy() {
        let (a, _) = resolve_seed(None).unwrap();
        let (b, _) = resolve_seed(None).unwrap();
        // Two OS draws collide with probability 2^-64.
        assert_ne!(a, b, "seed=None must draw OS entropy, not a constant");
    }
}
