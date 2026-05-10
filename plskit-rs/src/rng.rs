//! RNG path for plskit. ChaCha8Rng + pre-computed child seeds.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Type alias for the canonical RNG used throughout plskit.
pub type Rng = ChaCha8Rng;

/// Resolve `Option<u64>` to a concrete seed; on None, draw 64 bits from OS entropy.
/// Returns `(seed_used, rng_seeded_from_it)`.
#[must_use]
pub fn resolve_seed(seed: Option<u64>) -> (u64, Rng) {
    let s = seed.unwrap_or_else(draw_os_seed);
    (s, ChaCha8Rng::seed_from_u64(s))
}

/// Draw 64 bits of OS entropy via getrandom. Used when caller passes seed=None.
///
/// # Panics
///
/// Panics if the OS cannot provide entropy (getrandom failure — extremely rare).
#[must_use]
pub fn draw_os_seed() -> u64 {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf).expect("getrandom failed");
    u64::from_le_bytes(buf)
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
    fn child_seeds_are_deterministic_for_fixed_parent_seed() {
        let (_, mut rng_a) = resolve_seed(Some(42));
        let (_, mut rng_b) = resolve_seed(Some(42));
        let a = child_seeds(&mut rng_a, 100);
        let b = child_seeds(&mut rng_b, 100);
        assert_eq!(a, b);
    }

    #[test]
    fn resolve_seed_passes_through_caller_value() {
        let (s, _) = resolve_seed(Some(12345));
        assert_eq!(s, 12345);
    }

    #[test]
    fn resolve_seed_none_returns_nonzero_seed() {
        let (s, _) = resolve_seed(None);
        // Probability of OS entropy returning exactly 0 is 2^-64; flag if it happens.
        assert_ne!(s, 0);
    }

    #[test]
    fn child_rng_reproducible_from_seed() {
        use rand::Rng;
        let mut a = child_rng(7);
        let mut b = child_rng(7);
        let mut va = vec![0u64; 10];
        let mut vb = vec![0u64; 10];
        for i in 0..10 {
            va[i] = a.next_u64();
            vb[i] = b.next_u64();
        }
        assert_eq!(va, vb);
    }
}
