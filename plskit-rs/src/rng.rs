//! RNG path for plskit: tagged draw streams on commonstats' Philox.

use commonstats::{CommonStatsRng, Scheme, gen_resample_indices};

use crate::error::{PlsKitError, PlsKitResult};

/// Resolve `Option<u64>` to a concrete seed; on None, draw 64 bits from OS entropy.
///
/// # Errors
///
/// Returns `PlsKitError::InvalidArgument` when `n_rows > u32::MAX` (the random
/// draws index rows as `u32`), checked first, and `PlsKitError::Internal` if the
/// OS cannot provide entropy (a getrandom failure, which points to a severely
/// misconfigured or sandboxed environment).
pub(crate) fn resolve_seed(seed: Option<u64>, n_rows: usize) -> PlsKitResult<u64> {
    if n_rows > u32::MAX as usize {
        return Err(PlsKitError::InvalidArgument(format!(
            "n_rows = {n_rows} is above u32::MAX: the random draws index rows as u32"
        )));
    }
    match seed {
        Some(v) => Ok(v),
        None => draw_os_seed(),
    }
}

/// Draw 64 bits of OS entropy via getrandom. Used when caller passes seed=None.
fn draw_os_seed() -> PlsKitResult<u64> {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf)
        .map_err(|e| PlsKitError::Internal(format!("OS entropy unavailable: {e}")))?;
    Ok(u64::from_le_bytes(buf))
}

// Stream tags of plskit's draws on commonstats' Philox, each eight ASCII bytes
// read big-endian, as commonstats' own `STREAM_TAG_*` constants are.
//
// `PICKS`: the leftover picks of balanced split `j` (id `j`). `DSIGN`: the
// balanced-split design (id 0). `SPLIT`: the one train/test split of the `e`
// test (id 0). `FOLDS`: one fold shuffle (id 0, or the repeat index of
// repeated CV). `SUBSM`: CI replicate `r`, subsample and half split (id `r`).
// `BOOTS`: CI replicate `r`, bootstrap (id `r`). `FACTR`: parallel-analysis
// shuffle `b` (id `b`). `STEPS`: the test seed of sequence step `k` (id `k`).
// `SEQNC`: the seed of the k sequence (id 0).
//
// `CommonStatsRng::new_tagged(seed, id, tag)` opens Philox stream `id ^ tag`,
// so two keys share words exactly when their `id ^ tag` agree. The plskit tags
// differ pairwise at or above bit 24, and from commonstats' tags above bit 56,
// so streams stay apart while every id under a plskit tag stays below `2^24`.
// The null columns run on commonstats' `RESAMPLE` stream, whose ids have no
// bound. No option comes near the bound: inside one call the closest tags
// that share a seed (`TAG_SUBSM` and `TAG_SPLIT`) meet only at an id of about
// 8.5e7, and each id is a model fit.
pub(crate) const TAG_PICKS: u64 = u64::from_be_bytes(*b"PLSPICKS");
pub(crate) const TAG_DSIGN: u64 = u64::from_be_bytes(*b"PLSDSIGN");
pub(crate) const TAG_SPLIT: u64 = u64::from_be_bytes(*b"PLSSPLIT");
pub(crate) const TAG_FOLDS: u64 = u64::from_be_bytes(*b"PLSFOLDS");
pub(crate) const TAG_SUBSM: u64 = u64::from_be_bytes(*b"PLSSUBSM");
pub(crate) const TAG_BOOTS: u64 = u64::from_be_bytes(*b"PLSBOOTS");
pub(crate) const TAG_FACTR: u64 = u64::from_be_bytes(*b"PLSFACTR");
pub(crate) const TAG_STEPS: u64 = u64::from_be_bytes(*b"PLSSTEPS");
pub(crate) const TAG_SEQNC: u64 = u64::from_be_bytes(*b"PLSSEQNC");

/// Fisher–Yates shuffle of `v` in place: for `i` from `len − 1` down to 1, swap
/// `i` with `j = rng.bounded(i + 1)`.
///
/// This is commonstats' `Scheme::Permutation` algorithm in
/// `gen_resample_indices`, so on `0..n` it gives that function's indices for the
/// same stream; `shuffle_is_commonstats_permutation` holds the two together
/// (change together). Requires `v.len() <= u32::MAX`, which `resolve_seed`'s row
/// check guarantees.
pub(crate) fn shuffle<T>(rng: &mut CommonStatsRng, v: &mut [T]) {
    for i in (1..v.len()).rev() {
        let bound = u32::try_from(i + 1).expect("shuffle: length above u32::MAX");
        v.swap(i, rng.bounded(bound) as usize);
    }
}

/// A whole seed for a step that needs one of its own: words 0 and 1 of
/// `new_tagged(seed, id, tag)`, high word first.
pub(crate) fn child_seed(seed: u64, id: u64, tag: u64) -> u64 {
    let mut rng = CommonStatsRng::new_tagged(seed, id, tag);
    let hi = u64::from(rng.next_u32());
    (hi << 32) | u64::from(rng.next_u32())
}

/// Null column `c` (`c >= 1`) of every permutation null over `n` rows at run seed
/// `seed`: row `i` of the null column is row `out[i]` of the outcome. Equal to
/// neurostats' `draw_permutation(n, c, seed)` for `c >= 1`; column 0 is the
/// observed outcome and is never drawn.
pub(crate) fn null_permutation(n: usize, c: usize, seed: u64) -> Vec<u32> {
    debug_assert!(c >= 1, "null column 0 is the observed outcome");
    let mut out = vec![0; n];
    gen_resample_indices(Scheme::Permutation, n, c as u64, seed, &mut out)
        .expect("resolve_seed's row check keeps n <= u32::MAX");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use commonstats::{
        STREAM_TAG_RESAMPLE, STREAM_TAG_RLRT, STREAM_TAG_SIGNFLIP, STREAM_TAG_SIMULATE,
    };

    const PLSKIT_TAGS: [(u64, &[u8; 8]); 9] = [
        (TAG_PICKS, b"PLSPICKS"),
        (TAG_DSIGN, b"PLSDSIGN"),
        (TAG_SPLIT, b"PLSSPLIT"),
        (TAG_FOLDS, b"PLSFOLDS"),
        (TAG_SUBSM, b"PLSSUBSM"),
        (TAG_BOOTS, b"PLSBOOTS"),
        (TAG_FACTR, b"PLSFACTR"),
        (TAG_STEPS, b"PLSSTEPS"),
        (TAG_SEQNC, b"PLSSEQNC"),
    ];

    #[test]
    fn resolve_seed_none_draws_fresh_entropy() {
        let a = resolve_seed(None, 10).unwrap();
        let b = resolve_seed(None, 10).unwrap();
        // Two OS draws collide with probability 2^-64.
        assert_ne!(a, b, "seed=None must draw OS entropy, not a constant");
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn resolve_seed_rejects_rows_past_u32() {
        assert!(matches!(
            resolve_seed(Some(1), u32::MAX as usize + 1),
            Err(PlsKitError::InvalidArgument(_))
        ));
        assert_eq!(resolve_seed(Some(1), u32::MAX as usize).unwrap(), 1);
    }

    // Recorded from commonstats 0.3.0: gen_resample_indices(Scheme::Permutation, 10, c, 2809).
    const PERMS_N10_S2809: [[u32; 10]; 3] = [
        [4, 1, 0, 6, 5, 7, 2, 3, 9, 8],
        [7, 9, 4, 5, 1, 2, 6, 3, 8, 0],
        [1, 2, 4, 8, 5, 7, 9, 0, 3, 6],
    ];

    #[test]
    fn null_permutations_are_commonstats_draws() {
        for (i, want) in PERMS_N10_S2809.iter().enumerate() {
            assert_eq!(
                null_permutation(10, i + 1, 2809),
                want.to_vec(),
                "c = {}",
                i + 1
            );
        }
    }

    // Recorded from commonstats 0.3.0: words 0 and 1 of new_tagged(42, 0, tag), high first.
    #[test]
    fn child_seed_pins_each_tag_stream() {
        for (tag, want) in [
            (TAG_PICKS, 0x8d97_bdad_d35e_e069_u64),
            (TAG_DSIGN, 0x8b1c_f348_f384_dfac),
            (TAG_SPLIT, 0xce66_3d92_2307_29b1),
            (TAG_FOLDS, 0x15d7_0d12_4a0d_0fed),
            (TAG_SUBSM, 0x9731_9039_cf86_2ff2),
            (TAG_BOOTS, 0x0be7_ae72_3c8a_b6d7),
            (TAG_FACTR, 0xfab7_45cd_fd65_5f6d),
            (TAG_STEPS, 0x9558_dece_6678_f8da),
            (TAG_SEQNC, 0xce0b_e658_daae_5178),
        ] {
            assert_eq!(child_seed(42, 0, tag), want, "{:?}", tag.to_be_bytes());
        }
    }

    #[test]
    fn stream_tags_spell_their_names() {
        for (tag, spelling) in PLSKIT_TAGS {
            assert_eq!(&tag.to_be_bytes(), spelling);
        }
    }

    #[test]
    fn stream_tags_are_far_apart() {
        for (i, (a, an)) in PLSKIT_TAGS.iter().enumerate() {
            for (b, bn) in &PLSKIT_TAGS[i + 1..] {
                assert!(a ^ b >= 1 << 24, "{an:?} vs {bn:?}: {:#x}", a ^ b);
            }
            for b in [
                STREAM_TAG_RESAMPLE,
                STREAM_TAG_SIGNFLIP,
                STREAM_TAG_RLRT,
                STREAM_TAG_SIMULATE,
            ] {
                assert!(a ^ b >= 1 << 56, "{an:?} vs {:?}", b.to_be_bytes());
            }
        }
    }

    #[test]
    fn shuffle_is_commonstats_permutation() {
        for n in [0usize, 1, 2, 10, 257] {
            for c in 1..=3u64 {
                for s in [0, 2809, u64::MAX] {
                    let mut v: Vec<u32> = (0..u32::try_from(n).unwrap()).collect();
                    shuffle(
                        &mut CommonStatsRng::new_tagged(s, c, STREAM_TAG_RESAMPLE),
                        &mut v,
                    );
                    let mut want = vec![0u32; n];
                    gen_resample_indices(Scheme::Permutation, n, c, s, &mut want).unwrap();
                    assert_eq!(v, want, "n = {n}, c = {c}, s = {s}");
                }
            }
        }
    }
}
