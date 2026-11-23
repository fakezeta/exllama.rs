//! Small deterministic PRNG (SplitMix64) for sampling, no external deps.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 works with any state, including 0 (the first update adds
        // the gamma constant). Remapping 0 used to make it collide with the
        // gamma value itself.
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() & 0xFFFF_FFFF) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_splitmix64_sequence() {
        let mut r = Rng::new(0);
        let got: Vec<u64> = (0..4).map(|_| r.next_u64()).collect();
        // Standard SplitMix64 with state 0: after adding the gamma constant,
        // the mixing produces these values (independent of any remapping).
        let expected = [
            0xE220_A839_7B1D_CDAF,
            0x6E78_9E6A_A1B9_65F4,
            0x06C4_5D18_8009_454F,
            0xF88B_B8A8_724C_81EC,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn zero_seed_does_not_collide_with_gamma() {
        assert_ne!(
            Rng::new(0).next_u64(),
            Rng::new(0x9E3779B97F4A7C15).next_u64()
        );
    }
}
