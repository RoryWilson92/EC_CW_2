pub(crate) const CHARS: &[char] = &['0', '1']; // possible alleles
pub(crate) const N_DEMES: usize = 100; // number of populations
pub(crate) const POP_SIZE: usize = 10; // size of each population
pub(crate) const N: usize = 100; // genotype length - O: 400
pub(crate) const T: usize = 2; // number of peaks per sub-function
pub(crate) const W1: usize = 1; // value of first tgt string
pub(crate) const W2: usize = 10; // value of second tgt string
pub(crate) const B: usize = 10; // number of blocks per genotype - O: 20
pub(crate) const K: usize = 10; // size of each block - O: 20
pub(crate) const T1: &str = "1010101010"; // tgt string 1
pub(crate) const T2: &str = "1111111111"; // tgt string 2
pub(crate) const TARGETS: [&str; 2] = [T1, T2]; // list of target strings
pub(crate) const TGT_SCORES: [usize; 2] = [W1, W2]; // list of target scores
pub(crate) const MUTATION_RATE: f64 = 1.0 / (N as f64); // mutation rate
pub(crate) const GENERATIONS: usize = 300; // number of generations
pub(crate) const NEW: usize = POP_SIZE / 2; // number of new individuals to generate in island demes
