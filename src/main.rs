use rand::distributions::Slice;
use rand::seq::IndexedRandom;
use rand::Rng;

const CHARS: &str = "01";

fn get_chars() -> [char; 2] {
    CHARS.chars().collect::<Vec<char>>().try_into().unwrap()
}

fn generate_individual(length: usize) -> (String, f64) {
    let chars = &mut get_chars().clone();
    let char_dist = Slice::new(chars).unwrap();
    let individual: String = rand::thread_rng()
        .sample_iter(char_dist)
        .take(length)
        .collect();
    let fit = fitness(&individual);
    (individual, fit)
}

fn split_into_ns(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out: Vec<String> = Vec::new();
    for i in 0..(chars.len() / n) {
        out.push(chars[i * n..(i * n + n)].iter().collect::<String>())
    }
    out
}

fn fitness(individual: &str) -> f64 {
    let mut fitness = 0.0;
    let gis = split_into_ns(individual, K);
    for gi in gis {
        for t in 0..T {
            let hamming = gi
                .chars()
                .zip(TARGETS[t].chars())
                .filter(|(a, b)| a == b)
                .count();
            fitness += if hamming == 0 {
                TGT_SCORES[t] as f64
            } else {
                1.0 / ((1 + hamming) as f64)
            }
        }
    }
    fitness
}

fn mutate(individual: String) -> String {
    let mut new: Vec<char> = individual.chars().collect();
    for i in 0..new.len() {
        if rand::thread_rng().gen_range(0.0..1.0) <= MUTATION_RATE {
            new.splice(i..i + 1, [if new[i] == '0' { '1' } else { '0' }]);
        }
    }
    new.iter().collect()
}

fn one_point_crossover(individual: &(String, f64), population: &[(String, f64)]) -> (String, f64) {
    let mut rng = rand::thread_rng();
    let c = rng.gen_range(0..N);
    let xi = &individual.0;
    let xj = &population.choose(&mut rng).unwrap().0;
    let y1 = format!("{}{}", &xi[0..c], &xj[c..N]);
    let y2 = format!("{}{}", &xj[0..c], &xi[c..N]);
    let y = mutate(if fitness(&y1) > fitness(&y2) { y1 } else { y2 });
    let fit = fitness(&y);
    (y, fit)
}

const N_DEMES: usize = 100; // number of populations
const POP_SIZE: usize = 10; // size of each population
const N: usize = 400; // genotype length
const T: usize = 2; // number of peaks per sub-function
const W1: usize = 1; // value of first tgt string
const W2: usize = 10; // value of second tgt string
// const B: usize = 20; // number of blocks per genotype
const K: usize = 20; // size of each block
const T1: &str = "10101010101010101010"; // tgt string 1
const T2: &str = "11111111111111111111"; // tgt string 2
const TARGETS: [&str; 2] = [T1, T2];
const TGT_SCORES: [usize; 2] = [W1, W2];
const MUTATION_RATE: f64 = 1.0 / (N as f64);
const MIGRATION_RATE: f64 = 0.0004;
const GENERATIONS: usize = 100;

fn main() {
    // Initialisations
    let mut demes = vec![vec![generate_individual(N); POP_SIZE]; N_DEMES];
    let mut rng = rand::thread_rng();

    // Main loop
    for _ in 0..GENERATIONS {
        for d in 0..N_DEMES {
            let migrant_deme = &demes[rng.gen_range(0..N_DEMES)].clone();
            let deme = &mut demes[d];
            let mut new_individuals: Vec<(String, f64)> = Vec::new();
            deme.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (i, x) in deme.iter().enumerate() {
                if rng.gen_range(0.0..1.0) <= 1.0 / (i as f64) {
                    if rng.gen_range(0.0..1.0) <= MIGRATION_RATE {
                        let migrant = migrant_deme.choose(&mut rng).unwrap();
                        new_individuals.push(one_point_crossover(migrant, deme));
                    } else {
                        new_individuals.push(one_point_crossover(x, deme))
                    }
                }
            }
            deme.splice(
                (POP_SIZE - new_individuals.len())..POP_SIZE,
                new_individuals,
            );
        }
    }
}
