use std::time::Instant;

use rand::distributions::Slice;
use rand::prelude::SliceRandom;
use rand::Rng;

use crate::consts::{
    B, CHARS, GENERATIONS, K, MUTATION_RATE, N, N_DEMES, NEW, POP_SIZE, T, TARGETS, TGT_SCORES,
};

type CrossoverOperator = fn(
    &[(String, f64, [usize; N])],
    Option<&(String, f64, [usize; N])>,
) -> (String, f64, [usize; N]);
type Migrants<'a> = Option<&'a Vec<Vec<(String, f64, [usize; N])>>>;

pub(crate) fn permute(s: &str, permutation: &[usize]) -> String {
    let mut tmp = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    for i in 0..s.len() {
        tmp.push(chars[permutation[i]]);
    }
    tmp.iter().collect::<String>()
}

pub(crate) fn max_fitness_individual(
    population: &[(String, f64, [usize; N])],
) -> &(String, f64, [usize; N]) {
    population
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
}

pub(crate) fn average_fitness(population: &[(String, f64, [usize; N])]) -> f64 {
    population.iter().fold(0.0, |acc, x| acc + x.1) / (population.len() as f64)
}

pub(crate) fn generate_individual(length: usize) -> (String, f64, [usize; N]) {
    let char_dist: Slice<char> = Slice::new(CHARS).unwrap();
    let individual: String = rand::thread_rng()
        .sample_iter(char_dist)
        .take(length)
        .collect();
    let mut tmp = (0..N).collect::<Vec<usize>>();
    tmp.shuffle(&mut rand::thread_rng());
    let linkage = <[usize; N]>::try_from(tmp).unwrap();
    let fit = fitness(&permute(&individual, &linkage));
    (individual, fit, linkage)
}

pub(crate) fn generate_population(pop_size: usize) -> Vec<(String, f64, [usize; N])> {
    let mut pop: Vec<(String, f64, [usize; N])> = Vec::new();
    for _ in 0..pop_size {
        pop.push(generate_individual(N));
    }
    pop
}

pub(crate) fn rank_based_reproduction(
    population: &mut [(String, f64, [usize; N])],
    migrants: Migrants,
    crossover_operator: CrossoverOperator,
) -> (Vec<(String, f64, [usize; N])>, usize) {
    let mut new_individuals: Vec<(String, f64, [usize; N])> = Vec::new();
    population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    match migrants {
        // Case for receiving deme
        Some(demes) => {
            // Take the best individual from each deme and crossover with the receiving deme
            for deme in demes.iter() {
                let migrant = max_fitness_individual(deme);
                new_individuals.push(crossover_operator(population, Some(migrant)));
            }
        }
        // Case for island demes
        None => {
            for _ in 0..NEW {
                new_individuals.push(crossover_operator(population, None));
            }
        }
    }
    let evals = new_individuals.len();
    (new_individuals, evals)
}

pub(crate) fn select_best(
    population: &[(String, f64, [usize; N])],
    new_individuals: &[(String, f64, [usize; N])],
) -> Vec<(String, f64, [usize; N])> {
    let mut new_pop = population.to_vec();
    new_pop.append(&mut new_individuals.to_vec());
    new_pop.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    new_pop
        .iter()
        .take(POP_SIZE)
        .rev()
        .map(|x| x.to_owned())
        .collect::<Vec<(String, f64, [usize; N])>>()
}

pub(crate) fn split_into_ns(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out: Vec<String> = Vec::new();
    for i in 0..(chars.len() / n) {
        out.push(chars[i * n..(i * n + n)].iter().collect::<String>())
    }
    out
}

pub(crate) fn fitness(individual: &str) -> f64 {
    let mut fitness = 0.0;
    let tmp = individual;
    let chars = tmp.chars().collect::<Vec<char>>();
    for i in 0..B {
        let gi = &chars[(i * K)..(i * K + K)];
        for t in 0..T {
            let hamming = gi
                .iter()
                .zip(TARGETS[t].chars())
                .filter(|(a, b)| *a != b)
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

pub(crate) fn mutate(individual: String) -> String {
    let mut rng = rand::thread_rng();
    individual
        .chars()
        .map(|c| {
            if rng.gen_range(0.0..1.0) <= MUTATION_RATE {
                if c == '0' {
                    '1'
                } else {
                    '0'
                }
            } else {
                c
            }
        })
        .collect()
}

pub(crate) fn run_ga(crossover_operator: CrossoverOperator) -> Vec<f64> {
    // Initialise populations and variables
    let mut evaluations = 0;
    let mut fitness_history: Vec<f64> = Vec::new();
    let mut receiving_deme = generate_population(POP_SIZE);
    let mut island_demes: Vec<Vec<(String, f64, [usize; N])>> = Vec::new();
    for _ in 0..N_DEMES {
        island_demes.push(generate_population(POP_SIZE));
    }

    let init_fitness = average_fitness(&receiving_deme);
    println!("Average fitness at initialisation: {}", init_fitness);
    fitness_history.push(init_fitness);

    // Main loop
    let timer = Instant::now();
    for _g in 0..GENERATIONS {
        // Crossover for normal demes
        for deme in island_demes.iter_mut() {
            // Create new individuals proportionate to their rank
            let (new_individuals, new_evals) =
                rank_based_reproduction(deme, None, crossover_operator);
            evaluations += new_evals;

            // Combine new and original individuals and take the best POP_SIZE individuals.
            let new_pop = select_best(deme, &new_individuals);
            deme.splice(0..new_pop.len(), new_pop);
        }

        // Crossover for receiving deme
        let (new_individuals, new_evals) =
            rank_based_reproduction(&mut receiving_deme, Some(&island_demes), crossover_operator);
        evaluations += new_evals;

        // Combine new and original individuals and take the best POP_SIZE individuals.
        let new_pop = select_best(&receiving_deme, &new_individuals);
        receiving_deme.splice(0..new_pop.len(), new_pop);

        // Calculate average fitness of receiving deme
        let avg_fitness = average_fitness(&receiving_deme);
        println!("Average fitness in generation {}: {}", _g + 1, avg_fitness);
        fitness_history.push(avg_fitness);
    }

    println!(
        "Finished. {} evaluations in {:?}.",
        evaluations,
        timer.elapsed()
    );

    let max = max_fitness_individual(&receiving_deme);
    println!(
        "Highest fitness individual has fitness {}. Individual:\n{:?}\nPermuted Individual:\n{:?}\nLinkage:\n{:?}\n",
        max.1,
        split_into_ns(&max.0, K),
        split_into_ns(&permute(&max.0, &max.2), K),
        max.2
    );

    fitness_history
}
