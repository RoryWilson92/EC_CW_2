use csv::Writer;
use rand::distributions::Slice;
use rand::seq::IndexedRandom;
use rand::Rng;
use std::error::Error;
use std::time::Instant;

fn write_to_csv(data: [f64; GENERATIONS], file_path: &str) -> Result<(), Box<dyn Error>> {
    let mut writer = Writer::from_path(file_path)?;
    for value in data {
        writer.write_record(&[value.to_string()])?;
    }
    writer.flush()?;
    Ok(())
}

fn generate_individual(length: usize) -> (String, f64) {
    let char_dist: Slice<char> = Slice::new(CHARS).unwrap();
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
    let chars: Vec<char> = individual.chars().collect();
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
    // for gi in gis {
    //     for t in 0..T {
    //         let hamming = gi
    //             .chars()
    //             .zip(TARGETS[t].chars())
    //             .filter(|(a, b)| a != b)
    //             .count();
    //         fitness += if hamming == 0 {
    //             TGT_SCORES[t] as f64
    //         } else {
    //             1.0 / ((1 + hamming) as f64)
    //         }
    //     }
    // }
    fitness
}

fn mutate(individual: String) -> String {
    let mut rng = rand::thread_rng();
    individual.chars().map(|c| {
        if rng.gen_range(0.0..1.0) <= MUTATION_RATE {
            if c == '0' { '1' } else { '0' }
        } else { c }
    }).collect()
}

fn one_point_crossover(individual: &(String, f64), population: &[(String, f64)]) -> (String, f64) {
    let mut rng = rand::thread_rng();
    let xi = &individual.0;
    let xj = &population
        .iter()
        .enumerate()
        .collect::<Vec<(usize, &(String, f64))>>()
        .choose_weighted(&mut rng, |elem| elem.0)
        .unwrap()
        .1
         .0;
    let c = rng.gen_range(0..N);
    let mut y1 = String::with_capacity(N); y1.push_str(&xi[0..c]); y1.push_str(&xj[c..N]);
    let mut y2 = String::with_capacity(N); y2.push_str(&xj[0..c]); y2.push_str(&xi[c..N]);
    let y = mutate(if fitness(&y1) > fitness(&y2) { y1 } else { y2 });
    let fit = fitness(&y);
    (y, fit)
}

fn one_point_crossover2(population: &[(String, f64)]) -> (String, f64) {
    let mut rng = rand::thread_rng();
    let pop_and_weights = &population
        .iter()
        .enumerate()
        .collect::<Vec<(usize, &(String, f64))>>();
    let xi = &pop_and_weights.choose_weighted(&mut rng, |elem| elem.0).unwrap().1.0;
    let xj = &pop_and_weights.choose_weighted(&mut rng, |elem| elem.0).unwrap().1.0;
    let c = rng.gen_range(0..N);
    let mut y1 = String::with_capacity(N); y1.push_str(&xi[0..c]); y1.push_str(&xj[c..N]);
    let mut y2 = String::with_capacity(N); y2.push_str(&xj[0..c]); y2.push_str(&xi[c..N]);
    let y = mutate(if fitness(&y1) > fitness(&y2) { y1 } else { y2 });
    let fit = fitness(&y);
    (y, fit)
}

fn _one_point_crossover3(population: &[(String, f64)]) -> (String, f64) {
    let mut rng = rand::thread_rng();
    let c = rng.gen_range(0..N);
    let xi = &population.choose(&mut rng).unwrap().0;
    let xj = &population.choose(&mut rng).unwrap().0;
    let mut y1 = String::with_capacity(N); y1.push_str(&xi[0..c]); y1.push_str(&xj[c..N]);
    let mut y2 = String::with_capacity(N); y2.push_str(&xj[0..c]); y2.push_str(&xi[c..N]);
    let y = mutate(if fitness(&y1) > fitness(&y2) { y1 } else { y2 });
    let fit = fitness(&y);
    (y, fit)
}

const CHARS: &[char] = &['0', '1'];
const N_DEMES: usize = 100; // number of populations
const POP_SIZE: usize = 10; // size of each population
const N: usize = 100; // genotype length - O: 400
const T: usize = 2; // number of peaks per sub-function
const W1: usize = 1; // value of first tgt string
const W2: usize = 10; // value of second tgt string
const B: usize = 10; // number of blocks per genotype - O: 20
const K: usize = 10; // size of each block - O: 20
// const T1: &str = "10101010101010101010"; // tgt string 1
const T1: &str = "1010101010"; // tgt string 1
// const T2: &str = "11111111111111111111"; // tgt string 2

const T2: &str = "1111111111"; // tgt string 2
const TARGETS: [&str; 2] = [T1, T2];
const TGT_SCORES: [usize; 2] = [W1, W2];
const MUTATION_RATE: f64 = 1.0 / (N as f64);
// const _MUTATION_RATE: i32 = 2500;
const MIGRATION_RATE: f64 = 0.0004;
// const _MIGRATION_RATE: i32 = 400;
const GENERATIONS: usize = 5000;
const RF: f64 = 1.0; // Reproductive scaling factor, higher -> more individuals.
const NEW: usize = POP_SIZE;

fn run_ga() -> Vec<f64> {
    println!("Running GA");
    // Initialisations
    let mut demes: Vec<Vec<(String, f64)>> = Vec::new();
    for _ in 0..N_DEMES {
        let mut _deme: Vec<(String, f64)> = Vec::new();
        for _ in 0..POP_SIZE {
            _deme.push(generate_individual(N));
        }
        demes.push(_deme);
    }
    let mut evaluations = 0;
    let mut rng = rand::thread_rng();
    let mut avg_fit_list: Vec<f64> = Vec::new();
    // let norm_f: usize = (0..POP_SIZE).sum();
    let mut migrations = 0;

    // evaluations += POP_SIZE;

    // Main loop
    let timer = Instant::now();
    for _g in 0..GENERATIONS {
        // let mut avg_new = 0;
        for d in 0..N_DEMES {
            let migrant_deme = &demes.choose(&mut rng).unwrap().clone();
            let deme = &mut demes[d];
            let mut new_individuals: Vec<(String, f64)> = Vec::new();

            deme.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            if rng.gen_range(0.0..1.0) <= MIGRATION_RATE * POP_SIZE as f64{
                migrations += 1;
                let migrant = migrant_deme
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                new_individuals.push(one_point_crossover(migrant, deme));
                evaluations += 1;
                for _ in 0..NEW - 1 {
                    new_individuals.push(one_point_crossover2(deme));
                    evaluations += 1;
                }
            } else {
                for _ in 0..NEW {
                    new_individuals.push(one_point_crossover2(deme));
                    evaluations += 1;
                }
            }

            // for (i, x) in deme.iter().enumerate() {
            //     if rng.gen_range(0.0..1.0) <= (i as f64 / norm_f as f64) * RF {
            //         new_individuals.push(one_point_crossover(x, deme));
            //         evaluations += 1;
            //     }
            //     if rng.gen_range(0.0..1.0) <= MIGRATION_RATE {
            //     // if rng.gen_range(0..1000000) <= _MIGRATION_RATE {
            //         migrations += 1;
            //         // println!("Migration Event.");
            //         let migrant = migrant_deme
            //             .iter()
            //             .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            //             .unwrap();
            //         new_individuals.push(one_point_crossover(migrant, deme));
            //         evaluations += 1;
            //     }
            // }

            // let mut repr_pool: Vec<(String, f64)> = Vec::new();
            // deme.iter().enumerate().for_each(|(i, x)| repr_pool.append(&mut vec![x.clone(); i]));
            // if rng.gen_range(0..1000) <= _MIGRATION_RATE {
            //     let migrant = migrant_deme.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
            //     new_individuals.push(one_point_crossover(migrant, repr_pool.as_slice()));
            //     evaluations += 1;
            //     for _ in 0..POP_SIZE-1 {
            //         new_individuals.push(one_point_crossover3(repr_pool.as_slice()));
            //         evaluations += 1;
            //     }
            // } else {
            //     for _ in 0..POP_SIZE {
            //         new_individuals.push(one_point_crossover3(repr_pool.as_slice()));
            //         evaluations += 1;
            //     }
            // }

            // let min = &deme
            //     .iter()
            //     .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            //     .unwrap()
            //     .1;
            // new_individuals.retain(|x| x.1 >= *min);
            
            let mut tmp = deme.clone();
            tmp.append(&mut new_individuals);
            tmp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let new = tmp.iter().take(POP_SIZE).rev().map(|x| x.to_owned()).collect::<Vec<(String, f64)>>();

            deme.splice(0..new.len(), new);

            // deme.splice(
            //     (POP_SIZE - n_new)..POP_SIZE,
            //     // new,
            //     new_individuals,
            // );
        }
        let avg_fitness = demes
            .iter()
            .fold(0.0, |acc, x| acc + x.iter().fold(0.0, |acc, x| acc + x.1))
            / ((N_DEMES * POP_SIZE) as f64);
        println!("Average fitness in generation {}: {}", _g, avg_fitness);
        avg_fit_list.push(avg_fitness);
        // println!(
        //     "{} new individuals on average in generation {}",
        //     avg_new / N_DEMES,
        //     _g
        // )
    }
    println!("Finished. {} evaluations in {:?}.", evaluations, timer.elapsed());
    println!("{} migrations per generation on average.", migrations as f64 / GENERATIONS as f64);

    let max = demes
        .iter()
        .map(|deme| {
            deme.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    println!(
        "Highest fitness individual has fitness {}. Individual:\n{:?}",
        max.1,
        split_into_ns(&max.0, K)
    );

    avg_fit_list
}

fn run_hc() -> Vec<f64> {
    println!("Running HC");
    
    // Initialisations
    let mut evaluations = 0;
    let mut avg_fit_list: Vec<f64> = Vec::new();
    let mut population: Vec<(String, f64)> = Vec::new();
    for _ in 0..(N_DEMES * POP_SIZE) {
        population.push(generate_individual(N));
    }

    // Main loop
    let timer = Instant::now();
    for _g in 0..GENERATIONS {
        for i in 0..(N_DEMES * POP_SIZE) {
            let x = mutate(population[i].0.to_owned());
            let fit_x = fitness(&x);
            evaluations += 1;
            if fit_x > population[i].1 {
                population[i] = (x, fit_x)
            }
        }
        // let x = mutate(individual.0.to_owned());
        // let fit_x = fitness(&x);
        // if fit_x > individual.1 {
        //     evaluations += 1;
        //     individual = (x, fit_x)
        // }
        let avg_fitness = population.iter().fold(0.0, |acc, x| acc + x.1) / (N_DEMES * POP_SIZE) as f64;
        println!("Average fitness in generation {}: {}", _g, avg_fitness);
        avg_fit_list.push(avg_fitness);
    }
    
    println!("Finished. {} evaluations in {:?}.", evaluations, timer.elapsed());
    
    avg_fit_list
}

fn main() {
    let n = 1;
    let mut ga_res = [0.0; GENERATIONS];
    let mut hc_res = [0.0; GENERATIONS];

    for _ in 0..n {
        let ga_avg_fit_list = run_ga();
        let hc_avg_fit_list = run_hc();
        for i in 0..GENERATIONS {
            ga_res[i] += ga_avg_fit_list[i] / (n as f64);
            hc_res[i] += hc_avg_fit_list[i] / (n as f64);
        }
    }

    if let Err(err) = write_to_csv(ga_res, "ga_out.csv") {
        println!("An error occurred while writing to CSV: {}", err);
    }
    
    if let Err(err) = write_to_csv(hc_res, "hc_out.csv") {
        println!("An error occurred while writing to CSV: {}", err);
    }
}
