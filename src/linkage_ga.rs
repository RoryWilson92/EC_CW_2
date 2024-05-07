use rand::prelude::{IndexedRandom, SliceRandom};
use rand::Rng;

use crate::consts::*;
use crate::genetic_operators::{fitness, mutate, permute, run_ga};

fn linkage_mutate(linkage: [usize; N]) -> [usize; N] {
    let mut rng = rand::thread_rng();
    let mut new_linkage = linkage.to_owned();
    for i in 0..linkage.len() {
        if rng.gen_range(0.0..1.0) <= MUTATION_RATE / 0.5 {
            let j = rng.gen_range(0..linkage.len());
            new_linkage.swap(i, j);
        }
    }
    new_linkage
}

fn linkage_self_crossover(linkage: [usize; N]) -> [usize; N] {
    // let mut rng = rand::thread_rng();
    let mut new_linkage = [0; N];
    
    let mut tmp = (0..B).collect::<Vec<usize>>();
    tmp.shuffle(&mut rand::thread_rng());
    
    for i in 0..B {
        new_linkage[(i * K)..(i * K) + K].copy_from_slice(&linkage[(tmp[i] * K)..(tmp[i] * K) + K])
    }
    
    // let c = N / 2;
    // new_linkage[0..c].copy_from_slice(&linkage[c..N]);
    // new_linkage[c..N].copy_from_slice(&linkage[0..c]);
    new_linkage
    // linkage_mutate(new_linkage)
}

fn one_point_crossover(
    population: &[(String, f64, [usize; N])],
    individual: Option<&(String, f64, [usize; N])>,
) -> (String, f64, [usize; N]) {
    let mut rng = rand::thread_rng();
    let pop_and_weights = &population
        .iter()
        .enumerate()
        .collect::<Vec<(usize, &(String, f64, [usize; N]))>>();

    let x1 = match individual {
        Some(x) => x,
        None => {
            pop_and_weights
                .choose_weighted(&mut rng, |elem| elem.0)
                .unwrap()
                .1
        }
    };
    let x2 = &pop_and_weights
        .choose_weighted(&mut rng, |elem| elem.0)
        .unwrap()
        .1;

    // Crossover and mutate individuals
    let c = rng.gen_range(0..N);
    let y1 = mutate(x1.0[0..c].to_owned() + &x2.0[c..N]);
    let y2 = mutate(x2.0[0..c].to_owned() + &x1.0[c..N]);
    let y1fit = fitness(&permute(&y1, &x1.2));
    let y2fit = fitness(&permute(&y2, &x2.2));
    // Generate new linkage map and pick best individual
    if y1fit > y2fit {
        let new_linkage = linkage_self_crossover(x1.2);
        let new_fitness = fitness(&permute(&y1, &new_linkage));
        if new_fitness > y1fit {
            (y1, new_fitness, new_linkage)
        } else {
            (y1, y1fit, x1.2)
        }
    } else {
        let new_linkage = linkage_self_crossover(x2.2);
        let new_fitness = fitness(&permute(&y2, &new_linkage));
        if new_fitness > y2fit {
            (y2, new_fitness, new_linkage)
        } else {
            (y2, y2fit, x2.2)
        }
    }
}

pub(crate) fn linkage_ga() -> Vec<f64> {
    println!("Running Linkage GA");
    run_ga(one_point_crossover)
}
