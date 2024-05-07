use rand::prelude::{IndexedRandom, SliceRandom};
use rand::Rng;

use crate::consts::*;
use crate::genetic_operators::{fitness, mutate, permute, run_ga};

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
    // Perform crossover and mutation in linkage space and then permute back
    let c = rng.gen_range(0..N);
    let y1 = mutate(x1.0[0..c].to_owned() + &x2.0[c..N]);
    let y2 = mutate(x2.0[0..c].to_owned() + &x1.0[c..N]);
    let y1fit = fitness(&permute(&y1, &x1.2));
    let y2fit = fitness(&permute(&y2, &x2.2));
    // Generate new random linkage map (so no learning is possible)
    let mut tmp = (0..N).collect::<Vec<usize>>();
    tmp.shuffle(&mut rand::thread_rng());
    let linkage = <[usize; N]>::try_from(tmp).unwrap();
    // Select best individual
    if y1fit > y2fit {
        (y1, y1fit, linkage)
        // (y1, y1fit, x1.2)
    } else {
        (y2, y2fit, linkage)
        // (y2, y2fit, x2.2)
    }
}

pub(crate) fn ga() -> Vec<f64> {
    println!("Running GA");
    run_ga(one_point_crossover)
}
