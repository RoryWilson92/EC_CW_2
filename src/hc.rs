use std::time::Instant;

use crate::consts::*;
use crate::genetic_operators::{
    average_fitness, fitness, generate_population, max_fitness_individual, mutate, permute,
    split_into_ns,
};

pub(crate) fn hc() -> Vec<f64> {
    println!("Running HC");

    // Initialisations
    let mut evaluations = 0;
    let mut fitness_history: Vec<f64> = Vec::new();
    let mut population = generate_population((NEW * N_DEMES) + N_DEMES);

    let init_fitness = average_fitness(&population);
    println!("Average fitness at initialisation: {}", init_fitness);
    fitness_history.push(init_fitness);

    // Main loop
    let timer = Instant::now();
    for _g in 0..GENERATIONS {
        // Mutate all individuals and keep the best
        population = population
            .iter()
            .map(|i| {
                let x = mutate(i.0.to_owned());
                let fit_x = fitness(&x);
                evaluations += 1;
                if fit_x > i.1 {
                    (x, fit_x, i.2)
                } else {
                    i.to_owned()
                }
            })
            .collect();

        let avg_fitness = average_fitness(&population);
        println!("Average fitness in generation {}: {}", _g + 1, avg_fitness);
        fitness_history.push(avg_fitness);
    }

    println!(
        "Finished. {} evaluations in {:?}.",
        evaluations,
        timer.elapsed()
    );

    let max = max_fitness_individual(&population);
    println!(
        "Highest fitness individual has fitness {}. Individual:\n{:?}\nPermuted Individual:\n{:?}\nLinkage:\n{:?}\n",
        max.1,
        split_into_ns(&max.0, K),
        split_into_ns(&permute(&max.0, &max.2), K),
        max.2
    );

    fitness_history
}
