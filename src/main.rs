use std::error::Error;
use std::time::Instant;

use csv::Writer;

use crate::consts::*;
use crate::ga::ga;
use crate::hc::hc;
use crate::linkage_ga::linkage_ga;

mod consts;
mod ga;
mod genetic_operators;
mod hc;
mod linkage_ga;

fn write_to_csv(data: [f64; GENERATIONS], file_path: &str) -> Result<(), Box<dyn Error>> {
    let mut writer = Writer::from_path(file_path)?;
    for value in data {
        writer.write_record(&[value.to_string()])?;
    }
    writer.flush()?;
    Ok(())
}

fn main() {
    let n = 30;
    let mut lga_res = [0.0; GENERATIONS];
    let mut ga_res = [0.0; GENERATIONS];
    let mut hc_res = [0.0; GENERATIONS];

    let timer = Instant::now();
    for _ in 0..n {
        let lga_avg_fit_list = linkage_ga();
        let ga_avg_fit_list = ga();
        let hc_avg_fit_list = hc();
        for i in 0..GENERATIONS {
            lga_res[i] += lga_avg_fit_list[i] / (n as f64);
            ga_res[i] += ga_avg_fit_list[i] / (n as f64);
            hc_res[i] += hc_avg_fit_list[i] / (n as f64);
        }
    }
    println!("Total runtime: {:?}", timer.elapsed());

    if let Err(err) = write_to_csv(lga_res, "short_lga_out.csv") {
        println!("An error occurred while writing to CSV: {}", err);
    }

    if let Err(err) = write_to_csv(ga_res, "short_ga_out.csv") {
        println!("An error occurred while writing to CSV: {}", err);
    }

    if let Err(err) = write_to_csv(hc_res, "short_hc_out.csv") {
        println!("An error occurred while writing to CSV: {}", err);
    }
}
