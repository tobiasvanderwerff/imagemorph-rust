use std::{env, process};

use imagemorph_rust::Config;

fn main() {
    // Process command line arguments
    let args: Vec<String> = env::args().collect(); 
    let cfg = Config::build(&args).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {err}");
        process::exit(1);
    });

    if let Err(e) = imagemorph_rust::run(&cfg) {
        eprintln!("Application error: {e}");
        process::exit(1);
    }
}

