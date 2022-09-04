use std::io::{self, BufRead};

use regex_compiler::{compile, parse};

const USAGE: &str = "re PATTERN [FILE]";

fn main() -> Result<(), String> {
    let (debug, args) = std::env::args()
        .skip(1)
        .fold((false, vec![]), |(debug, mut args), arg| {
            if arg == "--debug" || arg == "-d" {
                (true, args)
            } else {
                args.push(arg);
                (debug, args)
            }
        });
    let arg_len = args.len();

    let (pattern, input) = match arg_len {
        1 => args
            .get(0)
            .map(|pattern| (pattern.as_str(), io::stdin()))
            .ok_or_else(|| USAGE.to_string()),
        _ => Err(USAGE.to_string()),
    }?;

    let program = parse(pattern)
        .map_err(|e| e.to_string())
        .and_then(compile)?;

    if debug {
        println!(
            "DEBUG
--------
{}--------
",
            program
        )
    }

    for line in input.lock().lines() {
        match line {
            Ok(line) => match regex_runtime::run::<0>(&program, &line) {
                Some([]) => println!("{}", line),
                None => continue,
            },
            Err(e) => return Err(format!("{}", e)),
        }
    }

    Ok(())
}
