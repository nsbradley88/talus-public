//! Talus CLI entry point (`talus-cli` / `strata-q36` / `talus-qwen35`).

fn main() {
    let code = talus_cli::run_cli_with_args(std::env::args());
    if code != 0 {
        std::process::exit(code);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cli_smoke() {
        assert_eq!(talus_cli::CRATE_NAME, "talus-cli");
    }
}
