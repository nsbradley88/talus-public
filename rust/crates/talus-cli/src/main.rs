//! Talus CLI entry point (`talus-cli` / `strata-q36` / `talus-qwen35`).

fn main() {
    // TALUS_X works as STRATA_X for every inherited STRATA_* switch (before any thread or the kernel library)
    talus_core_api::apply_env_aliases();
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
