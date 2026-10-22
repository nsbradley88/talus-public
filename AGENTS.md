# AGENTS.md

Talus is a local LLM inference engine for AMD Strix Halo (`gfx1151`, RDNA 3.5 APU with unified LPDDR5X memory). It combines a Rust host (`rust/`, building `talus-cli` and `talus-server`) with a HIP/CUDA kernel library (`src/`, `include/`, building `libtalus_kernels.so`).

## Documentation

- Overview and quick start: [README.md](README.md)
- Host and kernel architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Supported model families, quantizations, and serving settings: [docs/MODELS.md](docs/MODELS.md)
- Evaluation and benchmark methodology: [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
- Release benchmark and quality results: [docs/RESULTS.md](docs/RESULTS.md)

## Building and testing

- Full kernel + Rust build: `./build.sh`
- Rust workspace CI gate (formatting, clippy, unit tests, license check): `./rust/ci.sh`
- Documentation claim check: `python3 tools/check_claims.py check`

## Contributing guidelines

- Keep changes focused and preserve default numerical behavior unless explicitly gated and verified.
- Run `./rust/ci.sh` and `python3 tools/check_claims.py check` before submitting changes.
- Every performance or quality number in Markdown documentation must come from a verified release run via release placeholders in `docs/RESULTS.md` (`tools/check_claims.py`).
- Never expose `talus-server` or `serve/server.py` beyond `127.0.0.1` without configuring `--api-key`.

