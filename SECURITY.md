# Security

## Reporting a vulnerability

Please report security vulnerabilities **privately** through GitHub's private vulnerability reporting: the repository's **Security** tab, **Report a vulnerability** ([direct link](https://github.com/nsbradley88/talus-public/security/advisories/new)). Include what you found, how to reproduce it (request payload, configuration keys involved, and commit or release version), and the potential impact.

Ordinary hardening suggestions that are safe to discuss in public are welcome as a [GitHub issue](https://github.com/nsbradley88/talus-public/issues) or pull request.

Supported versions: the latest release and `main`.

## Server security model

Talus serves models over HTTP via `talus-server` (`rust/crates/talus-server`) or the Python wrapper (`serve/server.py`). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/MODELS.md](docs/MODELS.md) for configuration details.

- **Bind address:** `127.0.0.1` by default. Setting `--host 0.0.0.0` (or `"host"` in the server configuration JSON) exposes the server to the local network; always configure an API key before listening on a non-loopback interface.
- **API key authentication:** `"api_key"` in the configuration JSON (or `--api-key` / `STRATA_API_KEY`) gates `/v1/*` and control endpoints (`/status`, `/metrics`, `/settings`, `/props`, `/slots`, `/api/requests`, `/config`, and all `POST` endpoints) using a constant-time comparison.
- **Host and Origin checks:** When no API key is configured on loopback, the server enforces `Host` and `Origin` header validation to guard against DNS rebinding and cross-origin browser requests (`"allowed_hosts"`, `"trusted_origins"`, `"cors_origins"`).
- **CORS:** Disabled by default unless `"cors_origins"` is explicitly configured.

