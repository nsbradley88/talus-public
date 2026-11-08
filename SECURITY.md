# Security

## Reporting a vulnerability

Please report security vulnerabilities **privately** through GitHub's private vulnerability reporting: the repository's **Security** tab, **Report a vulnerability** ([direct link](https://github.com/nsbradley88/talus-public/security/advisories/new)). Include what you found, how to reproduce it (request payload, configuration keys involved, and commit or release version), and the potential impact.

Ordinary hardening suggestions that are safe to discuss in public are welcome as a [GitHub issue](https://github.com/nsbradley88/talus-public/issues) or pull request.

Supported versions: the latest release and `main`.

## Server security model

Talus serves models over HTTP via `talus-server` (`rust/crates/talus-server`) or the Python wrapper (`serve/server.py`). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/MODELS.md](docs/MODELS.md) for configuration details.

- **API key (set one before the server listens beyond this machine):** `TALUS_API_KEY=<key>` in the server's environment. `--api-key <key>` and `"api_key"` in the configuration JSON work too, and so does the older name `STRATA_API_KEY`. If more than one is set, `--api-key` wins, then the environment (`STRATA_API_KEY` over `TALUS_API_KEY`), then the configuration file. Clients send the key as `Authorization: Bearer <key>` or `x-api-key: <key>`. The key is compared in constant time; which requests need it is below.
- **Bind address:** `127.0.0.1` by default. Setting `--host 0.0.0.0` (or `"host"` in the server configuration JSON) exposes the server to the local network; set `TALUS_API_KEY` first.
- **Request limits:** request size, JSON values per request, JSON-schema complexity, large bodies received at once, stop-string length and open connections are limited, and a request past a limit gets an HTTP error while the server keeps serving. The limits are listed in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#serving).
- **Without an API key, Host check:** every request whose `Host` header names something other than `localhost` (or a `*.localhost` name), a loopback address, an IP address, or the configured `host` gets HTTP 403. This guards against DNS rebinding. A request without a `Host` header is let through.
- **Without an API key, Origin check:** a `POST` whose `Origin` is not local (`localhost`, `127.0.0.1`, `[::1]`, the configured `host`, or the request's own `Host`) gets HTTP 403, and a `POST` from a local origin must send `Content-Type: application/json` (HTTP 415 otherwise). `GET` responses carry no CORS headers, so a page on another origin can't read them.
- **CORS:** a preflight (`OPTIONS`) gets 204 with `Access-Control-Allow-Origin: *`, after the Host check. No other response carries CORS headers. talus-server has no CORS setting.
- **With an API key:** the Host and Origin checks are off, and every request needs the key except `GET /health`, `GET /api/health` and the `OPTIONS` preflight.
- **Python wrapper:** `"allowed_hosts"`, `"trusted_origins"` and `"cors_origins"` are settings of the Python wrapper (`serve/`), not of talus-server.

