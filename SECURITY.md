# Security Policy

## Supported versions

Security fixes are currently applied to the latest release on the `main` branch.

## Reporting a vulnerability

Please use GitHub private vulnerability reporting for this repository. Do not include API
keys, proprietary prompts, user documents, or other sensitive data in a public issue.

## REPL threat model

The local REPL combines a separate worker process, RestrictedPython, an import allowlist,
per-step wall-clock timeouts, bounded output, and optional operating-system resource limits.
These controls reduce accidental damage from model-generated Python. They do not make the
local REPL a security boundary for hostile or multi-tenant code.

RestrictedPython is a language subset, not a general Python sandbox. Python runtime or
dependency vulnerabilities may allow code to escape its restrictions. The worker also runs
under the same operating-system account as the caller. Do not give untrusted users direct
control over prompts or context and then execute the resulting code on a sensitive host.

For untrusted workloads, run the entire application in a hardened container or a dedicated
sandbox service with a read-only filesystem, no host credentials, restricted networking,
resource quotas, and a disposable identity. Keep model API credentials outside the worker
whenever the deployment architecture permits it.

## Optional worker limits

`RLM` accepts `repl_memory_limit_mb`, `repl_cpu_time_limit_seconds`, and
`repl_max_open_files`. These limits are opt-in because safe values depend on the Python
runtime, installed dependencies, platform, and workload. They use POSIX `setrlimit`; a
configured but unsupported limit fails worker startup explicitly instead of being ignored.

The existing `repl_timeout` remains a per-step wall-clock limit. CPU time is cumulative for
the lifetime of one persistent worker. Memory and file limits apply only to that worker, not
to provider requests made by the parent process.
