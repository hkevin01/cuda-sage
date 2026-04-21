# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅        |
| < 0.2   | ❌        |

## Scope

cuda-sage is a **static analysis tool** — it reads PTX text files and performs pure computation. It does not:

- Execute PTX or CUDA code
- Make network requests
- Access credentials or secrets
- Write to locations outside the `--output` path you specify

The primary attack surface is **malicious PTX input**. The parser uses Python `re` with bounded patterns and reads files as plain text. No `eval` or `exec` is used anywhere in the codebase.

## Reporting a Vulnerability

If you discover a security vulnerability (e.g., a crafted PTX file that causes path traversal via `--output`, a regex ReDoS, or unexpected code execution):

1. **Do not open a public issue.**
2. Email the maintainer directly with:
   - A description of the vulnerability
   - A minimal reproducer (PTX file or command)
   - Your assessment of severity and impact
3. You will receive a response within 72 hours.
4. Once a fix is released, the vulnerability will be disclosed publicly with credit to the reporter.

## Known Non-Issues

- The `--output` flag writes only to the exact path you supply; no directory traversal is possible.
- PTX parsing is regex-based with no shell invocation or `subprocess` usage.
