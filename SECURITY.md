# Security Policy

## Supported versions

SalienceGate is early-stage software. Security fixes are applied to the latest `0.2.x` release and
the current `main` branch.

## Reporting a vulnerability

Do not disclose a vulnerability, leaked credential, or sensitive trace in a public issue. Email
`pieroc.96@gmail.com` instead.

Include:

- the affected revision;
- the smallest safe reproducer;
- expected and observed behavior;
- impact and required preconditions;
- whether any secret or user data was exposed.

Remove real credentials and personal data before sending the report. An acknowledgement should
arrive within seven days. The remediation timeline depends on severity and reproducibility.

## Scope

Treat every trajectory and memory record as untrusted data. SalienceGate is pre-release software,
not a complete security boundary, and must not be used for unattended external side effects. Its
HMACs authenticate redacted local state but do not encrypt the database or reports. Keep databases,
reports, and installation keys access-controlled and back them up consistently.

The [security appendix](docs/security.md) defines the parser, redaction, repository, filesystem,
network, provider, and ATIF trust boundaries, along with their current limitations.
