# Contributions and licensing

Preserve numerical, ownership, memory, cancellation and capability contracts.
Record actual validation and unrun devices. Keep secrets, models, private data,
binary artifacts and signing keys out of commits.

Read [history](docs/LICENSE_HISTORY.md). Do not strip old notices or assert
ownership of imports. Provide origin, revision and complete licence for third-party
code/assets, including applicable restrictions on generated material.

For new external contributions intended for community and paid licensing,
maintainers must obtain actual authorized acceptance of the
[contributor agreement](docs/CONTRIBUTOR_AGREEMENT.md), including employer authority
where applicable. A PR or DCO sign-off alone does not assign copyright
or establish commercial sublicensing rights. Do not merge such contributions
until rights and the agreement are reviewed/recorded. Authors keep ownership and
credit; the template is not evidence anyone signed.

Licensing and contribution questions: sabbirimon@gmail.com.
Do not send credentials or private datasets.

## Hardware and enterprise test results

Large organizations, hardware labs and individual owners are welcome to share
non-production measurements. The most useful new coverage includes different
CPU architectures, GPU drivers, accelerator availability, memory domains,
multi-host links, device-loss cases and sustained workloads. Run the exact
supported contract first; label a planned adapter or unsupported operator as
unavailable instead of treating its presence in an inventory as execution.

Follow the [test-report guide](docs/TEST_REPORTING.md). Include a source commit,
toolchain and driver versions, hardware topology, reproducible workload, oracle
or output hash, all measured samples, memory and transfer scope, and the precise
failure or cancellation behavior. Mark permissions, data residency and shared
cluster conditions that affect a result. Redact hostnames, credentials, private
datasets and organizational identifiers. Existing reports in
[validation](docs/VALIDATION.md) show the level of evidence expected.

## Implementation and documentation workflow

Start with the [programming guide](docs/PROGRAMMING_GUIDE.md),
[build/test runbook](docs/BUILD_AND_TEST.md), and
[pending-feature handoff](docs/AGENT_BACKLOG.md). Record each fix's symptom, cause,
regression check and remaining target limits in the progress/validation documents.
Keep historical measurements and licensing records intact. For debugging, use
[the troubleshooting guide](docs/TROUBLESHOOTING.md) and submit redacted minimal
reproducers. A provider mock, device inventory or emitted kernel is not hardware
execution evidence.
