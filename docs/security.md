# Security model

This document states what the agent's code-execution sandbox does and does
not protect against. It is deliberately explicit about non-guarantees: the
honest boundary matters more than the marketing.

The agent can run model-emitted code through two tools — `run_python` and
`run_shell` — both dispatched through a `SandboxBackend`. There are two
backends, selected with `--sandbox` (or the `BANNA_SANDBOX` env var).

## Threat model

The actor we reason about is **the model**, not a human attacker on the box.
The model emits code or shell commands as part of solving a task; that code
may be wrong, may be adversarially steered by a prompt-injected web page or
document the agent read, or may simply do something destructive by accident.
The question this document answers is: *when the model emits hostile code,
what can it reach?*

We do **not** model a human adversary with shell access to the host, a
malicious dependency in the agent's own install, or a compromised LLM
provider. Those are out of scope.

## The two backends

### `process` (default)

Each `run_python` / `run_shell` call runs as a host subprocess.

- **Guarantees:** a real wall-clock timeout and a separate process (so a hang
  or a crash doesn't take down the agent). `run_shell` is further constrained
  by a regex allowlist — only matching command lines run, and under the
  interactive CLI each one is gated by a yes/no confirmation prompt.
- **Non-guarantees:** the subprocess inherits the user's filesystem, network,
  and credentials. It can read your files, make network calls, and spend money
  against any API key in the environment. The allowlist is the only
  command-level guard, and a regex allowlist is a speed bump, not a wall.

**Use `process` only for a research harness on your own machine, on input you
trust.** It is the backend the published GAIA numbers were produced with,
precisely because GAIA tasks are not adversarial.

### `docker`

Each call runs in a throwaway container. Every container we launch gets the
same hardening flags:

| Flag | Effect |
|---|---|
| `--rm` | container is deleted on exit; no state persists between calls |
| `--network none` | no network namespace — the code cannot make any network call |
| `--read-only` | root filesystem is read-only; only an explicit `tmpfs` scratch area is writable |
| `--cap-drop ALL` | all Linux capabilities dropped |
| `--security-opt no-new-privileges` | a setuid binary cannot regain privileges |
| `--pids-limit 256` | bounds fork-bombs |
| `--memory 512m` | hard memory ceiling (OOM-kills the container, not the host) |
| `--cpus 2` | CPU ceiling |
| `--user 65534:65534` | runs as `nobody`, never root inside the container |

- **Guarantees against the model:** no network exfiltration or callout, no
  persistence across calls, no access to the host filesystem or the user's
  credentials, no privilege escalation inside the container, and bounded
  CPU/memory/PID consumption.
- **Non-guarantees:** this is **container-level isolation, not a security
  boundary against a determined adversary.** Containers share the host
  kernel. A kernel-level exploit (a container escape via a kernel
  vulnerability) is not stopped by any of the flags above — they reduce the
  attack surface and the blast radius, they do not eliminate the shared-kernel
  risk.

**For genuinely hostile code, run under a stronger runtime:** a
syscall-filtering sandbox such as gVisor (`runsc`), or a microVM such as
Firecracker or Kata, each of which puts a real boundary between the workload
and the host kernel. The `--sandbox-image` flag and the `SandboxBackend`
interface are designed to accommodate a hardened runtime without changing the
agent.

## On-demand package install (docker backend)

Because the run container has `--network none`, a missing third-party package
cannot be `pip install`-ed at runtime — there is no network inside the
container, by design. Installing one therefore happens in a **separate,
two-phase step** that never lets model code run with the network on:

1. **Build phase** — network on, no model code. The backend writes a minimal
   Dockerfile (`FROM <base>` + `RUN pip install <pinned spec>`) to a temporary
   directory and runs `docker build`. The only thing executed here is `pip`,
   against pinned `name==version` specs. No code the model emitted runs in this
   phase.
2. **Run phase** — network off, read-only, as above. The model's code runs
   against the freshly built image, with the package baked into a read-only
   layer.

Derived images are cached, keyed by a hash of the base image plus the sorted
set of pins, so a given package set is built once and reused.

### Allowlist behavior

- **Allowlisted packages** (`import_name → dist==version` pins) install with no
  prompt, under every policy — including headless / batch runs. The allowlist
  ships with a curated, version-pinned default set (numpy, pandas, scipy,
  sympy, matplotlib, scikit-learn, pillow, opencv, requests, lxml, openpyxl,
  …). Extend or override it with `banna config packages add <import>
  <dist==version>`.
- **Non-allowlisted packages** prompt for approval in an interactive run
  (install-once / add-to-allowlist / deny). With **no human present** (a
  headless or batch run) or on **deny**, the import error is returned
  unchanged — the agent never silently installs something unvetted.

The trust decision is therefore explicit and auditable: a package either is on
a version-pinned allowlist you control, or a human approved it, or it did not
install.

## What this does not cover

- **Factual correctness of tool output.** The sandbox isolates *execution*; it
  says nothing about whether a web page the agent read was truthful. The
  verifier suite checks structural defensibility, not ground truth.
- **The `process` backend.** None of the docker guarantees apply to
  `--sandbox=process`; that backend trades isolation for zero setup and is for
  trusted input only.
- **Supply-chain trust of the agent's own dependencies.** This document is
  about code the *model* emits, not the packages the agent itself is built on.
