# Ablation — does intrinsic verification help? (and when?)

A 2×2 ablation on GAIA validation (165 tasks) crossing **model capacity**
(`gpt-5-nano` vs `gpt-5-mini`) with **verification** (bare `react` vs
`react` wrapped in `verifier_retry` over the four intrinsic verifiers:
`format`, `arithmetic`, `citation`, `coverage`).

The motivating question from the main validation run: the intrinsic
verifier suite *lowered* accuracy on `gpt-5-nano` (42.4% → 37.6%). Is that
a property of the verifier design, or an artifact of running it on a
capacity-limited model? This ablation tests whether the sign of the effect
depends on model capacity.

## Result

| Model | `react` | `react+intrinsic` | Δ (verification) |
|---|---|---|---|
| `gpt-5-nano` | 42.4% (70/165) | 37.6% (62/165) | **−4.8pp** |
| `gpt-5-mini` | 51.5% (85/165) | 53.9% (89/165) | **+2.4pp** |

**The sign flips.** Intrinsic verification is net-negative on the
capacity-limited model and net-positive on the stronger one. The
capacity × verification **interaction is +7.3pp**.

### Per level (`react` → `react+intrinsic`)

| Level | `gpt-5-nano` | `gpt-5-mini` |
|---|---|---|
| L1 | 49.1% → 47.2% (−1.9pp) | 62.3% → 66.0% (+3.8pp) |
| L2 | 46.5% → 38.4% (−8.1pp) | 53.5% → 54.7% (+1.2pp) |
| L3 | 15.4% → 15.4% (±0) | 23.1% → 26.9% (+3.8pp) |

On `gpt-5-nano` the damage concentrates at L2 (−8.1pp); on `gpt-5-mini`
every level is flat-to-positive.

## Mechanism: the verifier's false-positive rate

The interaction is explained by how often the verifier **rejects an answer
that was already correct** (a false-positive rejection — the bare-`react`
run got the task right, but the `verifier_retry` run, after re-checking and
retrying, got it wrong), versus how often it **rescues** a wrong answer.

| Model | False-positive rejections | Genuine fixes | Net |
|---|---|---|---|
| `gpt-5-nano` | **21 / 70 correct (30.0%)** | 13 | −8 |
| `gpt-5-mini` | **12 / 85 correct (14.1%)** | 16 | +4 |

On the weak model the verifier breaks correct answers more often than it
fixes wrong ones (30% false-positive rate, 13 fixes → net −8). On the
stronger model the false-positive rate roughly **halves to 14%**, and
genuine fixes finally outnumber breakages (net +4). That sign change *is*
the headline effect.

## Statistical honesty

Neither per-model Δ is statistically significant at n = 165.

| Model | `react`-only-right | `intrinsic`-only-right | McNemar (exact, 2-sided) |
|---|---|---|---|
| `gpt-5-nano` | 21 | 13 | p = 0.229 |
| `gpt-5-mini` | 12 | 16 | p = 0.572 |

The discordant-pair counts (34 and 28) are too small to call either
single-model effect significant. **This is a suggestive directional
finding plus a directly measured mechanism (the false-positive rate), not a
significance claim.** The false-positive rate is the most defensible number
here because it is a direct measurement rather than a test against a null.

## Cost

| Model | `react` | `react+intrinsic` |
|---|---|---|
| `gpt-5-nano` | $0.87 | $1.01 |
| `gpt-5-mini` | $6.74 | $7.00 |

Verification adds ~15–25¢ per full run (the retry ticks). The `mini` rows
are the two runs added for this ablation; the `nano` rows are reused from
the main validation run.

## Reproduce

The `nano` rows come from the main validation configs; the `mini` rows use
the same configs with `model: gpt-5-mini`. Each row is one YAML run through
`experiments/02_gaia_full/run.py --config <row>.yaml` over all 165 tasks
(`all_levels: true`, `budget_wall_s: 240`, `budget_cost: 2.0`).

```yaml
# react × gpt-5-mini
id: A_react_mini
policy: react
model: gpt-5-mini
provider: openai
all_levels: true
budget_wall_s: 240
budget_cost: 2.0
```

```yaml
# verifier_retry(react) + intrinsic verifiers × gpt-5-mini
id: B_verifier_retry_react_intrinsic_mini
policy: verifier_retry
inner: react
verifiers: [format, arithmetic, citation, coverage]
model: gpt-5-mini
provider: openai
all_levels: true
budget_wall_s: 240
budget_cost: 2.0
```

## Limitations

- **Underpowered per model.** 165 tasks; single-model Δs are within noise
  (see McNemar above). The result rests on the *direction* of the flip and
  the false-positive mechanism, not on significance.
- **Two models, one provider.** `gpt-5-nano` → `gpt-5-mini` is a capacity
  proxy on a single provider, not an isolated capacity variable.
- **Validation set, not held-out.** Same caveats as
  [`gaia_validation_report.md`](gaia_validation_report.md): the GAIA test
  split is private; this is the public validation set, also used during
  engineering iteration.
- **Intrinsic verifiers only.** This ablation does not cover extrinsic
  (reflexion-style) verification or planner policies.
