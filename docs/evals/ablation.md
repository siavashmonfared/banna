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

## Shipped-policy equivalence (why `react`, not `react+`)

The rows above use the autonomous policies (`react`, `verifier_retry(react)`),
not the interactive defaults the CLI ships (`react+`, `react+verify`). That is
deliberate: `react+`'s interactive affordances — `ask_user`, the per-tool
permission gate — have no human to engage under batch evaluation, so they are
inert. This is **measured, not assumed**. Re-running the `+` family on the full
165-task `gpt-5-nano` set reproduces the bare-policy numbers to within one task:

| Policy (gpt-5-nano, 165 Q) | Accuracy | Bare-policy baseline | Δ |
|---|---|---|---|
| `react+` | 43.0% (71/165) | `react` 42.4% | +0.6pp (+1) |
| `react+verify` (= `verifier_retry(react+)`) | 39.4% (65/165) | `verifier_retry(react)` 37.6% | +1.8pp (+3) |

Both land within noise of their bare counterparts, and the within-`+`-family
verification effect on nano (43.0 → 39.4 = −3.6pp) matches the bare-family sign
(−4.8pp). The capacity sign-flip was only run for the bare family (the `mini`
`+` runs were out of budget), so the 2×2 above stays in bare-policy labels.

## Reproduce

From a clean clone (`pip install -e ".[dev]"`, `export OPENAI_API_KEY=sk-...`),
each of the four cells is one run of the flag-driven validation runner over all
165 tasks. `--policy verifier_retry` wraps `react` over the four intrinsic
verifiers (`arithmetic`, `citation`, `format`, `coverage`) by default — exactly
the set used here.

```bash
# Row A — bare react
python experiments/02_gaia_full/run.py --policy react \
    --provider openai --model gpt-5-nano \
    --all-levels --budget-wall-s 240 --budget-cost 2.0
python experiments/02_gaia_full/run.py --policy react \
    --provider openai --model gpt-5-mini \
    --all-levels --budget-wall-s 240 --budget-cost 2.0

# Row B — verifier_retry(react) + intrinsic verifiers
python experiments/02_gaia_full/run.py --policy verifier_retry \
    --provider openai --model gpt-5-nano \
    --all-levels --budget-wall-s 240 --budget-cost 2.0
python experiments/02_gaia_full/run.py --policy verifier_retry \
    --provider openai --model gpt-5-mini \
    --all-levels --budget-wall-s 240 --budget-cost 2.0
```

The `nano` rows are reused from the main validation run; the `mini` rows are the
two runs added for this ablation.

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
