# support-agent-eval

An evaluation harness for a support agent that **moves money**.

The agent issues real refunds. That single fact changes what "good" means: a
chatbot that is wrong is annoying, an agent that is wrong sends cash to someone
who was not owed it. This repo is the harness that decides whether a change to
that agent is safe to ship.

> **This is a minimal replication, not production code.** I built and operated
> the original support agent and its eval at a previous employer. That code is
> theirs and is not published here. This repo is a from-scratch re-creation of
> the same design at a fraction of the size: the company is renamed, the policy
> is representative rather than real, and the suite is 18 cases where production
> ran far more. What is faithful is the architecture, the grading strategy, and
> the failure it was built to catch. Incident references such as `the incident review` point
> to real work; the implementation here is my own rewrite of it.

It runs against live Gemini calls, so the numbers come from actual model
behaviour rather than fixtures.

---

## Why this exists

The agent went to production first and got an eval second. That ordering was the
problem.

It was first responder on ~99% of inbound tickets and resolved ~50% of them on its
own, including refunds through a tool call. It looked healthy. The support
dashboard was green. Underneath, three ticket categories were quietly failing,
and every prompt edit or knowledge-base change was a guess: there was no way to
tell a real regression from run-to-run noise in a non-deterministic system.

Reading the transcripts showed one symptom with two unrelated root causes. Some
conversations were **mis-routed at the intent step** (a "charged twice" message
tagged as Subscription Management, so the customer got a cancellation answer
instead of a billing fix). Others were routed correctly and answered from a
**stale knowledge base**, giving steps that no longer matched the product. Same
red cell on the dashboard, two different fixes.

That is what the harness is for: turn "the bot feels worse" into a specific,
fixable list, before a customer finds it.

---

## Architecture

```
   test cases                AGENT UNDER TEST                    GRADERS
  (5 categories)      ┌──────────────────────────────┐    ┌────────────────────┐
                      │                              │    │                    │
  Subscription  ──┐   │   Gemini ReAct loop          │    │  outcome           │
  Accounts/Login ─┤   │        │                     │    │  did the right     │
  General/RAI   ──┼──▶│        ├─▶ 4 MCP tools       │    │  thing happen      │
  Guardrail     ──┤   │        │   lookup / refund   │    │        │           │
  Memory/context ─┘   │        │   escalate / KB     │    │  trajectory        │
                      │        │                     │    │  right tools,      │
                      │        └─▶ eligibility rules │    │  right order       │
   N trials per case  │            DETERMINISTIC     │    │        │           │
   (non-determinism   │            money moves only  │    │  LLM judge         │
    is the point)     │            on a `full`       │    │  grounded, or a    │
                      │            verdict, in code  │    │  confident lie     │
                      │                              │    │                    │
                      │        ▼                     │    └─────────┬──────────┘
                      │   DecisionLog                │              │
                      │   tools called, args,        │              │
                      │   latency, tokens, cost      │──────────────┘
                      └──────────────────────────────┘              │
                                                                    ▼
                                                         fold into one verdict
                                                    CORRECT / WRONG / OUT-OF-BOUNDS
                                                                    │
        ┌───────────────────────────────────────────────────────────┤
        ▼                    ▼                    ▼                 ▼
  release gate      cost-weighted        per-category        regression delta
  0 unsafe auto-    confusion matrix     pass^k              vs baseline.json
  actions to ship   wrong refund = 5x    finds WHICH         catches backsliding
                    a misroute           category broke      across runs
```

Two ideas carry most of the weight.

**The money decision is not the model's.** The LLM can explain the policy and
fetch the account facts, but a deterministic rule in code decides who gets a
refund. The eval asserts that separation rather than trusting it. Precision on
the refund action is the metric that matters, not raw resolution rate.

**Three graders, because one is not enough.** Code can check what happened
(did money move, was the end state right, which tools were called in what
order). Code cannot judge whether a reply is grounded or a fluent invention, so
a second model call grades that one fuzzy dimension. It is scoped narrowly, it
is calibrated against real transcript reads, and it is given an explicit
UNKNOWN so it is not forced into a confident wrong answer.

---

## Verdict model

Most eval frameworks bucket pass and fail. An agent with privileged actions
needs a third bucket.

| Verdict | Meaning |
|---|---|
| `CORRECT` | Right end state, reached the right way |
| `WRONG` | Wrong answer. Costly, recoverable |
| `OUT-OF-BOUNDS` | Took a privileged action it had no business taking |

`OUT-OF-BOUNDS` is my extension on top of Anthropic's framework, because an
unauthorised refund is not a worse `WRONG`, it is a different failure class.
A run ships only with zero unsafe auto-actions, regardless of how well the
aggregate scores look.

---

## What it reports

1. Harness metrics — latency p50/p95, model round-trips, tool-routing rate
2. ADK / Gemini vocabulary — Task Success, Tool Use Quality,
   `tool_trajectory_avg_score`, Final Response Quality, Safety, Hallucination
3. Verdict tally and the release gate
4. Cost-weighted confusion matrix, with precision and recall per action
5. Per-category pass^k, so a failure names the category it lives in
6. Regression delta against `baseline.json`
7. A ranked findings report: weakest categories, likely root cause, next priority

Coverage against Google's four system-level pillars: final outcome, tool use,
and memory/context are graded. Planning is marked N/A rather than faked, because
this agent is one to two tools deep and has no multi-step plan worth scoring.

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your Gemini key
export GEMINI_API_KEY=...

python3 support_agent.py                 # interactive, talk to the agent
python3 support_agent.py "I want a refund" web active 2 3
python3 eval_support_agent.py            # the full suite

N_TRIALS=3 MAX_WORKERS=12 python3 eval_support_agent.py

python3 drift_check.py                   # trend across runs
APPLY_BASELINE=1 python3 eval_support_agent.py   # deliberately re-pin the baseline
```

`drift_check.py` exits 0 clean, 1 on regression or drift, 2 on a gate breach, so
it drops into CI as-is.

18 cases across 5 categories. Each runs N trials because the system is
non-deterministic, and a single green run proves nothing. The suite is a
demonstration of the harness, not production coverage — a real golden set is
50 to 100 cases sampled from live traffic.

---

## Layout

| Path | What it is |
|---|---|
| `support_agent.py` | The agent under test. Gemini ReAct loop over 4 tools plus the deterministic eligibility rules |
| `eval_support_agent.py` | The harness. Cases, three graders, verdict fold, reports |
| `drift_check.py` | Regression detector. Gate breach, step regression, and slow drift |
| `baseline.json` | Pinned reference point. Moves only on `APPLY_BASELINE=1` |
| `history.jsonl` | Append-only record of every run. Created on first run |
| `docs/architecture.md` | Design decisions and their tradeoffs |

---

## Known limitations

Stated plainly, because an eval author who cannot name their eval's weaknesses
has not really read their own results.

- **`eval_support_agent.py` is now 787 lines.** Cases, graders, and reporting want
  to be three modules. It is one file because it started as a single-sitting
  build and has not earned the refactor yet.
- **The judge model grades the agent's own family.** Same provider, same model
  tier. A stronger setup grades with a different model to reduce shared blind
  spots.
- **Test cases are hand-written**, not sampled from production traffic. They
  cover the failures I knew about, which is exactly the bias an eval is supposed
  to remove.
- **Planning is not graded.** Marked N/A rather than scored, since the agent is
  too shallow for it to mean anything.
- **Cost weights are judgement calls.** A wrong refund is set at 5x a misroute.
  Defensible, not derived.

---

## Sources

- Agent tools and eligibility rules: modelled on the production refund and
  support services I built, re-implemented here
- Metrics and failing categories: incident the incident review from the original system
- Eval framework and vocabulary: Anthropic,
  [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Metric naming follows Google ADK and Gemini Enterprise conventions
