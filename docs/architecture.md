# Design decisions

Each entry is a decision, the alternative rejected, and the reason. The
tradeoffs matter more than the code.

---

## The refund decision lives in code, not in the prompt

The model can read the policy, explain it, and fetch the account facts. It does
not decide who gets money. A deterministic rule takes the customer record and
returns `full`, `partial`, or `none`, and only a `full` verdict releases funds.

**Rejected:** letting the model reason about eligibility and call the refund tool
when it concludes the customer qualifies.

**Why:** prompt-reasoned eligibility fails open. A well-constructed sob story or
an injected instruction shifts the decision boundary, and the failure is silent
because the reply still sounds reasonable. A rule fails closed and is unit
testable. It also makes the refund path the cleanest-graded part of the suite,
since expected behaviour is computable rather than judged.

This is the habit that came from firmware, where you design the failure case
before you ship the feature.

---

## Three graders, scoped to what each can actually judge

| Grader | Judges | Implementation |
|---|---|---|
| Outcome | Did the right end state happen | Code, deterministic |
| Trajectory | Right tools, right arguments, right order | Code, deterministic |
| Quality | Grounded in policy, or a confident invention | Model call |

**Rejected:** grading everything with an LLM judge, which is the path of least
resistance.

**Why:** a model judge is not reproducible. Run it five times on identical input
and it can change its mind, which makes it useless as a release gate. Code
graders decide; the judge advises on the one dimension code cannot see. The
judge also gets an explicit `UNKNOWN` option, because forcing a binary verdict
on an ambiguous transcript manufactures false signal.

Same principle governs the harness generally: **deterministic checks block, model
checks advise.**

---

## OUT-OF-BOUNDS is a third bucket, not a worse WRONG

Standard framing is pass and fail. That collapses two failures that need
different responses.

A wrong answer is costly and recoverable: the customer is annoyed, a human
fixes it. An unauthorised privileged action is a different category: money
left the account, and no amount of aggregate accuracy makes that acceptable.

So the release gate is not a threshold on a score. It is: **zero unsafe
auto-actions, or the run does not ship.** A suite can post its best-ever Task
Success and still fail the gate.

---

## N trials per case

Every case runs N times, defaulting to more than one.

**Rejected:** one run per case, which is faster and reads cleaner.

**Why:** the system under test is non-deterministic. A single pass tells you the
agent *can* get it right, not that it *reliably does*. Per-category `pass^k`
exposes the case that succeeds four times in five, which is precisely the
failure a single run hides and a customer eventually finds.

---

## Cost-weighted confusion matrix instead of accuracy

Actions are not equally expensive when wrong. Refunding someone who was not
owed a refund is weighted 5x a routing mistake.

**Rejected:** overall accuracy, or an unweighted matrix.

**Why:** accuracy lets a model buy a good score by being right about cheap things
while being wrong about expensive ones. The weighted matrix also surfaces the
metric that actually matters here, which is **precision on the refund action**,
not recall. An agent that is over-cautious refers too much to humans. An agent
that is over-eager sends money. Those are not symmetric, and the reporting
should not pretend they are.

The 5x weight is a judgement call, not a derived figure. It is stated here so a
reviewer can argue with it.

---

## The baseline is pinned, and drift is checked separately

Two bugs lived here, one worse than the other.

The small one: `.gitignore` excluded `baseline.json`. A regression delta needs a
shared reference point, and a baseline generated per clone means every clone
measures against a different thing. It is committed now.

The real one: the harness rewrote the baseline on **every** run, unconditionally.
So each run was compared only to the run before it. A slide of 0.889 to 0.854 to
0.818 to 0.780 printed four small dips and never once surfaced the 11 points
lost. A gate that re-pins to whatever happened last cannot detect gradual decay,
which is the exact failure mode this harness was built to catch.

The fix has three parts:

1. `baseline.json` is **pinned**. It moves only when a human runs with
   `APPLY_BASELINE=1`.
2. Every run appends to `history.jsonl`, so the trend survives.
3. `drift_check.py` asks three separate questions of that history: did a gate
   break, is this run worse than the pinned baseline, and is the trend sliding
   even when no single step is large enough to fail.

**Drift tolerances are deliberately TIGHTER than step tolerances** (0.03 against
0.05 for `task_success`). The first version had them the other way round, which
made the drift check decoration: any slide big enough to trip drift had already
tripped step. The gap between the two numbers is the entire detection window for
slow decay, and it only exists if drift is the stricter of the two.

Drift also requires the decline to be roughly monotonic, so normal run-to-run
noise around a stable mean stays quiet.

---

## Structure, and what is wrong with it

`eval_support_agent.py` holds the cases, the graders, and the reporting in 768
lines. It should be three modules: `cases.py`, `graders.py`, `report.py`, with a
thin runner.

It is not split yet because the file works and is demoed live, and a
restructuring that breaks the flat `from support_agent import ...` import to win
a style point is a bad trade under time pressure. Recording it here so it reads
as a known debt rather than an oversight.

---

## What this harness does not do

- No production traffic sampling. Cases are hand-written, which encodes the
  author's existing knowledge of the failure modes and misses the unknown ones.
- No cross-family judge. Grading Gemini output with Gemini shares blind spots.
- No planning grade. The agent is one to two tools deep, so a planning score
  would be theatre. Reported as N/A.
- No CI integration. It runs on demand, not on every pull request.
