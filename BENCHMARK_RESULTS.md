# Benchmark Results

These live measurements were collected on July 15, 2026 with Python 3.12.13. They are a small
engineering check, not a paper reproduction. Provider behavior is stochastic, only one generated
corpus seed was used, and prices are LiteLLM estimates.

## Reproduction

```bash
# Small deterministic tasks, three repetitions each
python benchmarks/compare_same_model.py MODEL --full --runs 3 --mode direct
python benchmarks/compare_same_model.py MODEL --full --runs 3 --mode rlm --max-depth 2

# One 100k-character deterministic task, three repetitions
python benchmarks/compare_same_model.py MODEL --generated-chars 100000 --seed 2026 --runs 3 --mode direct
python benchmarks/compare_same_model.py MODEL --generated-chars 100000 --seed 2026 --runs 3 --mode rlm --max-depth 2
```

The generated context has 100,098 characters and SHA-256
`1ee43f3b42f8db55369c337d2e37f1e7f61224abe3d538581a716865df8a6fcc`. Its exact answer key is:

```text
count=57 total_amount_cents=27404392 max_transaction_id=TX-0000883 max_amount_cents=993620
```

## Small tasks

The direct baseline is the correct choice for these short contexts. Both models passed every direct
run with one call. DeepSeek RLM also passed, but recursion added substantial latency and usage.

| Model and mode | Passed | p50 latency | Mean calls | Mean tokens | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5 mini, direct | 6/6 | 4.221 s | 1.00 | 598 | $0.0007996 |
| DeepSeek V4 Flash, direct | 6/6 | 2.635 s | 1.00 | 502 | $0.0000929 |
| DeepSeek V4 Flash, RLM depth 2 | 6/6 | 28.204 s | 7.17 | 6,875 | $0.0008071 |
| GPT-5 mini, RLM depth 2, incident task only | 1/3 | 167.341 s | 18.33 | 25,830 | $0.0262083 |

The GPT-5 mini RLM incident runs passed once and reached the configured 24-call cap twice. The full
small-task RLM suite was not repeated after that result because additional calls would not answer a
useful engineering question. The cap prevented a 25th provider request in both failed runs.

## Generated 100k-character task

RLM materially improved long-context correctness. GPT-5 mini also used about 78% fewer tokens and
46% lower estimated cost per run than its direct baseline. DeepSeek used about 61% fewer tokens and
17% lower estimated cost, but one RLM answer omitted the `TX-` prefix and failed exact grading.

| Model and mode | Passed | p50 latency | Mean calls | Mean tokens | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5 mini, direct | 0/3 | 35.278 s | 1.00 | 37,928 | $0.0088788 |
| GPT-5 mini, RLM depth 2 | 3/3 | 26.716 s | 5.00 | 8,224 | $0.0048132 |
| DeepSeek V4 Flash, direct | 0/3 | 20.738 s | 1.00 | 39,364 | $0.0012240 |
| DeepSeek V4 Flash, RLM depth 2 | 2/3 | 20.624 s | 7.33 | 15,209 | $0.0010156 |

The generated suite originally inherited a six-iteration benchmark cutoff. Raising only this suite
to ten iterations improved the combined RLM pass rate from 4/6 to 5/6: GPT-5 mini improved from 2/3
to 3/3, while DeepSeek remained at 2/3. This setting is now the generated-suite default; the global
24-call and 300-second caps remain unchanged.

## Interpretation

- Use direct completion for short contexts that fit comfortably in the model window.
- RLM is useful here when exact computation over a large context matters; the context stays outside
  model prompts and Python performs the aggregation.
- `max_depth=2` did not force recursion. The successful generated runs used the root REPL only, which
  is an important reminder that RLM's value includes externalized context and local computation, not
  only child-model calls.
- Three repetitions and one seed are enough to catch regressions, but not enough for broad model
  quality claims. Use more seeds and runs for release decisions.
