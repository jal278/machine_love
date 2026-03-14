# Machine Love — Refactored Codebase

This is a clean, modular Python package refactoring of the experimental code for the paper:

> **Machine Love**
> Joel Lehman

The original code used deprecated OpenAI Completion API (`text-davinci-002/003`) and was organized as flat Jupyter-style `.py` scripts. This package reorganizes it into a testable `src/` layout using the modern OpenAI Chat Completions API.

---

## Quick Start

```bash
# Install with uv
cd code_new
uv sync

# Set your API key (required for LM experiments 3–6)
export OPENAI_API_KEY=sk-...

# Run Experiment 1 (no API key needed, ~20 seconds)
uv run python scripts/run_flourishing.py

# Run all tests
uv run pytest tests/ -v
```

---

## Package Structure

```
src/maslow/
├── gridworld/
│   ├── base.py         # MaslowGridworld + MaslowAgent
│   └── attachment.py   # AttachmentGridworld + AttachmentAgent + Memory + cycles
├── lm/
│   ├── base.py         # Abstract LMProvider interface
│   ├── openai_provider.py  # OpenAI Chat Completions (gpt-4o-mini default)
│   └── prompts.py      # All prompt templates centralized
├── experiments/
│   ├── base.py         # extract_data, run_env_experiment helpers
│   ├── flourishing.py  # Exp 1: supportive vs adversarial
│   ├── optimization.py # Exp 2: GA optimise engagement vs flourishing
│   ├── care.py         # Exp 3: LM-based care (growth detection)
│   ├── respect.py      # Exp 4: LM-based respect (user autonomy)
│   ├── attachment.py   # Exp 5: ASQ scoring + attachment dating simulation
│   └── knowledge.py    # Exp 6: self-awareness dynamics
└── utils/
    └── stats.py        # logprob_to_prob, prob_for_label, run_ttest
```

---

## The Maslow Gridworld

The gridworld simulates an agent satisfying needs according to Maslow's hierarchy:

| Level | Need | Activity |
|-------|------|----------|
| 0 | Physiological | Eat a meal |
| 1 | Safety | Go to work |
| 2 | Belonging | Spend time with friends |
| 3 | Esteem | Go to therapy |
| 4 | Self-actualization | Write poetry |

Each cell has:
- **Salience**: how attractive the cell appears (drives exploration)
- **Nutrition**: how much need is satisfied per step

**Adversarial cells** (superstimuli) are high-salience but low-nutrition — the agent is drawn to them but they don't genuinely satisfy needs.

---

## Experiments

### Experiment 1: Flourishing (`run_flourishing.py`)

**Hypothesis**: Adversarial environments (high-salience, low-nutrition belonging cells) undermine flourishing.

```bash
uv run python scripts/run_flourishing.py
```

Expected: adversarial avg_need < supportive avg_need, p < 0.05.

### Experiment 2: GA Optimisation (`run_optimization.py`)

**Hypothesis**: Optimising for engagement diverges from optimising for flourishing — a system maximising engagement may not maximise wellbeing.

```bash
uv run python scripts/run_optimization.py
```

Runs a genetic algorithm with two fitness objectives (engagement, needs). The tension between them is the central result.

### Experiment 3: Care (`run_care.py`)

**Hypothesis**: An LM acting as a "trusted friend" can detect whether an agent's recent life shows genuine growth vs. superstimuli engagement.

```bash
OPENAI_API_KEY=sk-... uv run python scripts/run_care.py
```

### Experiment 4: Respect (`run_respect.py`)

**Hypothesis**: A respectful AI system should respond differently to users with healthy vs. addictive activity patterns.

```bash
OPENAI_API_KEY=sk-... uv run python scripts/run_respect.py
```

### Experiment 5: Attachment (`run_attachment_date.py`)

**Hypothesis**: LMs can infer attachment styles from journal entries, and ASQ scores produced by an LM match expected patterns for each attachment style.

```bash
OPENAI_API_KEY=sk-... uv run python scripts/run_attachment_date.py
```

### Experiment 6: Knowledge (`run_knowledge.py`)

**Hypothesis**: A simulated relationship app that identifies attachment patterns helps insecurely-attached agents gain self-awareness and reduce harmful engagement.

```bash
# Requires attachment_results.pkl (from run_attachment_date.py) for app condition
OPENAI_API_KEY=sk-... uv run python scripts/run_knowledge.py
```

---

## Extending the Package

### New LM Provider

Implement `LMProvider` from `maslow.lm.base`:

```python
from maslow.lm.base import LMProvider

class MyProvider(LMProvider):
    def complete(self, prompt, temperature=0.5, max_tokens=300, stop=None):
        ...
    def chat(self, messages, temperature=0.5, max_tokens=300):
        ...
```

### New Experiment

1. Add a function in `src/maslow/experiments/my_exp.py`
2. It should accept an `LMProvider` and return a dict with `raw` key
3. Add a runner script in `scripts/run_my_exp.py`

---

## Configuration

- **API key**: set `OPENAI_API_KEY` environment variable
- **Model**: `OpenAIProvider(model="gpt-4o-mini")` (default) or any OpenAI-compatible model
- **No settings.py**: configuration is explicit, not global

---

## TODO

- **Logprob-based classification**: The original used `text-davinci-003` logprobs for attachment/yes-no classification. The current implementation uses text parsing (first word of response). Future work: restore logprob support via the `top_logprobs` parameter of the Chat Completions API (`logprobs=True, top_logprobs=5`).

- **Integration tests**: All tests are currently unit tests with mocked API calls. Add integration tests that hit the real API (tag with `pytest.mark.integration`).

- **Comparison with original results**: Add a script that loads original `.pkl` files from the parent directory and compares with new results.

- **Parallel experiment runs**: The `GeneticAlgorithm` has a `parallel` flag but pool-based parallelism may need adjustment for multiprocessing on macOS.
