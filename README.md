# Recursive Language Models (RLM)

Python implementation of Recursive Language Models for processing unbounded context lengths.

**Based on [the paper](https://alexzhang13.github.io/blog/2025/rlm/) by Alex Zhang and Omar Khattab (MIT, 2025)** | [arXiv](https://arxiv.org/abs/2512.24601)


## What is RLM?

RLM enables language models to process extremely long contexts (100k+ tokens) by:
- Storing context as a Python variable instead of in the prompt
- Allowing the LM to recursively explore and partition the context
- Avoiding "context rot" (performance degradation with long context)

Instead of this:
```python
llm.complete(prompt="Summarize this", context=huge_document)  # Context rot!
```

RLM does this:
```python
rlm = RLM(model="gpt-5-mini")
result = rlm.complete(
    query="Summarize this",
    context=huge_document  # Stored as variable, not in prompt
)
```

The LM can then peek, search, and recursively process the context adaptively.

## Installation

**Note:** This package is not yet published to PyPI. Install from source:

```bash
# Clone the repository
git clone https://github.com/ysz/recursive-llm.git
cd recursive-llm

# Install in editable mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

**Future:** Once published to PyPI, you'll be able to install with `pip install recursive-llm`

## Requirements

- Python 3.9 or higher
- An API key for your chosen LLM provider (OpenAI, Anthropic, etc.)
- Or a local model setup (Ollama, llama.cpp, etc.)

## Quick Start

```python
from rlm import RLM

# Initialize with any LLM
rlm = RLM(model="gpt-5-mini")

# Process long context
result = rlm.complete(
    query="What are the main themes in this document?",
    context=long_document
)
print(result)
```

### Usage and Cost Statistics

`RLM.stats` aggregates model calls across the complete recursion tree. Token usage comes from
provider responses, while cost is calculated on a best-effort basis using LiteLLM's model pricing
metadata.

```python
rlm = RLM(
    model="gpt-5-mini",
    recursive_model="deepseek/deepseek-v4-flash",
)
result = rlm.complete(query="Summarize this", context=document)

print(rlm.stats)
# {
#     "llm_calls": 11,
#     "root_calls": 3,
#     "recursive_calls": 8,
#     "prompt_tokens": 12500,
#     "completion_tokens": 3200,
#     "cached_tokens": 6000,
#     "estimated_cost_usd": 0.0047,
#     "by_model": {
#         "gpt-5-mini": {"calls": 3, ...},
#         "deepseek/deepseek-v4-flash": {"calls": 8, ...},
#     },
# }
```

`estimated_cost_usd` is `None` when LiteLLM has no pricing metadata for any completed call. Compare
`priced_calls` with `llm_calls` before treating the estimate as the full run cost.

### Live Model Comparison

The comparison script uses the same model for both root and recursive calls. It runs one small
recursive smoke test by default and reports latency, calls, tokens, and estimated cost:

```bash
python benchmarks/compare_same_model.py gpt-5-mini
python benchmarks/compare_same_model.py deepseek/deepseek-v4-flash
```

Use `--full` for the slower two-task suite. Live benchmarks make paid API calls and require the
corresponding provider keys.

## API Keys Setup

Copy the example environment file and add keys only for the providers you use:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=...
MOONSHOT_API_KEY=...
```

Use the LiteLLM provider prefix for non-OpenAI models, for example
`deepseek/deepseek-v4-flash` or `moonshot/kimi-k2.6`. This lets a hybrid RLM select the correct API
key for each model automatically.

Or pass directly in code:
```python
rlm = RLM(model="gpt-5-mini", api_key="sk-...")
```

## Supported Models

Works with 100+ LLM providers via LiteLLM:

```python
# OpenAI
rlm = RLM(model="gpt-5")
rlm = RLM(model="gpt-5-mini")

# Anthropic
rlm = RLM(model="claude-sonnet-4")
rlm = RLM(model="claude-sonnet-4-20250514")

# Ollama (local)
rlm = RLM(model="ollama/llama3.2")
rlm = RLM(model="ollama/mistral")

# llama.cpp (local)
rlm = RLM(
    model="openai/local",
    api_base="http://localhost:8000/v1"
)

# Azure OpenAI
rlm = RLM(model="azure/gpt-4-deployment")

# And many more via LiteLLM...
```

## Advanced Usage

### Two Models (Optimize Cost)

Use a cheaper model for recursive calls:

```python
rlm = RLM(
    model="gpt-5",              # Root LM (main decisions)
    recursive_model="gpt-5-mini"  # Recursive calls (cheaper)
)
```

### Async API

For better performance with parallel recursive calls:

```python
import asyncio

async def main():
    rlm = RLM(model="gpt-5-mini")
    result = await rlm.acomplete(query, context)
    print(result)

asyncio.run(main())
```

### Configuration

```python
rlm = RLM(
    model="gpt-5-mini",
    max_depth=5,         # Maximum recursion depth
    max_iterations=20,   # Maximum REPL iterations
    # Optional LiteLLM params: temperature, timeout, etc.
)
```

## How It Works

1. **Context is stored as a variable** in a Python REPL environment
2. **Root LM gets only the query** plus instructions
3. **LM can explore context** using Python code:
   ```python
   # Peek at context
   context[:1000]

   # Search with regex
   import re
   re.findall(r'pattern', context)

   # Recursive processing
   recursive_llm("extract dates", context[1000:2000])
   ```
4. **Returns final answer** via `FINAL(answer)` statement

## Examples

See the `examples/` directory for complete working examples:
- `basic_usage.py` - Simple complete with OpenAI
- `ollama_local.py` - Using Ollama locally
- `two_models.py` - Cost optimization with two models
- `long_document.py` - Processing 50k+ token documents
- `data_extraction.py` - Extract structured data from text
- `multi_file.py` - Process multiple documents
- `custom_config.py` - Advanced configuration

Run an example:
```bash
# Set your API key first
export OPENAI_API_KEY="sk-..."

# Run example
python examples/basic_usage.py
```

## Performance

### Paper Results

On OOLONG benchmark (132k tokens):
- GPT-5: baseline
- RLM(GPT-5-Mini): **33% better than GPT-5** at similar cost

### Our Benchmark Results

Tested with GPT-5-Mini on structured data queries (counting, filtering) across 5 different test cases:

**60k token contexts:**
- **RLM**: 80% accurate (4/5 correct)
- **Direct OpenAI**: 0% accurate (0/5 correct, all returned approximations)

RLM wins on accuracy. Both complete requests, but only RLM gives correct answers.

**150k+ token contexts:**
- **Direct OpenAI**: Fails (rate limit errors)
- **RLM**: Works (processes 1M+ tokens successfully)

**Token efficiency:** RLM uses ~2-3k tokens per query vs 95k+ for direct approach, since context is stored as a variable instead of being sent in prompts.

## Development

```bash
# Clone repository
git clone https://github.com/ysz/recursive-llm.git
cd recursive-llm

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=src/rlm --cov-report=term-missing

# Type checking
mypy src/rlm

# Linting
ruff check src/rlm

# Format code
black src/rlm tests examples
```

## Architecture

```
RLM
├── Core (async completion logic)
├── REPL Executor (safe code execution via RestrictedPython)
├── Prompt Builder (system prompts)
└── Parser (extract FINAL() answers)
```

Built on top of LiteLLM for universal LLM support.

## Limitations

- REPL execution is sequential (no parallel code execution yet)
- No prefix caching (future enhancement)
- Recursion depth is limited (configurable via `max_depth`)
- No streaming support yet

## Troubleshooting

### "Max iterations exceeded"
- Increase `max_iterations` parameter
- Simplify your query
- Check if the model is getting stuck in a loop

### "API key not found"
- Copy `.env.example` to `.env` and set the appropriate provider variable:
  - `OPENAI_API_KEY` for OpenAI
  - `DEEPSEEK_API_KEY` for DeepSeek
  - `MOONSHOT_API_KEY` for Kimi
- Or pass `api_key` parameter to RLM constructor

### "Model not found"
- Check model name format for your provider
- See LiteLLM docs: https://docs.litellm.ai/docs/providers

### Using Ollama
- Make sure Ollama is running: `ollama serve`
- Pull a model first: `ollama pull llama3.2`
- Use model format: `ollama/model-name`

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`pytest tests/`)
5. Follow code style (use `black` and `ruff`)
6. Submit a pull request

## Citation

This implementation is based on the RLM paper by Alex Zhang and Omar Khattab.

**To cite this implementation:**
```bibtex
@software{rlm_python,
  title = {recursive-llm: Python Implementation of Recursive Language Models},
  author = {Gvadzabia, Grisha},
  year = {2025},
  url = {https://github.com/ysz/recursive-llm}
}
```

**To cite the original paper:**
```bibtex
@misc{zhang2025rlm,
  title = {Recursive Language Models},
  author = {Zhang, Alex and Khattab, Omar},
  year = {2025},
  month = {October},
  url = {https://alexzhang13.github.io/blog/2025/rlm/},
  eprint = {2512.24601},
  archivePrefix = {arXiv}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Based on the Recursive Language Models paper by Alex Zhang and Omar Khattab from MIT CSAIL.

Built using:
- LiteLLM for universal LLM API support
- RestrictedPython for safe code execution

## Links

- **Paper**: https://alexzhang13.github.io/blog/2025/rlm/
- **arXiv**: https://arxiv.org/abs/2512.24601
- **LiteLLM Docs**: https://docs.litellm.ai/
- **Issues**: https://github.com/ysz/recursive-llm/issues
