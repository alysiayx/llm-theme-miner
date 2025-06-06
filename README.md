# themex

[![PyPI version](https://img.shields.io/pypi/v/themex.svg)](https://pypi.org/project/themex/)
[![Python](https://img.shields.io/pypi/pyversions/themex.svg)](https://pypi.org/project/themex/)
[![License](https://img.shields.io/github/license/alysiayx/llm-theme-miner.svg?cacheSeconds=60)](https://github.com/alysiayx/llm-theme-miner/blob/main/LICENSE)

> ⚠️ **Caution**: This package is under active development and is currently **not stable**. Interfaces, file structure, and behaviour may change without notice.

**themex** is a flexible, modular framework designed to support large language model (LLM) tasks across social care, health, and research contexts — including **thematic extraction**, **sentiment analysis**, and more.

It supports both **local HuggingFace models** and **remote APIs** (such as Azure OpenAI), with configurable prompts, structured outputs, and logging.


---

## 📦 Installation

```bash
pip install themex
```

---

## 📁 Project Structure

```
llm-theme-miner/
├── poetry.lock
├── pyproject.toml
├── README.md
└── themex/
    ├── __init__.py
    ├── llm_runner/          # Core logic for calling LLMs
    ├── logger.py            # Logging utilities
    ├── paths.py             # Default paths and file naming logic
    ├── prompts/             # Prompt template files
    └── utils.py             # General utility functions
```

---

## 🚀 Quick Start

This framework is designed for flexible and extensible usage. Below are two minimal working examples.

### Example 1 - Using a local HuggingFace model

```python
from themex.llm_runner import run_llm
from pathlib import Path
from multiprocessing import Process

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
sys_tmpl = Path("./prompts/system_prompt.txt")
user_tmpl = Path("./prompts/theming_sentiment.txt")

p = Process(target=run_llm, kwargs={
    "execution_mode": "local",
    "provider": "huggingface",
    "model_id": model_id,
    "inputs": ["This is an example comment."],
    "sys_tmpl": sys_tmpl,
    "user_tmpl": user_tmpl,
    "gen_args": {
        "temperature": 0.7,
        "max_new_tokens": 300
    },
    "output_filename": "output.csv",
    "csv_logger_filepath": "log.csv",
    "extra_inputs": {
        "question": "What are the strengths and weaknesses in this case?",
        "domain": "Strength"
    }
})
p.start()
p.join()
```

---

### Example 2 - Using Azure OpenAI remotely

```python
p = Process(target=run_llm, kwargs={
    "execution_mode": "remote",
    "provider": "azure",
    "model_id": "gpt-4.1",
    "api_version": "2025-01-01-preview",
    "inputs": ["Another example comment."],
    "sys_tmpl": Path("./prompts/system_prompt.txt"),
    "user_tmpl": Path("./prompts/theming_sentiment.txt"),
    "gen_args": {
        "temperature": 0.4,
    },
    "output_filename": "azure_output.csv",
    "csv_logger_filepath": "azure_log.csv",
    "extra_inputs": {
        "question": "What are the strengths and weaknesses in this case?",
        "domain": "Strength"
    }
})
p.start()
p.join()
```

### 💡 Note on Multi-Process Execution

The examples use Python's `multiprocessing.Process` to run each task in a separate subprocess.

This is **not mandatory**, but can be helpful, particularly when using **local models** (e.g. with `execution_mode="local"`).

Running in a subprocess ensures that memory (especially GPU memory) is fully released after the task completes, helping prevent memory leaks or out-of-memory errors during batch processing.

Feel free to adapt the structure for your own scheduling or orchestration needs.

---

## 📄 Output Format (Example)

### 🧠 Field Definitions

- **`evidence`**: A verbatim quote from the original input text that supports or illustrates the identified `topic`. It serves as direct justification for the theme.
- **`root_cause`**: If the `impact` is `"negative"`, this field provides a short explanatory phrase reflecting the likely underlying structural, procedural, or systemic cause of the issue. It is **not a restatement of the evidence**, but an inferred explanation.


The framework saves structured outputs to CSV. Fields depend on prompt structure, but may include:

| comment_id | model_id | domain  | topic                   | evidence  | impact   | root_cause | sentiment |
|------------|----------|---------|--------------------------|-----------|----------|-------------|-----------|
| 1          | gpt-4.1  | Strength| Family Contact Support   | ...       | positive |             | positive   |

---

## 🧾 CSV Logger Output (Optional)

If `csv_logger_filepath` is specified, the framework will save an additional **per-call log file** capturing key runtime statistics, LLM behaviour, and inputs/outputs.

### ✅ When is it created?

- Only when `csv_logger_filepath` is explicitly set in `run_llm` parameters
- If omitted, no logger file is generated

### 📋 Example fields in the logger:

| comment_id | context_len | current_mem_MB | do_sample | extra_fields          | generated_token_len | increment_MB | input_len | input_token_len | max_new_tokens | model_id | output | peak_mem_MB | raw_output | system_prompt | temperature | tokens_per_sec | torch_dtype | total_time_sec | user_prompt |
|------------|-------------|----------------|-----------|------------------------|----------------------|--------------|-----------|------------------|----------------|----------|--------|--------------|-------------|----------------|-------------|----------------|--------------|----------------|--------------|
| id         |             | 1.57           |           | {"domain": "Strength"} | 55                   | 1.57         | 1         | 991              |                | gpt-4.1  | …      | 1.63         | …           | …              | 0.2         | 40.86          | None         | 1.35           | …            |

---

## ⚙️ Key Parameters

| Parameter              | Description |
|------------------------|-------------|
| `execution_mode`       | `"local"` or `"remote"` |
| `provider`             | `"huggingface"` / `"azure"` |
| `model_id`             | Model name or deployment ID |
| `api_version`          | Azure API version if applicable |
| `inputs`               | List of input strings |
| `sys_tmpl`             | Path to system prompt |
| `user_tmpl`            | Path to user prompt |
| `gen_args`             | Dict of generation parameters (e.g. temperature, max_tokens) |
| `output_filename`      | Where to save the result |
| `csv_logger_filepath`  | Filepath for detailed logs |
| `extra_inputs`         | Additional template fields (e.g. `domain`, `question`) |

---


<!-- ## 🧩 Prompt Templates

Place prompt templates in the `themex/prompts/` directory. You may use placeholders like `{domain}` or `{question}` inside prompts.

Example layout:

```
themex/prompts/
├── system_prompt.txt
├── theming_sentiment.txt
```

--- -->

## 🧪 Development Status

This project is still in development. Breaking changes are likely.  
**Use with caution** in production environments.

---

## 📬 Contact

To report bugs, request features, or contribute ideas, please open an issue on GitHub or contact the maintainer.

---