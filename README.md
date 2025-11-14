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
    ├── llm_runner                    # Core logic for calling LLMs
    │   ├── direct_runner.py
    │   ├── hf_runner.py
    │   ├── langchain_runner.py
    │   ├── schema.py
    │   └── utils.py
    ├── logger.py                     # Logging utilities
    ├── paths.py                      # Default paths and file naming logic
    ├── prompts/                      # Prompt template files
    └── utils.py                      # General utility functions


```

---

## 🚀 Quick Start

This framework supports flexible execution of large language models (LLMs) via local or remote backends. You can choose to run models on your own machine (```"execution_mode": "local"```) or through hosted APIs like Azure OpenAI and OpenRouter (```"execution_mode": "remote"```).

### 🔐 API Key Configuration

By default, API keys are loaded from a `.env` file:

```env
# For Azure OpenAI
AZURE_API_KEY=your_azure_key
AZURE_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_DEPLOYMENT_NAME=your_deployment_name

# For OpenRouter
OPENROUTER_API_KEY=your_openrouter_key
```

If not found, you can pass them as parameters:

```python
# For Azure
api_key="your_azure_key", azure_endpoint="https://...", deployment_name="your_deployment_name",

# For OpenRouter
api_key="your_openrouter_key"
```

### Example 1 - Using a local HuggingFace model

```python
from themex.llm_runner.direct_runner import run_llm
from pathlib import Path
from multiprocessing import Process

p = Process(target=run_llm, kwargs={
    "execution_mode": "local",
    "provider": "huggingface",
    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "inputs": ["This is an example comment."],
    "sys_tmpl": Path("./prompts/system_prompt.txt"),
    "user_tmpl": Path("./prompts/theming_sentiment.txt"),
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
from themex.llm_runner.direct_runner import run_llm
from pathlib import Path
from multiprocessing import Process

p = Process(target=run_llm, kwargs={
    "execution_mode": "remote",
    "provider": "azure",
    "model_id": "gpt-4.1",
    "api_version": "2025-01-01-preview",
    "inputs": ["This is an example comment."],
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

#### 💡 Note on Multi-Process Execution

The examples use Python's `multiprocessing.Process` to run each task in a separate subprocess.

This is **not mandatory**, but can be helpful, particularly when using **local models** (e.g. with `execution_mode="local"`).

Running in a subprocess ensures that memory (especially GPU memory) is fully released after the task completes, helping prevent memory leaks or out-of-memory errors during batch processing.

Feel free to adapt the structure for your own scheduling or orchestration needs.


### Example 3 - Using LangChain with OpenRouter as LLM Backend

```python
from themex.llm_runner.langchain_runner import run_chain_openrouter_async 

results, failed = await run_chain_openrouter_async(
    model_name="meta-llama/llama-3.3-70b-instruct:free",
    "inputs": ["This is an example comment."],
    sys_tmpl=Path("./prompts/system_prompt.txt"),
    user_tmpl=Path("./prompts/theming_sentiment.txt"),
    output_filename="output.csv",
    csv_logger_filepath="log.csv",
    gen_args={"temperature": 0.0}
)
```

---

## 📄 Output Format (Example)

The example output assumes that you are using the prompts included in this repository.

👉 [View prompt template on GitHub](https://github.com/alysiayx/llm-theme-miner/tree/main/themex/prompts)

In this setup, the prompt is written in a step-by-step manner and bundles multiple sub-tasks into a single instruction block. However, instead of executing everything sequentially, you can distribute these sub-tasks by launching them as separate `multiprocessing.Process` workers. Each worker handles one step of the prompt, and you can then aggregate their outputs at the end to form the final result. **What we found is the longer the prompt, the worse the performance.**



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
