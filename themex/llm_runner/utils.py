import os
import csv
import json
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional
from pathlib import Path
import gc
import torch
import time
import tracemalloc
import functools
from jsonfinder import jsonfinder

from ..paths import PROMPT_DIR
from ..logger import get_logger

logger = get_logger()

def load_prompt(filepath: Union[str, Path]) -> str:
    """
    Load a prompt txt file.

    If `filepath` has no parent folder, auto prepend PROMPT_DIR.
    Accepts both str and Path as input.

    Args:
        filepath: filename or full path to prompt file (with or without .txt)
=
    Returns:
        str: prompt content
    """
    filepath = Path(filepath)

    if filepath.suffix != ".txt":
        filepath = filepath.with_suffix(".txt")

    if filepath.parent == Path('.'):
        filepath = PROMPT_DIR / filepath

    with filepath.open("r", encoding="utf-8") as f:
        return f.read()

def init_csv(csv_file, fieldnames):
    """Create a CSV file with header if it doesn't exist."""
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
        
def get_device():
    """Return best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_dtype_for_model(model_id):
    """Return dtype based on model size."""
    return torch.float16 if "7B" in model_id else torch.float32

def track_memory_and_time(func):
    """
    Decorator to measure the memory usage and execution time of a function.

    Returns:
        Tuple[result, metrics_dict]: result is the original return of the function,
        metrics_dict contains memory and time stats.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        baseline, _ = tracemalloc.get_traced_memory()
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = end - start
        increment = current - baseline
        
        print(f"\n🧠 Memory Usage:")
        print(f"  - Current:   {current / 1024 / 1024:.2f} MB")
        print(f"  - Peak:      {peak / 1024 / 1024:.2f} MB")
        print(f"  - Increment: {increment / 1024 / 1024:.2f} MB")
        
        print(f"\n⏱️ Time elapsed: {total_time:.2f} sec")
    
        return result, {
            "total_time_sec": round(total_time, 2),
            "current_mem_MB": round(current / 1024**2, 2),
            "peak_mem_MB": round(peak / 1024**2, 2),
            "increment_MB": round(increment / 1024**2, 2)
        }
    return wrapper

def count_tokens_safe(pipe, text: str) -> int:
    try:
        return len(pipe.tokenizer(text)["input_ids"])
    except Exception as e:
        print(f"[TokenCount Warning] Failed to tokenize. Fallback to word-count estimate. Error: {e}")
        return int(len(text.split()) * 1.3)

def cleanup_memory(*vars_to_clear, device=None):
    """Free memory after inference."""
    for var in vars_to_clear:
        del var
    gc.collect()
    
    if device is None:
        device = get_device()
    
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

def write_to_csv(csv_file, row: list, header: list = None):
    """Append a row to CSV. If file does not exist and header is provided, write header first."""
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists and header:
            writer.writerow(header)
        writer.writerow(row)

def write_to_excel(excel_file: str, row: list, header: list):
    """Append a row to Excel. If file does not exist, create it with header."""
    new_df = pd.DataFrame([row], columns=header)

    if os.path.exists(excel_file):
        # Append to existing file
        old_df = pd.read_excel(excel_file, engine="openpyxl")
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        # Create new file with header
        combined_df = new_df

    combined_df.to_excel(excel_file, index=False, engine="openpyxl")

def save_json_to_table(json_data: dict, output_path: Union[str, Path]) -> None:
    """
    Flatten a JSON object into a table and append it as a new row to an existing CSV or Excel file.

    Args:
        json_data (dict): The extracted JSON object.
        output_path (Union[str, Path]): The output file path. Supports both .csv and .xlsx formats.

    Returns:
        None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    record_path = None
    meta = []

    for key, value in json_data.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            record_path = key
            break
        elif isinstance(value, dict):
            meta.append(key)

    if record_path:
        df = pd.json_normalize(json_data, record_path, meta, sep=".")
    else:
        df = pd.json_normalize(json_data, sep=".")

    if output_path.suffix == ".csv":
        df.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
    elif output_path.suffix in [".xls", ".xlsx"]:
        if not output_path.exists():
            # If the file does not exist, create it and write headers
            with pd.ExcelWriter(output_path, mode="w", engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Sheet1", index=False)
        else:
            # If the file exists, append data
            with pd.ExcelWriter(output_path, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Sheet1", index=False, header=False, startrow=writer.sheets["Sheet1"].max_row)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

def json_to_dataframe(json_data: Dict) -> pd.DataFrame:
    """
    Convert a JSON object into a pandas DataFrame.

    Args:
        json_data (Dict): The extracted JSON object.

    Returns:
        pd.DataFrame: Flattened DataFrame.
    """
    record_path = None
    meta = []

    for key, value in json_data.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            record_path = key
        else:
            meta.append(key) 

    if record_path:
        df = pd.json_normalize(json_data, record_path, meta, sep=".")
    else:
        df = pd.json_normalize(json_data, sep=".")

    return df

def dataframe_to_excel(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """
    Save a DataFrame to an existing or new CSV or Excel file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (Union[str, Path]): The output file path. Supports both .csv and .xlsx formats.

    Returns:
        None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".csv":
        df.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)

    elif output_path.suffix in [".xls", ".xlsx"]:
        if not output_path.exists():
            with pd.ExcelWriter(output_path, mode="w", engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Sheet1", index=False)
        else:
            with pd.ExcelWriter(output_path, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
                startrow = writer.sheets["Sheet1"].max_row
                df.to_excel(writer, sheet_name="Sheet1", index=False, header=False, startrow=startrow)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

# def extract_json(text: str) -> Optional[str]:
#     """
#     Attempt to extract the first well-formed JSON block (from { to }) from raw LLM output text.

#     Args:
#         text (str): Raw output text from LLM, possibly containing explanation or other non-JSON content.

#     Returns:
#         Optional[str]: A cleaned JSON string block if found, otherwise None.

#     Notes:
#         - This function only extracts the first top-level `{...}` block.
#         - It does not guarantee valid JSON structure (use json.loads() to validate).
#         - Useful for pre-cleaning outputs before feeding to a JSON parser or Pydantic parser.
#     """
#     start_index = text.find("{")
#     if start_index == -1:
#         return None

#     last_brace_index = text.rfind("}")
#     if last_brace_index == -1:
#         return None

#     json_block = text[start_index:last_brace_index + 1].strip()
#     return json_block if json_block else None

def extract_json(text: str) -> str:
    """
    Attempt to extract the first JSON block from raw text.
    If extraction fails, return the original text and log a warning.

    Args:
        text (str): Raw output text from LLM.

    Returns:
        str: JSON string block if found, otherwise the original unmodified text.

    Example:
        input: "Here is the result:\n\n{\"a\": 1, \"b\": 2}"
        output: "{\"a\": 1, \"b\": 2}"
    """
    # start_index = text.find("{")
    # end_index = text.rfind("}")
    
    # if start_index != -1 and end_index != -1 and end_index > start_index:
    #     return text[start_index:end_index + 1].strip()
    
    json_blocks = [obj for  _, _, obj in jsonfinder(text) if obj is not None]
    last_json = json_blocks[-1] if json_blocks else None
    
    if not last_json:
        logger.warning("No JSON block found. Returning original text.")
        return text.strip()
    else:
        return json.dumps(last_json, indent=2)

def build_chat_messages(system_msg: str, user_msg: str) -> List[Dict[str, str]]:
    """
    Formats system and user messages into the OpenRouter chat message structure.

    Args:
        system_msg (str): System-level message (e.g., prompt instructions).
        user_msg (str): User message (e.g., comment with ID).

    Returns:
        List[Dict]: List of chat-style messages for LLM input.
    """
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def to_pairs(
    inputs: Union[List[str], List[Tuple[int, str]], Dict[int, str], pd.DataFrame],
    id_col: Optional[str] = None,
    text_col: Optional[str] = None
) -> List[Tuple[int, str]]:
    """
    Normalize various input formats into a standard list of {"id": ..., "text": ...} dictionaries.

    Args:
        inputs (Union[List[str], List[Tuple[int, str]], Dict[int, str], pd.DataFrame]):
            The input data in one of the supported formats:
              - List[str]: Automatically assigns index starting from 1.
              - List[Tuple[int, str]]: Explicit index-text pairs.
              - Dict[int, str]: Dictionary of {index: text}.
              - pd.DataFrame: Requires `index_col` and `text_col` to be specified.
        id_col (str, optional): If `inputs` is a DataFrame, the column name to use as text id.
        text_col (str, optional): If `inputs` is a DataFrame, the column name to use as text text.

    Returns:
        List[Dict[str, Any]]: A standardized list of {"id": ..., "text": ...} dictionaries.

    Raises:
        ValueError: If input format is invalid or required columns are missing.
    """
    if isinstance(inputs, pd.DataFrame):
        if not id_col or not text_col:
            raise ValueError("When passing a DataFrame, both `index_col` and `text_col` must be specified.")
        return [
            {"id": row[id_col], "text": row[text_col]}
            for _, row in inputs[[id_col, text_col]].iterrows()
        ]

    elif isinstance(inputs, dict):
        return [{"id": k, "text": v} for k, v in inputs.items()]

    elif all(isinstance(x, tuple) and isinstance(x[0], int) and isinstance(x[1], str) for x in inputs):
        return [{"id": k, "text": v} for k, v in inputs]

    elif all(isinstance(x, str) for x in inputs):
        return [{"id": i, "text": x} for i, x in enumerate(inputs, start=1)]

    else:
        raise ValueError("comments must be a List[str], List[Tuple[int, str]], Dict[int, str], or a DataFrame with index_col and text_col.")

def get_max_context_length(model, tokenizer):
    """
    Get the maximum context length for a given model and tokenizer.
    This function checks various attributes of the model and tokenizer to determine the maximum context length.
    Args:
        model: The model object (e.g., from HuggingFace).
        tokenizer: The tokenizer object associated with the model.
    Returns:
        int: The maximum context length.
    """
    # Check model config for max position embeddings
    # and other attributes that may indicate context length
    # Note: The order of checks is important; some models may have multiple attributes
    # that indicate context length.
    # The first non-None value will be used.
    context_length = getattr(model.config, "max_position_embeddings", None) or \
                     getattr(model.config, "n_positions", None) or \
                     getattr(model.config, "n_ctx", None) or \
                     getattr(model.config, "model_max_length", None) or \
                     tokenizer.model_max_length
                     
    return context_length
