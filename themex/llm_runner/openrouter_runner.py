# NOTE:
# OpenRouter Limits:
# 20 requests/minute
# 50 requests/day
# 1000 requests/day with $10 credit balance

# WARNING: The data will be used for training as mentiond in "https://openrouter.ai/settings/privacy"

import os
import time
import pandas as pd
from IPython.display import clear_output
import tracemalloc
import json
from tqdm import tqdm
from json_repair import repair_json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import (
    load_prompt,
    get_logger,
    track_memory_and_time,
    extract_json,
    build_chat_messages,
    to_pairs,
    write_to_csv,
    json_to_dataframe,
    dataframe_to_excel,
)
from .schema import GenerationResult, TopicWrapper

logger = get_logger()

# Load .env if exists
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def call_openrouter_model(
    model_name: str,
    messages: List[Dict[str, str]],
    gen_args: Dict = None,
    api_key: str = None,
) -> str:
    """
    Sends a chat completion request to OpenRouter API.

    Args:
        model_name (str): The OpenRouter model ID (e.g., 'mistralai/mistral-7b-instruct').
        messages (List[Dict]): A list of chat messages with roles: system/user/assistant.
        gen_args (Dict): Optional generation parameters (max_tokens, temperature, etc).
        api_key (str): Optional override for API key.

    Returns:
        str: The generated assistant message content.

    Raises:
        RuntimeError: If the API call fails.
    """
    api_key = api_key or OPENROUTER_API_KEY
    
    logger.info(f"Calling OpenRouter with model: {model_name}")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    try:
        response = client.chat.completions.create(
          model=model_name,
          messages=messages,
          **gen_args
        )
    except Exception as e:
        logger.error(f"OpenRouter API call failed: {e}")
        
    return response.choices[0].message.content

def run_llm_openrouter(
    model_id: str,
    comments: Union[List[str], List[Tuple[int, str]], Dict[int, str], pd.DataFrame],
    sys_tmpl: Union[str, Path],
    user_tmpl: Union[str, Path],
    gen_args: Dict = None,
    id_col: Optional[str] = None,
    text_col: Optional[str] = None,
    output_filename: str = None,
    csv_logger_filepath: str = None,
) -> List[Dict]:
    """
    Runs OpenRouter LLM on a list of (id, comment) pairs.
    
    Args:
        model_id (str): OpenRouter model ID (e.g., 'mistralai/mistral-7b-instruct').
        comments (Union[List[str], List[Tuple[int, str]], Dict[int, str], pd.DataFrame]): Comments to process.
        sys_tmpl (Union[str, Path]): Path to system prompt template file.
        user_tmpl (Union[str, Path]): Path to user prompt template file.
        gen_args (Dict): Generation arguments (e.g., max_tokens, temperature).
        id_col (Optional[str]): Column name for ID if `comments` is DataFrame.
        text_col (Optional[str]): Column name for text if `comments` is DataFrame.
        output_filename (str): Excel file path to save extracted output.
        csv_logger_filepath (str): CSV file path to log metrics per generation.

    Returns:
        List[Dict]: A list of result dictionaries (including ID and output text).
    """
    results = []
    
    logger.info(f"\n{'==' * 10}\n Using model: {model_id}")
    
    comment_pairs = to_pairs(comments, id_col=id_col, text_col=text_col)
     
    for idx, (comment_id, comment) in enumerate(comment_pairs, start=1):
        clear_output(wait=True)
        logger.info(f"\n🟩 Processing comment {idx}/{len(comment_pairs)} (ID={comment_id})...")
        
        system_msg = load_prompt(sys_tmpl)
        user_msg = load_prompt(user_tmpl).format(comment_id=comment_id, comment=comment)
        
        messages = build_chat_messages(system_msg=system_msg, user_msg=user_msg)
        
        tracemalloc.start()
        baseline_current, _ = tracemalloc.get_traced_memory()
        start_time = time.time()

        output = call_openrouter_model(model_id, messages, gen_args)
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        increment = (current - baseline_current) / 1024 / 1024
        current_mem = current / 1024 / 1024
        peak_mem = peak / 1024 / 1024
        total_time = end_time - start_time
        
        clean_output = extract_json(output)

        try:
            json_data = json.loads(repair_json(clean_output))
            df = json_to_dataframe(json_data)
            df.insert(0, "Model", model_id)
            dataframe_to_excel(df, output_filename)
        except Exception as e:
            logger.warning(f"Error converting JSON to DataFrame: {e} \nGenerated text: {clean_output}")
            continue
        
        results.append({
            "comment_id": comment_id,
            "output": clean_output,
        })
        
        result = GenerationResult(
            model_id=model_id,
            comment_id=comment_id,
            total_time_sec=total_time,
            current_mem_MB=current_mem,
            peak_mem_MB=peak_mem,
            increment_MB=increment,
            output=clean_output,
            temperature=gen_args.get("temperature"),
        )
        
        print(f"\n⏱️ Time elapsed: {result.total_time_sec:.2f} sec")
        print(f"🧠 Memory Usage:")
        print(f"  - Current:   {result.current_mem_MB:.2f} MB")
        print(f"  - Peak:      {result.peak_mem_MB:.2f} MB")
        print(f"  - Increment: {result.increment_MB:.2f} MB")
        
        result_dict = asdict(result)
        
        write_to_csv(
            csv_logger_filepath,
            list(result_dict.values()),
            header=list(result_dict.keys())
        )

    return results

# TODO: Not yet tested
def run_openrouter_batch(
    model_id: str,
    comments: Union[List[str], List[Tuple[int, str]], Dict[int, str], pd.DataFrame],
    sys_tmpl: Union[str, Path],
    user_tmpl: Union[str, Path],
    gen_args: Dict = None,
    id_col: Optional[str] = None,
    text_col: Optional[str] = None,
    output_filename: Optional[str] = None,
    csv_logger_filepath: Optional[str] = None,
    max_workers: int = 8,
) -> List[Dict]:
    """
    Runs OpenRouter LLM in parallel on a batch of comments using multithreading.
    
    Args:
        model_id (str): OpenRouter model ID (e.g., 'mistralai/mistral-7b-instruct').
        comments (Union[List[str], List[Tuple[int, str]], Dict[int, str], pd.DataFrame]): Comments to process.
        sys_tmpl (Union[str, Path]): Path to system prompt template file.
        user_tmpl (Union[str, Path]): Path to user prompt template file.
        gen_args (Dict): Generation arguments (e.g., max_tokens, temperature).
        id_col (Optional[str]): Column name for ID if `comments` is DataFrame.
        text_col (Optional[str]): Column name for text if `comments` is DataFrame.
        output_filename (str): Excel file path to save extracted output.
        csv_logger_filepath (str): CSV file path to log metrics per generation.
        max_workers (int): Number of threads to use for parallel processing.

    Returns:
        List of dicts containing GenerationResult fields.
    """
    comment_pairs = to_pairs(comments, id_col=id_col, text_col=text_col)

    logger.info(f"\n{'=' * 20} Running OpenRouter in parallel with {max_workers} workers {'=' * 20}")

    def process_one(comment_id: int, comment: str) -> Optional[Dict]:
        try:
            system_msg = load_prompt(sys_tmpl)
            user_msg = load_prompt(user_tmpl).format(comment_id=comment_id, comment=comment)
            messages = build_chat_messages(system_msg, user_msg)

            tracemalloc.start()
            baseline_current, _ = tracemalloc.get_traced_memory()
            start_time = time.time()

            output = call_openrouter_model(model_id, messages, gen_args)

            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            increment = (current - baseline_current) / 1024 / 1024
            current_mem = current / 1024 / 1024
            peak_mem = peak / 1024 / 1024
            total_time = end_time - start_time

            clean_output = extract_json(output)
            json_data = json.loads(repair_json(clean_output))
            df = json_to_dataframe(json_data)
            df.insert(0, "Model", model_id)
            dataframe_to_excel(df, output_filename)

            result = GenerationResult(
                model_id=model_id,
                comment_id=comment_id,
                total_time_sec=total_time,
                current_mem_MB=current_mem,
                peak_mem_MB=peak_mem,
                increment_MB=increment,
                output=clean_output,
                temperature=gen_args.get("temperature"),
            )

            result_dict = asdict(result)
            write_to_csv(
                csv_logger_filepath,
                list(result_dict.values()),
                header=list(result_dict.keys())
            )

            return result_dict

        except Exception as e:
            logger.warning(f"❌ Error processing comment ID {comment_id}: {e}")
            return None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one, cid, ctext) for cid, ctext in comment_pairs]
        for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing batch")):
            clear_output(wait=True)
            result = future.result()
            if result:
                results.append(result)

    return results