from __future__ import annotations
import os
import time, tracemalloc, json
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional, Callable, Literal, Any
from dataclasses import asdict
from collections import OrderedDict
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from transformers import (
    pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
)
from json_repair import repair_json

from .utils import (
    to_pairs, load_prompt, build_chat_messages,
    extract_json, json_to_dataframe,
    dataframe_to_excel, write_to_csv,
    get_device, cleanup_memory, get_max_context_length
)

from .schema import GenerationResult

from ..logger import get_logger

logger = get_logger()

# Load .env if exists
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

# Not yet implemented
def _build_quant_config(load_in_4bit: bool):
    """Constructs a quantization config for 4-bit NF4 quantization if enabled."""
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

def _load_cached_model(model_id, device, *, use_pipeline, load_in_4bit):
    # quant_cfg = _build_quant_config(load_in_4bit)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device, torch_dtype="auto",
            # quantization_config=quant_cfg,
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cpu", torch_dtype="auto",
        )
    if use_pipeline:
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    else:
        model = model.to(device)
        pipe = None
    return pipe, model, tokenizer

def _setup_generator(
    execution_mode: str,
    provider: Optional[str],
    model_id: str,
    *,
    use_pipeline: bool,
    load_in_4bit: bool,
    api_key: str | None,
    azure_endpoint: str | None,
    deployment_name: str | None,
    api_version: str,
) -> Tuple[Callable, str]:
    if execution_mode == "local":
        device = get_device()
        pipe, model, tokenizer = _load_cached_model(
            model_id, device,
            use_pipeline=use_pipeline,
            load_in_4bit=load_in_4bit,
        )
        torch_dtype = str(model.dtype)
        context_len = get_max_context_length(model, tokenizer)

        def generate_fn(messages, gen_args):
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
            input_token_len = input_ids.shape[1]

            if use_pipeline:
                outputs = pipe(full_prompt, **gen_args)
                generated_text = outputs[0]["generated_text"]
            else:
                generated_ids = model.generate(input_ids, **gen_args)
                decoded = tokenizer.batch_decode(
                    generated_ids[:, input_token_len:], skip_special_tokens=True)
                generated_text = decoded[0]

            generated_token_len = len(tokenizer(generated_text).input_ids)
            return generated_text, input_token_len, generated_token_len

        return generate_fn, torch_dtype, context_len

    elif execution_mode == "remote":
        torch_dtype = None
        context_len = None

        def _call_api(messages, gen_args):
            if provider == "openrouter":
                client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                try:
                    rsp = client.chat.completions.create(
                        model=model_id, messages=messages, **gen_args)
                except Exception as e:
                    logger.error(f"OpenRouter API call failed: {e}")
            elif provider == "azure":
                client = AzureOpenAI(
                        api_key=api_key, api_version=api_version,
                        azure_endpoint=azure_endpoint)
                try:
                    rsp = client.chat.completions.create(
                        model=deployment_name or model_id, messages=messages, **gen_args)
                except Exception as e:
                    logger.error(f"Azure API call failed: {e}")
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            return rsp

        def generate_fn(messages, gen_args):
            rsp = _call_api(messages, gen_args)
            generated_text = rsp.choices[0].message.content
            input_token_len = rsp.usage.prompt_tokens
            generated_token_len = rsp.usage.completion_tokens
            return generated_text, input_token_len, generated_token_len

        return generate_fn, torch_dtype, context_len

    else:
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")

def _post_process_and_save(
    result: GenerationResult,
    output_filename: str | None,
    csv_logger_filepath: str | None,
):  
    print(result.raw_output)
    clean_output = extract_json(result.raw_output)
    
    try:
        json_data = json.loads(repair_json(clean_output))
        df = json_to_dataframe(json_data)
        df.insert(0, "domain", result.domain)
        df.insert(0, "model_id", result.model_id)
        df.insert(0, "comment_id", result.comment_id)
        if output_filename:
            dataframe_to_excel(df, output_filename)
            logger.info(f"Results saved to {output_filename}")
        result.output = clean_output
    except Exception as e:
        logger.warning(f"JSON->DataFrame failed: {e}\nRaw output: {clean_output}")

    logger.info(
        f"⏱ {result.total_time_sec:.2f}s | "
        f"🧠 {result.current_mem_MB:.1f}/{result.peak_mem_MB:.1f} MB | "
        f"🔢 {result.generated_token_len} tok"
    )

    if csv_logger_filepath:
        result_dict = OrderedDict(sorted(asdict(result).items()))
        write_to_csv(
            csv_logger_filepath,
            list(result_dict.values()),
            header=list(result_dict.keys())
        )
        logger.info(f"Logged to {csv_logger_filepath}")

def run_llm(
    execution_mode: Literal["local", "remote"],        # "local" or "remote"
    provider: Literal["openrouter", "azure", "huggingface"],    # For remote: "openrouter", "azure"
    model_id: str,
    inputs: Union[List[str], List[Tuple[int, str]], Dict[int, str], pd.DataFrame],
    sys_tmpl: Union[str, Path],
    user_tmpl: Union[str, Path],
    extra_inputs: Optional[Dict[str, Any]] = None,
    gen_args: Dict | None = None,
    output_filename: str | None = None,
    csv_logger_filepath: str | None = None,
    use_pipeline: bool = True,
    load_in_4bit: bool = False, # not yet implemented
    api_key: str | None = None,
    azure_endpoint: str | None = None,
    deployment_name: str | None = None,
    api_version: str = "2024-02-15-preview",
    id_col: str | None = None,
    text_col: str | None = None,
):
    logger.info(f"\n{'='*20}\nBackend: {execution_mode}   Provider: {provider}\nModel: {model_id}")
    
    # Argument validation
    if execution_mode == "local":
        if provider != "huggingface":
            logger.info("Provider has been set to 'huggingface' for local execution.")
            provider = "huggingface"
    elif execution_mode == "remote":
        if provider not in {"openrouter", "azure"}:
            raise ValueError("For remote execution, provider must be 'openrouter' or 'azure'.")
    else:
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")

    gen_args = gen_args or (
        {"max_new_tokens": 512, "temperature": 0.0, "do_sample": False}
        if execution_mode == "local" else {})
    
    azure_endpoint = azure_endpoint or AZURE_ENDPOINT
    api_key = api_key or (
        OPENROUTER_API_KEY if provider == "openrouter" else
        AZURE_API_KEY if provider == "azure" else None
    )

    generate_fn, torch_dtype, context_len = _setup_generator(
        execution_mode, provider, model_id,
        use_pipeline=use_pipeline,
        load_in_4bit=load_in_4bit,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        deployment_name=deployment_name,
        api_version=api_version,
    )

    input_pairs = to_pairs(inputs, id_col=id_col, text_col=text_col)

    for idx, item in enumerate(input_pairs, start=1):
        id = item["id"]
        text = item["text"]
        
        logger.info(f"\n🟩 Processing {idx}/{len(input_pairs)}  ID={id}")

        system_msg = load_prompt(sys_tmpl)
        inputs = {"comment_id": id, "comment": text}
        if extra_inputs:
            inputs.update(extra_inputs)

        user_msg = load_prompt(user_tmpl).format(**inputs)
        messages = build_chat_messages(system_msg, user_msg)

        # Start memory tracking and timing
        tracemalloc.start()
        baseline, _ = tracemalloc.get_traced_memory()
        start = time.time()

        output_text, input_token_len, generated_token_len = generate_fn(messages, gen_args)

        # Stop memory tracking and timing
        end = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = GenerationResult(
            model_id=model_id,
            comment_id=id,
            system_prompt=system_msg,
            user_prompt=user_msg,
            output=output_text,
            raw_output=output_text,
            input_len=len(text.split()),
            context_len=context_len,
            input_token_len=input_token_len,
            generated_token_len=generated_token_len,
            total_time_sec=end-start,
            tokens_per_sec=generated_token_len/ (end-start) if end-start>0 else 0,
            current_mem_MB=current/1024**2,
            peak_mem_MB=peak/1024**2,
            increment_MB=(current-baseline)/1024**2,
            do_sample=gen_args.get("do_sample"),
            temperature=gen_args.get("temperature"),
            max_new_tokens=gen_args.get("max_new_tokens"),
            torch_dtype=str(torch_dtype),
        )
        result.domain = extra_inputs.get("domain", "")
        _post_process_and_save(result, output_filename, csv_logger_filepath)

    logger.info("\n✅ All comments processed.\n")
    