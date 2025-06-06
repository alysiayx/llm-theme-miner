import asyncio
import os
import time
import json
import resource
import logging
from json_repair import repair_json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from collections import OrderedDict
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatOpenAI, AzureChatOpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Union, Dict, Optional, Any, Literal
from langchain.schema import SystemMessage, HumanMessage
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
from transformers import pipeline

from .utils import track_memory_and_time, get_logger, extract_json, get_dtype_for_model, load_prompt, json_to_dataframe, dataframe_to_excel, write_to_csv, to_pairs
from .schema import GenerationResult
from ..prompt_registry import PromptSpec

load_dotenv()

logger = get_logger()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

def build_llm_hf(model_id: str = "meta-llama/Llama-3.2-3B-Instruct", generation_params: dict = None):
    """
    Build a HuggingFacePipeline LLM instance for text generation.

    Args:
        model_id (str): Name or path of the Hugging Face model.
        generation_params (dict): Parameters for text generation, such as:
            - max_new_tokens (int)
            - temperature (float)
            - do_sample (bool)
            - return_full_text (bool)

    Returns:
        HuggingFacePipeline: Wrapped generation pipeline usable by LangChain.
    """
    if generation_params is None:
        generation_params = dict(
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
            torch_dtype = get_dtype_for_model(model_id),
            return_full_text=False,
        )

    logger.info(f"Loading model: {model_id}")
    pipe = pipeline(
        task="text-generation",
        model=model_id,
        **generation_params
    )
    return HuggingFacePipeline(pipeline=pipe)

def build_chain_hf(
    pydantic_type,
    prompt_spec: PromptSpec, 
    model_id: Optional[str] = None,
    llm: Optional[HuggingFacePipeline] = None,
    generation_params: Optional[Dict[str, Any]] = None,
    ):
    """
    Build an LLMChain and OutputParser using a PromptSpec object.

    Args:
        llm (HuggingFacePipeline): Pre-built HuggingFace LLM.
        model_id: HuggingFace model ID, required if llm is not provided.
        prompt_spec (PromptSpec): PromptSpec containing template, input_vars, and inputs.
        pydantic_type (BaseModel): Pydantic model for structured output.

    Returns:
        Tuple[LLMChain, PydanticOutputParser]: Ready-to-use chain and parser.
    """
    if llm is None:
        if not model_id:
            raise ValueError("Either `llm`, `chain`, or `model_id` must be provided.")
        llm = build_llm_hf(model_id=model_id, generation_params=generation_params)
            
    parser = PydanticOutputParser(pydantic_object=pydantic_type)
    prompt = PromptTemplate(
        template=prompt_spec.template,
        input_variables=prompt_spec.input_vars,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    logger.info(f"Building chain for prompt: {prompt_spec.name}")
    return chain, parser

@track_memory_and_time
def run_chain_hf(
    prompt_spec,
    pydantic_type,
    model_id: Optional[str] = None,
    llm: Optional[HuggingFacePipeline] = None,
    chain: Optional[LLMChain] = None,
    generation_params: Optional[Dict[str, Any]] = None,
):
    """
    Run a HuggingFace LLM on a given prompt specification and parse the output.
    
    Args:
        prompt_spec (PromptSpec): Prompt specification containing template and input variables.
        pydantic_type (BaseModel): Pydantic model for structured output.
        model_id (str, optional): HuggingFace model ID.
        llm (HuggingFacePipeline, optional): Pre-built HuggingFace LLM.
        chain (LLMChain, optional): Pre-built LLMChain.
        generation_params (dict, optional): Parameters for text generation.
    
    Returns:
        dict: Dictionary containing parsed results, raw output, and input string length.
    
    Raises:
        ValueError: If neither `llm`, `chain`, nor `model_id` is provided.
    """
  
    if chain is None:
        if llm is None:
            if not model_id:
                raise ValueError("Either `llm`, `chain`, or `model_id` must be provided.")
            llm = build_llm_hf(model_name=model_id, generation_params=generation_params)
        chain, parser = build_chain_hf(llm, prompt_spec, pydantic_type)
    else:
        parser = PydanticOutputParser(pydantic_object=pydantic_type)

    logger.info("\n" + "="*60)
    logger.info(f"Task: {prompt_spec.name}")
    
    primary_key = prompt_spec.input_vars[0]
    input_str_len = len(prompt_spec.inputs[primary_key])
    
    logger.info(f"Input length: {input_str_len} characters")

    raw = chain.run(**prompt_spec.inputs)

    text = extract_json(raw)

    try:
        parsed = parser.parse(text)
        results = getattr(parsed, 'themes', getattr(parsed, 'root', []))
        logger.info(f"Parsed {len(results)} items from model output")
    except OutputParserException as e:
        logger.warning(f"Parsing failed: {e} \nRaw output:\n{raw}")
        results = []

    return {
        "results": results,
        "raw_output": text,
        "input_str_len": input_str_len
    }

semaphore = asyncio.Semaphore(8) 

@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.warning),
)
async def process_text_async(
    chat: ChatOpenAI,
    id: int,
    comment: str,
    system_msg: str,
    user_msg_template: str,
    output_filename: str,
    csv_logger_filepath: str,
    model_id: str,
    gen_args: dict
) -> dict:
    async with semaphore:
        logger.info(f"[async] Processing ID {id}")
     
        user_msg = user_msg_template.format(comment_id=id, comment=comment)
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ]

        start_time = time.time()
        start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            response = await chat.ainvoke(messages)
        except Exception as e:
            logger.error(f"Error while invoking model for comment {id}: {e}")

        end_time = time.time()
        end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        total_time = end_time - start_time
        current_mem_MB = end_mem / 1024
        increment_MB = (end_mem - start_mem) / 1024
        
        raw_output = response.content

        try:
            clean_output = extract_json(raw_output)
            json_data = json.loads(repair_json(clean_output))
            df = json_to_dataframe(json_data)
            df.insert(0, "Model", model_id)
            df.insert(0, "comment_id", id)
            dataframe_to_excel(df, output_filename)
        except Exception as e:
            logger.warning(f"Error converting JSON to DataFrame: {e} \nGenerated text: {raw_output}")

        try:
            result = GenerationResult(
                model_id=model_id,
                comment_id=id,
                output=clean_output,
                input_len=len(comment),
                total_time_sec=total_time,
                current_mem_MB=current_mem_MB,
                increment_MB=increment_MB,
                temperature=gen_args.get("temperature"),
            )

            print(f"\n⏱️ Time elapsed: {result.total_time_sec:.2f} sec")
            print(f"🧠 Memory Usage:")
            print(f"  - Current:   {result.current_mem_MB:.2f} MB")
            print(f"  - Increment: {result.increment_MB:.2f} MB")

            if csv_logger_filepath:
                result_dict = OrderedDict(sorted(asdict(result).items()))
                write_to_csv(
                    csv_logger_filepath,
                    list(result_dict.values()),
                    header=list(result_dict.keys())
                )
        except Exception as e:
            logger.warning(f"Error while logging metrics: {e}")

        return {"comment_id": id, "output": clean_output}

async def run_chain_async(
    model_name: str,
    provider: Literal["openrouter", "azure"],
    inputs: Union[List[str], List[Tuple[int, str]], Dict[int, str], pd.DataFrame],
    sys_tmpl: Union[str, Path],
    user_tmpl: Union[str, Path],
    api_key: Union[str, None] = None,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    max_concurrency: int = 8,
    max_retry_rounds: int = 2,
    retry_delay_sec: float = 2.0,
    output_filename: str = None,
    csv_logger_filepath: str = None,
    gen_args: dict = None,
    id_col: Optional[str] = None,
    text_col: Optional[str] = None,
    failed_output_path: str = "failed.csv",
) -> Tuple[List[dict], List[Tuple[int, str, str]]]:
    global semaphore
    semaphore = asyncio.Semaphore(max_concurrency)

    azure_endpoint = azure_endpoint or AZURE_OPENAI_ENDPOINT
    api_key = api_key or (
        OPENROUTER_API_KEY if provider == "openrouter" else
        AZURE_OPENAI_API_KEY if provider == "azure" else None
    )

    if provider == "openrouter":
        client = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            **gen_args,
        )
    elif provider == "azure":
        client = AzureChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            openai_api_type="azure",
            **gen_args,
        )

    system_msg = load_prompt(sys_tmpl)
    user_msg_template = load_prompt(user_tmpl)
    
    input_pairs = to_pairs(inputs, id_col=id_col, text_col=text_col)

    tasks = [
        process_text_async(client, cid, text, system_msg, user_msg_template, output_filename, csv_logger_filepath, model_name, gen_args)
        for cid, text in input_pairs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    cleaned_results = []
    failed = [(input_pairs[i][0], input_pairs[i][1], str(results[i]))
              for i in range(len(results)) if isinstance(results[i], Exception)]

    for i, res in enumerate(results):
        if not isinstance(res, Exception):
            cleaned_results.append(res)

    for retry_round in range(1, max_retry_rounds + 1):
        if not failed:
            break

        logger.warning(f"\nRetry round {retry_round}: {len(failed)} failed items")
        await asyncio.sleep(retry_delay_sec)

        retry_tasks = [
            process_text_async(client, cid, text, system_msg, user_msg_template, output_filename, csv_logger_filepath, model_name, gen_args)
            for cid, text, _ in failed
        ]
        retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

        next_failed = []
        for i, r in enumerate(retry_results):
            if isinstance(r, Exception):
                cid, cmt, _ = failed[i]
                next_failed.append((cid, cmt, str(r)))
                logger.error(f"Final failure ID {cid}: {r}")
            else:
                cleaned_results.append(r)

        failed = next_failed
    
    if failed:
        failed_df = pd.DataFrame(failed, columns=["comment_id", "comment", "error"])
        failed_df.to_csv(failed_output_path, index=False)
        logger.warning(f"Saved {len(failed)} failed items to {failed_output_path}")

    return cleaned_results, failed
