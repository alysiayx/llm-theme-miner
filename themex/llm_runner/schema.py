from typing import List, Literal
from pydantic import BaseModel, Field, RootModel
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from collections import OrderedDict
from types import SimpleNamespace

@dataclass
class GenerationResult:
    """
    Container for all information related to a single LLM generation result.

    Attributes:
        model_id (Optional[str]): The ID of the model used for generation.
        input_len (Optional[int]): The length of the input text.
        context_len (Optional[int]): The context length of the model.
        output (str): The cleaned output text.
        system_prompt (str): The system prompt used for generation.
        user_prompt (str): The user prompt used for generation.
        raw_output (str): The raw output text from the model.
        total_time_sec (Optional[float]): Total time taken for generation in seconds.
        current_mem_MB (Optional[float]): Current memory usage in MB.
        peak_mem_MB (Optional[float]): Peak memory usage in MB.
        increment_MB (Optional[float]): Memory increment in MB.
        comment_id (Optional[int]): ID of the comment associated with the generation.
        input_token_len (Optional[int]): Number of input tokens.
        generated_token_len (Optional[int]): Number of generated tokens.
        tokens_per_sec (Optional[float]): Tokens generated per second.
        do_sample (Optional[bool]): Whether sampling was used during generation.
        temperature (Optional[float]): Temperature used for sampling.
        max_new_tokens (Optional[int]): Maximum number of new tokens to generate.
        torch_dtype (Optional[str]): Data type used for PyTorch tensors.
    """
    output: str
    raw_output: str
    system_prompt: str = None
    user_prompt: str = None
    model_id: Optional[str] = None
    input_len: Optional[int] = None
    context_len: Optional[int] = None
    total_time_sec: Optional[float] = None
    current_mem_MB: Optional[float] = None
    peak_mem_MB: Optional[float] = None
    increment_MB: Optional[float] = None
    comment_id: Optional[int] = None
    input_token_len: Optional[int] = None
    generated_token_len: Optional[int] = None
    tokens_per_sec: Optional[float] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None
    torch_dtype: Optional[str] = None
    
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    
    def __setattr__(self, key, value):
        if key in self.__dataclass_fields__:
            super().__setattr__(key, value)
        else:
            self.extra_fields[key] = value

    def __getattr__(self, item):
        if item in self.extra_fields:
            return self.extra_fields[item]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
    
    # def as_ordered_dict(self):
    #     return OrderedDict([
    #         ("model_id", self.model_id),
    #         ("comment_id", self.comment_id),
    #         ("output", self.output),
    #         ("total_time_sec", round(self.total_time_sec, 2))
    #     ])


# class GenerationResult(SimpleNamespace):
#     pass

class TopicItem(BaseModel):
    topic: str = Field(..., description="Short phrase for the topic")
    description: str = Field(..., description="1-2 sentence explanation")
    sentiment: Literal['positive', 'neutral', 'negative']
    # emotion: str

class TopicList(RootModel[List[TopicItem]]):
    pass

class TopicWrapper(BaseModel):
    ID: int
    themes: List[TopicItem]