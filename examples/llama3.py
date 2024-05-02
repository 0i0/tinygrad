#!/usr/bin/env python3
import os, sys, traceback
sys.path.append(os.getcwd())
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from io import StringIO
from contextlib import redirect_stdout
from tinygrad import Tensor, nn, Device, dtypes
from tinygrad.helpers import Timing, colored, getenv, fetch
from extra.models.llama import Transformer, convert_from_huggingface
from typing import List, Dict, Any

class TiktokenWrapper:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]
    def encode(self, text):
        return self.model.encode(text,allowed_special="all")

    def decode(self, tokens):
        return self.model.decode(tokens,errors="replace")

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id


def fix_bf16(weights:Dict[Any, Tensor]):
  if getenv("SUPPORT_BF16", 1):
    # TODO: without casting to float16, 70B llama OOM on tinybox.
    return {k:v.float().half() if v.dtype == dtypes.bfloat16 else v for k,v in weights.items()}
  # TODO: check if device supports bf16
  return {k:v.llvm_bf16_cast(dtypes.half).to(v.device) if v.dtype == dtypes.bfloat16 else v for k,v in weights.items()}


# **** helper functions ****
def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.startswith("tok_embeddings.") or name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

if __name__ == "__main__":
  Tensor.no_grad = True

  with Timing("create model: "):
    model = Transformer(4096, 14336, n_heads=32, n_layers=32, norm_eps=1e-5, vocab_size=128256, n_kv_heads=8, max_context=4096,rope_theta=500000.0, jit=getenv("JIT", 1))

  with Timing("lodaing weights: "):
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B
    weights = concat_weights([nn.state.torch_load("weights/llama3/consolidated.00.pth")])
  
  with Timing("cast weights: "):
    weights = fix_bf16(weights)

  with Timing("weights -> model: "):
    nn.state.load_state_dict(model, weights, strict=False)
  
  # https://huggingface.co/meta-llama/Meta-Llama-3-8B
  spp = TiktokenWrapper(model_path="weights/llama3/tokenizer.model")
  IM_END = 128000
  IM_START = 128001
  def encode_prompt(k, v): return [IM_START]+spp.encode(f"{k}\n{v}")+[IM_END]+spp.encode("\n")
  def start_prompt(k): return [IM_START]+spp.encode(f"{k}\n")
  def output(outputted, toks, color):
    cur = spp.decode(toks)[len(outputted):]
    sys.stdout.write(colored(cur, color))
    sys.stdout.flush()
    outputted += cur
    return outputted

  # *** app below this line ***

  toks = [spp.bos_id()] + encode_prompt("system", "You are Quentin. Quentin is a useful assistant who writes Python code to answer questions. He keeps the code as short as possible and doesn't read from user input")

  PROMPT = getenv("PROMPT", 1)
  temperature = getenv("TEMP", 0.7)

  start_pos = 0
  outputted = output("", toks, "green")
  turn = True
  while 1:
    if PROMPT:
      toks += encode_prompt("user", input("Q: ")) + start_prompt("assistant")
    else:
      toks += start_prompt("user" if turn else "assistant")
      turn = not turn
    old_output_len = len(outputted)
    while 1:
      tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).item()
      start_pos = len(toks)
      if tok != 128256:
        toks.append(tok)
      outputted = output(outputted, toks, "blue" if not turn else "cyan")
      if tok == IM_END: break
      if tok == spp.eos_id(): break
      new_output = outputted[old_output_len:]

      if new_output.endswith("```") and '```python\n' in new_output:
        python_code = new_output.split('```python\n')[1].split("```")[0]
        # AI safety. Warning to user. Do not press y if the AI is trying to do unsafe things.
        if input(colored(f" <-- PYTHON DETECTED, RUN IT? ", "red")).lower() == 'y':
          my_stdout = StringIO()
          try:
            with redirect_stdout(my_stdout): exec(python_code)
            result = my_stdout.getvalue()
          except Exception as e:
            result = ''.join(traceback.format_exception_only(e))
          toks += spp.encode(f"\nOutput:\n```\n{result}```")
          outputted = output(outputted, toks, "yellow")
          old_output_len = len(outputted)
    print("")