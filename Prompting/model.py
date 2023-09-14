from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import toml

config = toml.load("config.toml")

class Falcon:
    def __init__(self):
        self.llm = config['model']['llm']
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(self.llm, device_map="auto", trust_remote_code = True)
        self.pipeline = transformers.pipeline("text-generation", model=self.model,
            tokenizer=self.tokenizer, torch_dtype=torch.bfloat16,
            trust_remote_code=True, )

    def infer(self, sequence):
        results = self.pipeline(sequence, max_length = 200, do_sample = True, top_k=10, num_return_sequences=1)
        return results
