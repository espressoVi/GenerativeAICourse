from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers
import torch
import toml

config = toml.load("config.toml")

class Falcon:
    def __init__(self):
        self.llm = config['model']['llm']
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm, padding_side = "left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.llm,
                                                          trust_remote_code = True,
                                                          torch_dtype = torch.bfloat16)
        self.model.to("cuda")
        self.generation_config = GenerationConfig(max_new_tokens = config['model']['MAX_TOKENS'],
                                                  temperature=config['model']['TEMP'],
                                                  do_sample = True,
                                                  num_return_sequences = 1,
                                                  pad_token_id = self.tokenizer.eos_token_id,
                                                  eos_token_id = self.tokenizer.eos_token_id)
    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            results = self.model.generate(input_ids = inputs['input_ids'],
                                          attention_mask = inputs['attention_mask'],
                                          generation_config = self.generation_config)
        results = self.tokenizer.decode(results[0], skip_special_tokens=True)
        return results
    def __call__(self, prompt):
        preamble = config['model']['PREAMBLE']
        prompt = preamble+prompt
        result = self.infer(prompt)
        return result[len(prompt):]

