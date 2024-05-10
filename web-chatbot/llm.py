# Helper functions
import transformers
import torch
import tqdm
import numpy as np

# CUDA gradient accumulation
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()



class largeLanguageModel:

    def __init__(self, model_path, device, sys_prompt):
        self.sys_prompt = sys_prompt
        self.model = transformers.GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.model.to(self.device)

        self.history = self.sys_prompt

        # Hyperparameters
        self.temperature = 0.75
        self.top_k = 20
        self.max_length = 2000
        self.num_beams = 1
        self.length_penalty = 0.0
        self.no_repeat_ngram_size = 0
        self.early_stopping = False
        self.num_return_sequences = 1
        self.top_p = 1.0
    
    def generate(self, prompt_text:str):
        # Tokenize input
        self.history += "User: " + prompt_text
        batch = self.tokenizer([self.history], return_tensors='pt')

        # Copy prompt to model on GPU
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        
        # Generate the text
        out = self.model.generate(input_ids=batch['input_ids'], 
                                  attention_mask=batch['attention_mask'], 
                                  max_length=self.max_length, 
                                  temperature=self.temperature, 
                                  top_k=self.top_k, 
                                  do_sample=True,
                                  num_beams=self.num_beams,
                                  length_penalty=self.length_penalty,
                                  no_repeat_ngram_size=self.no_repeat_ngram_size,
                                  early_stopping=self.early_stopping,
                                  num_return_sequences=self.num_return_sequences,
                                  top_p=self.top_p )
        out = self.tokenizer.batch_decode(out.cpu())[0]

        # Update chat context
        input_length = len(self.history)
        self.history += "\nBot: " + out[input_length:input_length+100].split("User: ")[0] + "\n" #only save 100 characters of context from the bot
        
        # Return processed output
        return out[input_length:].split("User: ")[0]
    
    def set_temp(self, temperature:float):
        self.temperature = temperature
    
    def set_top_k(self, top_k: int):
        self.top_k = top_k

    def set_top_p(self, top_p: int):
        self.top_p = top_p
    
    def set_max_length(self, max_length: int):
        self.max_length = max_length

    def set_num_beams(self, num_beams: int):
        self.num_beams = num_beams

    def set_length_penalty(self, length_penalty: float):
        self.length_penalty = length_penalty

    def set_no_repeat_ngram_size(self, no_repeat_ngram_size: int):
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def set_early_stopping(self, early_stopping: bool):
        self.early_stopping = early_stopping

    def set_num_return_sequences(self, num_return_sequences: int):
        self.num_return_sequences = num_return_sequences