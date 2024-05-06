# GPT2-large Chat Loop
# Using system prompt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

model = AutoModelForCausalLM.from_pretrained("gpt2-large")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

model.to(device)

context = "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.:\n"

def generate_text(prompt):
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_length=200,  # Adjust based on desired output length
        do_sample=True,  # Enable sampling for diversity
        temperature=0.32,  # Adjust for desired randomness
        top_k=20,  # Adjust for diversity without sacrificing quality
        top_p=0.9,  # Adjust for diversity while maintaining quality
    )

    gen_text = tokenizer.batch_decode(generated_ids)[0]
    return gen_text

print(generate_text(context))

while True:
    prompt = input()

    print(generate_text(prompt))
    
