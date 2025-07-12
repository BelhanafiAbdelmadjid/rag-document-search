from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f'./models/${model_name}_cache')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=f'./models/${model_name}_cache')

prompt = """
### Instruction:
You are a helpful AI assistant. Your task is to answer questions based on the provided context. 

Please provide your response in the following format:
Answer: <your answer here>

### Context:
LangChain is an open-source framework that simplifies building applications powered by large language models (LLMs). It provides tools to connect LLMs with external data sources, build chains of prompts and actions, and integrate retrieval-augmented generation (RAG) into applications. Developers use LangChain to create chatbots, knowledge assistants, and other AI apps that require combining LLM capabilities with private data.

### Question:
What is LangChain mainly used for?

### Answer:
"""



inputs = tokenizer(prompt, return_tensors="pt")

start_time = time.time()

outputs = model.generate(**inputs, max_new_tokens=100)

end_time = time.time()
generation_time = end_time - start_time

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = result[len(prompt):].strip()
print(answer)
print(f"\nGeneration time: {generation_time:.2f} seconds")
print(f"Tokens per second: {100/generation_time:.2f} tokens/s")
