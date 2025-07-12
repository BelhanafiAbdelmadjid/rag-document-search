from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
from dotenv import load_dotenv

class LLM:
    promptHat = """
        ### Instruction:
        You are a helpful AI assistant. Your task is to answer questions based on the provided context. 

        Please provide your response in the following format:
        Answer: <your answer here>

        ### Context:
    """ 
    promptQuestion = """
        ### Question:
    """ 
    promptTail = """
        ### Answer:
    """
    def __init__(self):
        self.model_name = None
        self.tokenizer = None
        self.model = None
        self.get_model_name()
        self._load_model()
    
    def get_model_name(self):
        """Load model name from environment variables and set the modelname property"""
        load_dotenv()
        self.model_name = os.getenv('MODEL_NAME', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    
    def _load_model(self):
        """Load the tokenizer and model"""
        if self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=f'./models/${self.model_name}_cache'
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                cache_dir=f'./models/${self.model_name}_cache'
            )

    def _contextThePrompt(self, context: str) -> str:
        """Prepare the prompt for the model"""
        
        return f'{self.promptHat}{context}'
    
    def _addQuestionInPrompt(self,contextedPrompt:str, question: str) -> str:
        """Add the question to the prompt"""
        return f'{contextedPrompt}{self.promptQuestion}{question}{self.promptTail}'

    def _buildPrompt(self, context: str, question: str) -> str:
        """Build the complete prompt for the model"""
        contextedPrompt = self._contextThePrompt(context)
        return self._addQuestionInPrompt(contextedPrompt, question)
    def generate_response(self, context: str, question: str):
        """Generate a response from the model based on the provided prompt"""
        start_time = time.time()
        inputs = self.tokenizer(self._buildPrompt(context,question), return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        end_time = time.time()
        generation_time = end_time - start_time
        print(f"\nGeneration time: {generation_time:.2f} seconds")
        print(f"Tokens per second: {100/generation_time:.2f} tokens/s")
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = result[len(self._buildPrompt(context, question)):].strip()
        return answer.strip()

if __name__ == "__main__":
    llm = LLM()

    context = input("Enter context: ")
    question = input("Enter question: ")

    answer = llm.generate_response(context, question)

    print(answer)

