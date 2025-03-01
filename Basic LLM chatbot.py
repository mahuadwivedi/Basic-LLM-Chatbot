# Most basic LLM chatbot
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_name = "mistralai/Mistral-7B-v0.1"  #heavy with 7b paramteres
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage:
while True:
    user_input = input("You: ")
    start=time.time()
    if user_input.lower() in ["exit", "quit"]:
        break
    response = generate_response(user_input)
    print(f"LLM: {response}\n")
    ended=time.time()
    response_time=start-ended
    print("Response Time:",response_time)