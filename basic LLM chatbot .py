from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Model Name
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load Model with FP16 & GPU Optimization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  
    device_map="auto"  # Automatically place on GPU
)

# Apply Torch Compile for Speed Boost
#model = torch.compile(model)

# Check if Model is Fully on GPU
print("Device Map:", model.hf_device_map)

# Function to Generate Response
def generate_response(prompt, max_length=50):
    torch.cuda.empty_cache()  # Free up VRAM before inference
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Chatbot Loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    start = time.time()
    response = generate_response(user_input)
    ended = time.time()

    print(f"LLM: {response}\n")
    print("Response Time:", round(ended - start, 2), "seconds")
