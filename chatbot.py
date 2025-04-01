from transformers import pipeline
import torch

# Load Falcon-7B-Instruct Model
generator = pipeline(
    "text-generation",
    model="tiiuae/Falcon-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16  # Use lower precision for performance
)


print("🤖 Falcon-7B Chatbot Ready! Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye! 👋")
        break

    # Generate response
    response = generator(user_input, max_length=100, truncation=True)
    print("\nFalcon-7B:", response[0]["generated_text"])
