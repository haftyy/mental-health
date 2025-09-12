import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model repo
model_name = "tanusrich/Mental_Health_Chatbot"

# Load tokenizer and model in fp16
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,   # use half precision to save VRAM
    device_map="auto"            # let HF place layers across GPU/CPU if needed
)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Model loaded on: {device}")

def generate_response(user_input: str,
                      max_new_tokens: int = 200,
                      temperature: float = 0.7,
                      top_k: int = 50,
                      top_p: float = 0.9,
                      repetition_penalty: float = 1.2) -> str:
    prompt = f"User: {user_input}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("Assistant:")[-1].strip()
    return response

if __name__ == "__main__":
    print("🤖 Chatbot is ready! Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Take care!")
            break
        resp = generate_response(user_input)
        print("Chatbot:", resp)
def chatbot_reply(user_input: str) -> str:
    return generate_response(user_input)
