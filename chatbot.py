# chatbot.py
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from PIL import Image

image_model_name = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
processor = AutoImageProcessor.from_pretrained(image_model_name)
image_model = AutoModelForImageClassification.from_pretrained(image_model_name)


text_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForCausalLM.from_pretrained(
    text_model_name, torch_dtype=torch.float16, device_map="auto"
)

def chatbot_reply(user_input: str, image_path: str = None) -> str:
    """
    If image_path is provided -> classify skin disease.
    Otherwise -> generate text reply.
    """

    if image_path:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(image_model.device)
            outputs = image_model(**inputs)
            predicted_class = outputs.logits.argmax(-1).item()
            label = image_model.config.id2label[predicted_class]
            return f"🩺 My analysis suggests this looks like **{label}**. Please consult a dermatologist for confirmation."
        except Exception as e:
            return f"⚠️ Error analyzing image: {str(e)}"


    try:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(text_model.device)

        outputs = text_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reply
    except Exception as e:
        return f"⚠️ Error generating reply: {str(e)}"


if __name__ == "__main__":
    print("💬 Text Test:")
    print(chatbot_reply("Hello, I feel stressed these days."))

    print("\n🖼️ Image Test:")
    print(chatbot_reply("", image_path="test_skin.jpg")) 
