from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Initialize model and tokenizer (using a smaller Qwen model)
MODEL_NAME = "Qwen/Qwen-1_8B-Chat"
print(f"Loading model: {MODEL_NAME}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/generate', methods=['POST'])
def generate():
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 50)  # Shorter default
    max_new_tokens = data.get('max_new_tokens', 30)  # Control new tokens only
    temperature = data.get('temperature', 0.7)
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=min(max_length, input_length + max_new_tokens),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            "prompt": prompt,
            "response": response,
            "generated_text": response[len(prompt):]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)