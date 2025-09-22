from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from rag_system import RAGSystem  # Temporarily disabled

app = Flask(__name__)

# Initialize RAG system (temporarily disabled)
# rag = RAGSystem()
rag = None

# Initialize model and tokenizer (using a lightweight compatible model)
import os
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
print(f"Loading model: {MODEL_NAME}")

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "rag_loaded": False,
        "document_chunks": 0
    })

@app.route('/load_document', methods=['POST'])
def load_document():
    data = request.json
    file_path = data.get('file_path', '')
    
    if not file_path:
        return jsonify({"error": "No file_path provided"}), 400
    
    success = rag.load_markdown_document(file_path)
    if success:
        return jsonify({
            "message": "Document loaded successfully",
            "chunks": len(rag.documents)
        })
    else:
        return jsonify({"error": "Failed to load document"}), 500

@app.route('/generate', methods=['POST'])
def generate():
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 30)
    temperature = data.get('temperature', 0.7)
    use_rag = data.get('use_rag', True)
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # Get RAG context if available and requested (temporarily disabled)
        context = ""
        enhanced_prompt = prompt
        
        # Tokenize input
        inputs = tokenizer(enhanced_prompt, return_tensors="pt", truncation=True, max_length=512)
        input_length = inputs.input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (exclude the input)
        new_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            "prompt": prompt,
            "response": full_response,
            "generated_text": generated_text.strip(),
            "context_used": None,
            "rag_enabled": False
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Auto-load document if it exists (temporarily disabled)
    # doc_path = os.getenv("RAG_DOCUMENT", "/app/knowledge.md")
    # if os.path.exists(doc_path):
    #     print(f"Auto-loading RAG document: {doc_path}")
    #     rag.load_markdown_document(doc_path)
    
    app.run(host='0.0.0.0', port=8000, debug=False)