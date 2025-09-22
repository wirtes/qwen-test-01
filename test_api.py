import requests
import json

def test_qwen_api():
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Health: {response.json()}")
    
    # Test RAG-enabled generation
    print("\nTesting RAG generation...")
    payload = {
        "prompt": "What is machine learning?",
        "max_new_tokens": 50,
        "temperature": 0.7,
        "use_rag": True
    }
    
    response = requests.post(
        f"{base_url}/generate",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print(f"RAG Enabled: {result['rag_enabled']}")
        if result.get('context_used'):
            print(f"Context Used: {result['context_used'][:100]}...")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    # Test without RAG
    print("\nTesting without RAG...")
    payload["use_rag"] = False
    response = requests.post(
        f"{base_url}/generate",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Generated (no RAG): {result['generated_text']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_qwen_api()