import requests
import json

def test_qwen_api():
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Health: {response.json()}")
    
    # Test generation endpoint
    print("\nTesting generation endpoint...")
    payload = {
        "prompt": "Hello, how are you?",
        "max_length": 50
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
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_qwen_api()