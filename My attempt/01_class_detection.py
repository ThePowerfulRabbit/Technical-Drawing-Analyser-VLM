import ollama

image_path = "image/002_redacted.png"

response = ollama.chat(
    
    model = 'qwen3-vl:8b',
    messages = [
        {
            "role" : "user",
            "content": "Analyze this engineering drawing",
            "images" : [image_path]
        }
    ]
    
)

print("Response:", response)