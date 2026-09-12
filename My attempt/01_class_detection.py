import ollama

image_path = "image/004_redacted.png"

response = ollama.chat(
    
    model = 'qwen3-vl:8b',
    messages = [
        {
            "role" : "user",
            "content": "What type of component is shown in this engineering drawing? Answer with only the class.",
            "images" : [image_path]
        }
    ]
    
)

message = response["message"]
content = message["content"]
print(content)