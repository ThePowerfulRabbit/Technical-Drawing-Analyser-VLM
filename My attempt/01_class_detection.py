import ollama

image_path = "image/001_without_table.png"

response = ollama.chat(
    
    model = 'qwen3-vl:8b',
    messages = [
        {
            "role" : "user",
            "content": "Classify the component shown in this engineering drawing. Answer with only its class: Sheet or Tube",
            "images" : [image_path]
        }
    ]
    
)

message = response["message"]
content = message["content"]
print(content)