import ollama

image_path = "image/001_without_table.png"

response = ollama.chat(
    
    model = 'qwen3-vl:8b',
    messages = [
        {
            "role" : "user",
            "content": "Classify the component shown in this engineering drawing. Return a JSON object with key : 'class', the value must be either 'Sheet' or 'tube'",
            "images" : [image_path]
        }
    ]
    
) # here respose is a python dictionary inside which the output of the model will be saved

#since its a nested dictionary the hierarchy is like this:
# response:
#     message:
#         role
#         content
#we want to access the content only

# message = response["message"] # retrieve information associated with "message key"
# content = message["content"] # retrieve information associated with "content" key
# print(content)

#or i can just do this:
print(response["message"]["content"])