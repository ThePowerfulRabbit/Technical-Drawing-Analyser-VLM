import ollama
import json

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

# or i can just do this:
print(response["message"]["content"])
# output:
# {
#   "class": "Sheet"
# }
# Now lets try to extract the class name from this output and store it in a variable

# right now the output is a string, we will first parse the JSON string into python dictionary

json_data = json.loads(response["message"]["content"]) # Here we are converting the content of "content"(which is a string in json format) in to python dictionary

drawing_class = json_data["class"] # so now we can index the value of the key "class" and store it in a variable

print("Class output:", drawing_class)
# Class output: Sheet 
# Now we can easily ask the model to give more output while still being able to store class data in a separate variable