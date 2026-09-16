import ollama
import json

image_path = "image/002_redacted.png"
print("Sending image to model...")
response = ollama.chat(
    model = "qwen3-vl:8b-instruct", #changing the model to tne instruct varient because the normal one just kept thinking and did nothing other than thinking
    messages = [
        {
            "role" : "user",
            "content" : """Identify the following regions in the given engineering drawing:  flat_pattern,
                            orthographic_view, isometric_view, section_view, title_block.
                            for each region you find, return its label and bounding box.
                            bounding box format : [x0,y0,x1,y1]
                            coordinates must be normalised from 0 to 1000
                            return in JSON format only""",
            "images": [image_path]
        }
    ],
    format = "json",
    
    options = {
        "num_ctx" : 8192, # image + text prompt + model's output must all fit within 8192 tokens combined. i consume more vram when i increase this number
        "temperature" : 0
        # "num_predict" : 4000 #this limit is the output token limiter. It limits the maximum token use for generating output
    }
)

#since its a nested dictionary the hierarchy is like this:
# response:
#     message:
#         role
#         content
#we want to access the content only

print(repr(response["message"]["content"]))
# current output: 
# '{\n    "flat_pattern": [100, 100, 900, 700],\n    "orthographic_view": [100, 100, 900, 700],\n    "isometric_view": [],\n    "section_view": [],\n    "title_block": [100, 700, 900, 900]\n}'

