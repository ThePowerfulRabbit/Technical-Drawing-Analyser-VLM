import ollama
import json

# Lets now import all the images in Stage_1_Output_cropped_images folder and store them in a list
import os
folder_path = "Stage_1_Output_cropped_images"
image_files = os.listdir(folder_path) 

# lets import the json content into a separate variable
json_file = open("json/Classification/Classification.json", "r")
classification_data = json.load(json_file)
classification_result = classification_data["class"]

print("Starting Bend Count")

if (classification_result == "Sheet"):
    # Storing path of every image in the required folder in a list. (same thing already done and explained in classification.py)
    image_paths = []
    for i in range(0,len(image_files)):
        # Joining the path because image_files contain name of the images only while ollama needs complete image path as input
        image_path = os.path.join(folder_path, image_files[i])
        image_paths.append(image_path)
    # print(image_paths)
    
    # lets read the prompt text from a separate file as its getting a bit cluttered here:
    file = open("Prompts/Bend_Count.txt", "r")
    prompt = file.read()
    file.close()
    
    response = ollama.chat(
        model = "qwen3-vl:8b-instruct",
        messages = [
            {
                "role" : "user",
                "content": prompt,
                "images" : image_paths
            }
        ],
        
        options = {
            "num_ctx" : 8192, # image + text prompt + model's output must all fit within 8192 tokens combined. it consumes more vram when i increase this number
            "temperature" : 0, # Temperature controls randomization, if emperature =0 it will give straightforward answers everytime and wont show too much creativity which is what we want here
            "num_predict": 200
        }
    ) # here respose is a python dictionary inside which the output of the model will be saved
    
    # since its a nested dictionary the hierarchy is like this:
    # response:
    #     message:
    #         role
    #         content
    #we want to access the content only
    print(response["message"]["content"])

else:
    print("Drawing is not a sheet")


json_data = json.loads(response["message"]["content"])
import shutil

output_folder = "json/Bend_count"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Lets convert the output python list into a json file and export it:
json_file = open("json/Bend_count/Bend_count.json", "w")
json.dump(json_data,json_file, indent=4)
json_file.close()

print("Bend Count Complete")