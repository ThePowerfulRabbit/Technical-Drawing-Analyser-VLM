import ollama
import json

# Lets now import all the images in Stage_1_Output_cropped_images folder and store them in a list
import os
# folder_path = "Stage_1_Output_cropped_images"
folder_path = "images/Localization output (just for reference)/003_redacted"

image_files = os.listdir(folder_path)

#lets read the prompt text from a separate file as its getting a bit cluttered here:
file = open("Prompts/Classification.txt", "r")
prompt = file.read()
file.close()

#lets create an empty list to store class of every cropped drawing
classes = []

for i in range(0,len(image_files)):
    
    image_path = os.path.join(folder_path, image_files[i]) # Joining the path because image_files contain name of the images only while ollama needs complete image path as input
    response = ollama.chat(

        model = "qwen3-vl:8b-instruct",
        messages = [
            {
                "role" : "user",
                "content": prompt,
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
    
    print(response["message"]["content"])
    #lets parse the string into json
    
    json_data = json.loads(response["message"]["content"])
    drawing_class = json_data["class"]
    classes.append(drawing_class) # every class will be stored inside the classes list

#now lets take the most repeated class value to display as the final class:

no_sheet = classes.count("sheet")
no_tube = classes.count("Tube")

if (no_sheet > no_tube):
    final_classification = "sheet"
else:
    final_classification = "Tube"

print("Final Classification:", final_classification)