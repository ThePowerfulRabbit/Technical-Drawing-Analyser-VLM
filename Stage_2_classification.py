import ollama
import json

# Lets now import all the images in Stage_1_Output_cropped_images folder and store them in a list
import os
folder_path = "Stage_1_Output_cropped_images"

image_files = os.listdir(folder_path)
#So im trying to give the images to the model all at once instead of putting it in a loop.
#Putting it in a loop made the model classify each image separately by considering them separate drawings
#By giving all images at once and modifying prompts it may produce desired classification result.

# Storing path of every image in the required folder in a list
image_paths = []
for i in range(0,len(image_files)):
    # Joining the path because image_files contain name of the images only while ollama needs complete image path as input
    image_path = os.path.join(folder_path, image_files[i])
    image_paths.append(image_path)
# print(image_paths)

# lets read the prompt text from a separate file as its getting a bit cluttered here:
file = open("Prompts/Classification.txt", "r")
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
        "temperature" : 0 # Temperature controls randomization, if emperature =0 it will give straightforward answers everytime and wont show too much creativity which is what we want here
    }
) # here respose is a python dictionary inside which the output of the model will be saved

# since its a nested dictionary the hierarchy is like this:
# response:
#     message:
#         role
#         content
#we want to access the content only

print(response["message"]["content"])
#lets parse the string into json

json_data = json.loads(response["message"]["content"])
drawing_class = json_data["class"]
evidence = json_data["evidence"]
print("Final Classification:", drawing_class)
print("Evidence/Explanation:", evidence)
# Lets verify output for all 4 input drawings:

# 001_redacted: This is a Sheet Metal 
# Model Output:
# {
#     "class": "Sheet",
#     "confidence": 1.0,
#     "evidence": "The drawing explicitly labels 'Abwicklung' (unfolded view) and 'Blechdicke: 3mm' (sheet thickness), which are definitive cues for sheet-metal fabrication. The component is shown as a flat pattern with holes and rounded corners, consistent with sheet metal, not a hollow tube."
# }
# Final Classification: Sheet

# 002_redacted: This is a Sheet Metal 
# Model Output:
# {
#     "class": "Sheet",
#     "confidence": 1.0,
#     "evidence": "The drawing's title block explicitly labels the component as 'PLATE'. Additionally, the material is specified as '3.3535 / AlMg3', which is a plate material, and the drawing shows a flat-pattern representation with dimensions consistent with a sheet-metal part."
# }
# Final Classification: Sheet

# 003_redacted: This is a Tube
# Model Output:
# {
#     "class": "Tube",
#     "confidence": 1.0,
#     "evidence": "The component is shown with a consistent hollow square cross-section in multiple views (images 2 and 3), and the title block explicitly labels it as 'SQUARE TUBE 80x80x10'. The long dimension (2790) in image 2 indicates a long profile relative to its cross-section, which is characteristic of a tube. The term 'TUBE' is directly visible in the drawing's title block."
# }
# Final Classification: Tube


# 004_redacted: This is a Sheet Metal
# Model Output:
# {
#     "class": "Sheet",
#     "confidence": 0.95,
#     "evidence": "The drawing includes the term 'ADAPTER PLATE' in the title block, which directly indicates a sheet-metal component. Additionally, the cross-section view shows a solid, flat profile with chamfered edges (5 X 45°) and a rounded corner (R5), which are typical of sheet metal fabrication rather than tube construction. The overall shape is consistent with a flat plate, not a hollow profile."
# }
# Final Classification: Sheet