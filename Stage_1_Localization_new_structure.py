import ollama
import json
import pymupdf
import cv2


# Adding a small code snippet here to clear the Stage_1_Output_cropped_images folder so that its empty on each new run
import os
import shutil

output_folder = "Stage_1_Output_cropped_images"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Importing the pdf: (Refer to the pymupdf cheatsheet inside the cheatsheet folder for references)
input_pdf_path = "Input_pdfs/002_redacted.pdf"
pdf = pymupdf.open(input_pdf_path)

page = pdf[0] #first page

pix = page.get_pixmap(dpi = 200) #increasing pixel density with dpi = 200 to give the model more detailed image
pix.save("images/page_0.png")
pdf.close()

image_path = "images/page_0.png"

#lets read the prompt text from a separate file as its getting a bit cluttered here:
file = open("Prompts/localization.txt", "r")
prompt = file.read()
file.close()

print("Sending image to model...")
response = ollama.chat(
    model = "qwen3-vl:8b-instruct", #changing the model to tne instruct varient because the normal one just kept thinking and did nothing other than thinking
    messages = [
        {
            "role" : "user",
            "content" :prompt,
            "images": [image_path]  
        }
    ],
    format = "json",
    
    options = {
        "num_ctx" : 8192, # image + text prompt + model's output must all fit within 8192 tokens combined. it consumes more vram when i increase this number
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

# print(repr(response["message"]["content"])) #repr() is used to display any empty strings as output

# Output is: 
# {
#   "regions": 
#    [
#     {
#       "label": "title_block",
#       "box": [94, 774, 951, 974],
#       "conf": 0.95
#     },
#     {
#       "label": "orthographic_view",
#       "box": [174, 170, 524, 670],
#       "conf": 0.95
#     },
#     {
#       "label": "orthographic_view",
#       "box": [700, 170, 780, 670],
#       "conf": 0.95
#     }
#   ]
# }

# I have entirely changed the output structure right now.
# Rather than storing different views in their separate lists,
# we are storing every view in one list called "regions" where we assign labels for each block
# The output is a Json which has one key called "regions" whose value is a list
# And the list contains dictionaries for each detected region with 3 key value pairs. the keys are label, box and conf

# Lets parse the JSON now to store the output in new variables
# Refer to the Json cheatsheet attached inside the Cheatsheets folder for reference
json_string = response["message"]["content"]
json_data = json.loads(json_string)

# print(json_data)
# Output:
# {'regions': [{'label': 'title_block', 'box': [93, 774, 951, 974], 'conf': 0.95}, {'label': 'orthographic_view', 'box': [120, 168, 520, 670], 'conf': 0.95}, {'label': 'orthographic_view', 'box': [700, 175, 780, 665], 'conf': 0.95}]}

regions = json_data["regions"]

#lets make separate lists for each category so that previously made functions for cropping and saving can be reused here
flat_pattern_normalised_box = []
orthographic_view_normalised_box = []
isometric_view_normalised_box = []
section_view_normalised_box = []
title_block_normalised_box = []

# let's write a loop to cycle through all the output dictionries and store bounding boxes of each category to its own dedicated list
for region in regions:
  label = region["label"] # take the value of label of one particular dictionary in the list regions and store it in a separate variable
  box = region["box"]
  
  if label == "orthographic_view":
    orthographic_view_normalised_box.append(box)
    
  elif label == "isometric_view":
    isometric_view_normalised_box.append(box)
    
  elif label == "section_view":
    section_view_normalised_box.append(box)
    
  elif label == "flat_pattern":
    flat_pattern_normalised_box.append(box)
    
  elif label == "title_block":
    title_block_normalised_box.append(box)

print("Parsed Output:")
print("Flat Pattern:", flat_pattern_normalised_box)
print("Orthographic View:", orthographic_view_normalised_box )
print("Isometric View:", isometric_view_normalised_box)
print("Section View", section_view_normalised_box)
print("Title Block", title_block_normalised_box)

image = cv2.imread(image_path) # read the image converted from pdf input to get image width and image height
image_height, image_width, channels = image.shape  #image.shape returns [height, width, channels]

# Now lets make a function for changing the normalized bounding box coordinates to pixel values
def normalized_to_pixel(image_height, image_width, bounding_boxes):
    # since bounding_boxes is a collection of list each with 4 coordinates in form [x0,y0,x1,y1]
    boxes = []
    
    for i in range(0, len(bounding_boxes)):
        bounding_box = bounding_boxes[i]
        x0,y0,x1,y1 = bounding_box
        
        #lets scale them from 0-1000 to 0-image_width and 0-image_height
        x0 = (x0/1000) * image_width
        y0 = (y0/1000) * image_height
        x1 = (x1/1000) * image_width
        y1 = (y1/1000) * image_height
        boxes.append([int(x0),int(y0),int(x1),int(y1)])  #typecasting to int because pixel coordinates must be integers
    return boxes


# lets scale the coordinates and crop the image right at these coordinates
# opencv can directly crop and save the images with this function: crop = image[y0:y1, x0:x1]

# first lets un-normailise the bounding boxes into pixel coordinates:
flat_pattern_pixel_bounding_box = normalized_to_pixel(image_height, image_width, flat_pattern_normalised_box)
orthographic_view_pixel_bounding_box = normalized_to_pixel(image_height, image_width, orthographic_view_normalised_box)
isometric_view_pixel_bounding_box = normalized_to_pixel(image_height, image_width, isometric_view_normalised_box)
section_view_pixel_bounding_box = normalized_to_pixel(image_height, image_width, section_view_normalised_box)
title_block_pixel_bounding_box = normalized_to_pixel(image_height, image_width, title_block_normalised_box)

# # now lets crop the images and save them in the output folder
# x0, y0, x1, y1 = title_block_pixel_bounding_box[0] #since there is only one title block, we can directly access the first element of the list
# crop = image[y0:y1, x0:x1]

# cv2.imwrite("Stage_1_Output_cropped_images/title_block_cropped.png", crop) #saving the cropped image in the output folder

# now lets make a generic crop function that can work for lists with multiple bounding boxes and also for the ones with empty lists

def crop_and_save_image(bounding_boxes, image_name):
    for i in range (len(bounding_boxes)):
        bounding_box = bounding_boxes[i]
        x0,y0,x1,y1 = bounding_box
        
        crop = image [y0:y1, x0:x1]
        cv2.imwrite(f"Stage_1_Output_cropped_images/{image_name}_{i+1}.png", crop)


crop_and_save_image(flat_pattern_pixel_bounding_box, "flat_pattern_pixel_bounding_box")
crop_and_save_image(orthographic_view_pixel_bounding_box, "orthographic_view_pixel_bounding_box")
crop_and_save_image(isometric_view_pixel_bounding_box, "isometric_view_pixel_bounding_box")
crop_and_save_image(section_view_pixel_bounding_box, "section_view_pixel_bounding_box") 
crop_and_save_image(title_block_pixel_bounding_box, "title_block_pixel_bounding_box")