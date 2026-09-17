import ollama
import json
import pymupdf
import cv2

# Importing the pdf: (Refer to the pymupdf cheatsheet inside the cheatsheet folder for references)
input_pdf_path = "Input_pdfs/002_redacted.pdf"
pdf = pymupdf.open(input_pdf_path)

page = pdf[0] #first page

pix = page.get_pixmap(dpi = 200) #increasing pixel density with dpi = 200 to give the model more detailed image
pix.save("images/page_0.png")
pdf.close()

image_path = "images/page_0.png"


print("Sending image to model...")
response = ollama.chat(
    model = "qwen3-vl:8b-instruct", #changing the model to tne instruct varient because the normal one just kept thinking and did nothing other than thinking
    messages = [
        {
            "role" : "user",
            "content" : """Identify the following regions in the given engineering drawing:
            
                            - flat_pattern: an unfolded sheet-metal representation, if present.
                            - orthographic_view: front, top, or side projection views.
                            - isometric_view: a 3D pictorial/isometric view.
                            - section_view: a sectional view showing an internal cut.
                            - title_block: the drawing information block.
                            
                            For each region that is actually present, return a tight bounding box around that region.
                            
                            return a separate bounding box for each individual view.
                            if there are multilple objects with same region types, make a separate bounding box for that region under the same category
                            if any region is missing, dont skip it, return empty value for that region
                            the output should look like: 
                            {
                                "flat_pattern": [],
                                "orthographic_view": [
                                    [something, something, something, something],
                                    [something, something, something, something]
                                ],
                                "isometric_view": [],
                                "section_view": [],
                                "title_block": [
                                    [something, something, something, something]
                                ]
                            }
                            Do not classify an ordinary orthographic view as a flat_pattern.
                            
                            Bounding box format: [x0, y0, x1, y1].
                            Coordinates must be normalized from 0 to 1000.
                            
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

print(repr(response["message"]["content"])) #repr() is used to display any empty strings as output

# now lets parse the json and store the coordinates in new variables
json_data = json.loads(response["message"]["content"])

flat_pattern = json_data["flat_pattern"]
orthographic_view = json_data["orthographic_view"]
isometric_view = json_data["isometric_view"]
section_view = json_data["section_view"]
title_block = json_data["title_block"]

# parsed output:
print("Parsed Output:")
print("Flat Pattern:", flat_pattern)
print("Orthographic View:", orthographic_view )
print("Isometric View:", isometric_view)
print("Section View", section_view)
print("Title Block", title_block)

image = cv2.imread(image_path) # read the image converted from pdf input to get image width and image height
image_height, image_width, channels = image.shape  #image.shape returns [height, width, channels]

# print("Image height:",image_height)
# print("Image width:",image_width)

# Now lets make a function for changing the normalized bounding box coordinates to pixel values

def normalized_to_pixel(image_height, image_width, bounding_box):
    # since bounding_box is a list with 4 coordinates in form [x0,y0,x1,y1]
    x0,y0,x1,y1 = bounding_box
    
    #lets scale them from 0-1000 to 0-image_width and 0-image_height
    x0 = (x0/1000) * image_width
    y0 = (y0/1000) * image_height
    x1 = (x1/1000) * image_width
    y1 = (y1/1000) * image_height
    
    return [x0,y0,x1,y1]

#lets scale the coordinates and crop the image right at these coordinates
# opencv can directly crop and save the images with this function: crop = image[y0:y1, x0:x1]

#first lets normaize the bounding boxes into pixel coordinates:
flat_pattern_pixel_bounding_box = normalized_to_pixel(image_height, image_width, flat_pattern)
orthographic_view_pixel_bounding_box = normalized_to_pixel(image_height, image_width, orthographic_view)
isometric_view_pixel_bounding_box = normalized_to_pixel(image_height, image_width, isometric_view)
section_view_pixel_bounding_box = normalized_to_pixel(image_height, image_width, section_view)
title_block_pixel_bounding_box = normalized_to_pixel(image_height, image_width, title_block)
