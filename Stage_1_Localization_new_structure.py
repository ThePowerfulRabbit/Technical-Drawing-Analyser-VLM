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
            "content" :"""You are an expert engineering drawing parser.
                        Analyze this technical drawing image and detect ALL distinct visual components on the page.
                        
                        Extract bounding boxes in 0-1000 normalized coordinates [ymin, xmin, ymax, xmax] for every instance of:
                        1. "title_block": The entire metadata table in the bottom-right corner (include revisions, tolerances, company logos).
                        2. "orthographic_view": 2D projected engineering views (front, top, side) including their surrounding dimension lines.
                        3. "isometric_view": 3D projected views of the component.
                        4. "section_view": Sectional view of a component showing an internal cut.
                        5. "flat_pattern": Unfolded sheet metal layout views, if present.
                        
                        Return ONLY a valid JSON object matching this exact schema:
                        {
                          "regions": 
                          [
                            {
                              "label": "title_block",
                              "box": [x0, y0, x1, y1],
                              "conf": 0.95
                            }
                          ]
                        }""",
                        
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

print(repr(response["message"]["content"])) #repr() is used to display any empty strings as output

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

for i in regions:
  label = i["label"] # take the value of label of one particular dictionary in the list regions and store it in a separate variable
  box = i["box"]
  
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