#Lets make a pipeline code to run each stage serially
import subprocess

#running each python file step by step
subprocess.run(["python", "Stage_1_Localization.py"])
subprocess.run(["python", "Stage_2_Classification.py"])
subprocess.run(["python", "Stage_3_BendCount.py"])

# Output:
# Starting localization
# Parsed Output:
# Flat Pattern: [[210, 50, 770, 420]]
# Orthographic View: [[87, 467, 300, 767], [300, 467, 670, 767]]
# Isometric View: [[770, 500, 977, 767]]
# Section View []
# Title Block [[537, 777, 977, 977]]
# Localization Complete
# Starting Classification
# {
#     "class": "Sheet",
#     "confidence": 1.0,
#     "evidence": "The drawing explicitly labels 'Abwicklung' (unfolded view) and 'Blechdicke: 3mm' (sheet thickness), which are direct indicators of sheet-metal fabrication. The flat pattern view and the presence of 'Blech' (sheet) in the material description further confirm it is a sheet component."
# }
# Final Classification: Sheet
# Evidence/Explanation: The drawing explicitly labels 'Abwicklung' (unfolded view) and 'Blechdicke: 3mm' (sheet thickness), which are direct indicators of sheet-metal fabrication. The flat pattern view and the presence of 'Blech' (sheet) in the material description further confirm it is a sheet component.
# Classification Complete
# Starting Bend Count
# {
#   "num_bends": 2,
#   "bend_confidence": 1.0,
#   "bend_evidence": "Two bends from U-channel profile in side view.",
#   "flags": []
# }
# Bend Count Complete