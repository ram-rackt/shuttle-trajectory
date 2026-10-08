import torch
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from PIL import Image
import numpy as np

print("Loading processor...")
processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
print("Loading model...")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")

print("Model loaded. Testing inference...")
img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
inputs = processor(images=Image.fromarray(img), return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(720, 1280)]), threshold=0.1
)[0]

print("Person label index:")
for k, v in model.config.id2label.items():
    if v.lower() == 'person':
        print(f"{k}: {v}")

print("Inference successful. Output keys:", outputs.keys())
