# -*- coding: utf-8 -*-
"""blip2_instructed_generation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb

#### Large RAM is required to load the larger models. Running on GPU can optimize inference speed.
"""
import time
import sys
if 'google.colab' in sys.modules:
    print('Running in Colab.')
    #pip3 install salesforce-lavis

import torch
from PIL import Image
import requests
##from lavis.models import load_model_and_preprocess

from transformers import BlipProcessor, Blip2Processor, Blip2ForConditionalGeneration

file_path = r"C:\Users\kyanzhe\Downloads\blip2\image_path.txt"  # Use raw string for file path
with open(file_path, 'r') as file:
    data = file.read().rstrip().replace('"', '')
print(data)

t0 = time.time()





##time.sleep(99999)


"""#### Load an example image"""




##img_path = r"C:\Users\kyanzhe\Downloads\cars-get-bigger-but-not-parking-spaces_1.jpg"
img_path = data

image = Image.open(img_path)

user_input = "What is this a picture of"
##user_input = input("Enter question:\n")
##print(user_input)


##display(raw_image.resize((596, 437)))

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

"""#### Load pretrained/finetuned BLIP2 captioning model"""

# we associate a model with its preprocessors to make it easier for inference.
##model, vis_processors, _ = load_model_and_preprocess(
##    name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
##)

##url = "blob:https://replicate.com/ccd7299a-0272-467e-accc-2120d48041b2"
##image = Image.open(requests.get(url, stream=True).raw)


##processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
##model = Blip2ForConditionalGeneration.from_pretrained(
####    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16 #this throws error on CPU
##    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32
##)

processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")

t1 = time.time()
print(t1-t0)

# model.to(device)


print("\n\n\n")
t0 = time.time()
prompt = "Question: " +user_input +"? Answer:"
##inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
inputs = processor(image, prompt, return_tensors="pt").to("cuda", torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
t1 = time.time()
print(t1-t0)
    
while True:
    user_input = input("Enter question:\n")

    file_path = r"C:\Users\kyanzhe\Downloads\blip2\image_path.txt"  # Use raw string for file path
    with open(file_path, 'r') as file:
        data = file.read().rstrip().replace('"', '')
    print(data)

    img_path = data

    image = Image.open(img_path).convert('RGB')


    # prompt = "Question: " +user_input +"? Answer:"
    prompt = user_input
    print(prompt)

    t0 = time.time()

    ##inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    inputs = processor(image, prompt, return_tensors="pt").to("cuda", torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    

    print(generated_text)
    t1 = time.time()
    print(t1-t0)
    
