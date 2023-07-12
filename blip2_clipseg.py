# -*- coding: utf-8 -*-
"""blip2_instructed_generation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb

#### Large RAM is required to load the larger models. Running on GPU can optimize inference speed.
"""

import re
import time
import matplotlib.pyplot as plt
import cv2
from matplotlib import image as mpimg
import numpy as np
from scipy.ndimage import zoom

import sys
if 'google.colab' in sys.modules:
    print('Running in Colab.')
    #pip3 install salesforce-lavis

import torch
from PIL import Image
import requests
##from lavis.models import load_model_and_preprocess


t0 = time.time()
from transformers import BlipProcessor, Blip2ForConditionalGeneration, CLIPSegProcessor, CLIPSegForImageSegmentation
# from transformers import BlipProcessor, Blip2Processor, Blip2ForConditionalGeneration, CLIPSegProcessor, CLIPSegForImageSegmentation
t1 = time.time()
print("import time "+str(time.time()-t0))


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

t0 = time.time()
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")


clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
t1 = time.time()
print("model load time "+str(time.time()-t0))



# model.to(device)

t1 = time.time()
print(t1-t0)

print("\n\n\n")
t0 = time.time()
prompt = "Question: " +user_input +"? Answer:"
##inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)



generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
t1 = time.time()
print(t1-t0)
    
while True:
    print("\n")
    user_input = input("Enter question:\n")

    file_path = r"C:\Users\kyanzhe\Downloads\blip2\image_path.txt"  # Use raw string for file path
    with open(file_path, 'r') as file:
        data = file.read().rstrip().replace('"', '')
    print(data)

    img_path = data

    image = Image.open(img_path).convert('RGB')


##    prompt = "Question: " +user_input +"? Answer:"
##    prompt = user_input
    prompt = 'Imagine that the image is blank. In the sentence "'+user_input+'", what are we looking for?'
    print(prompt)
    
    t0 = time.time()

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float32)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    

    print(generated_text)
    print("prompt processing time "+ str(time.time()-t0))


    pattern = re.compile(r'people', re.IGNORECASE)
    generated_text = pattern.sub('a man or woman', generated_text)

    pattern = re.compile(r'next', re.IGNORECASE)
    generated_text = pattern.sub('', generated_text)

    generated_text = " "+generated_text
    pattern = re.compile(r' the ', re.IGNORECASE)
    generated_text = pattern.sub(' a ', generated_text).strip()

    search_object = generated_text

    ##--------------

    prompt = 'Only answer yes if the entire sentence is correct. Sentence: "In this image there is '+search_object+'" Answer: '
    print(prompt)
    t0 = time.time()

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
    # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float32)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    

    print(generated_text)
    print("blip2 inference time "+ str(time.time()-t0))

    #clipseg--------------------------------------------
    if re.search(r'\byes\b', generated_text, flags=re.IGNORECASE):
        t0 = time.time()
        print(image.size)
        w,h=image.size

        clipseg_inputs = clipseg_processor(text=search_object, images=image, padding="max_length", return_tensors="pt")
        
        with torch.no_grad(): #grad probably means gradient, is a tensor thing?
            outputs = clipseg_model(**clipseg_inputs)
            preds = outputs.logits

        filename = r"C:\Users\kyanzhe\Downloads\prompt-to-mask-main\mask.png"
        mask_array = torch.sigmoid(preds)

        original_h, original_w = mask_array.shape

        zoom_factors = (h / original_h, w / original_w)

        # Interpolate the array
        interpolated_array = zoom(mask_array, zoom_factors)


        # plt.imsave(filename, torch.sigmoid(preds))
        plt.imsave(filename, interpolated_array)



        # cv2.imshow("mask", torch.sigmoid(preds))
        # mask = torch.sigmoid(preds)
        # mask.show()
        mask_img = mpimg.imread(filename)



        plt.imshow(image)
        # plt.imshow(image, cmap='jet', alpha=0.5, aspect=h/w)
        plt.imshow(mask_img, cmap='jet', alpha=0.5)
        # plt.imshow(mask_img)
        # plt.gca().set_aspect(h/w)

        
        print("clipseg inference time "+str(time.time()-t0)) #1.3s on cpu, 2.3s on gpu

        # plt.imshow(mask_img)
        plt.show()