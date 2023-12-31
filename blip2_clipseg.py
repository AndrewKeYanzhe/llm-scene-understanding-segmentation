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

def swap_first_two_words(input_string):
    words = input_string.split()
    
    if len(words) >= 2:
        words[0], words[1] = words[1], words[0]
        return ' '.join(words)
    else:
        return input_string




t0 = time.time()
from transformers import BlipProcessor, Blip2ForConditionalGeneration, CLIPSegProcessor, CLIPSegForImageSegmentation
# from transformers import BlipProcessor, Blip2Processor, Blip2ForConditionalGeneration, CLIPSegProcessor, CLIPSegForImageSegmentation
print("import time "+str(time.time()-t0))


file_path = r"C:\Users\kyanzhe\Downloads\prompt-to-mask-main\image_path.txt" # Use raw string for file path
with open(file_path, 'r') as file:
    img_path = file.read().rstrip().replace('"', '')
print(img_path)

image = Image.open(img_path)



black_img = Image.open(r"C:\Users\kyanzhe\Downloads\prompt-to-mask-main\black.png")

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
print("model load time "+str(time.time()-t0))



# model.to(device)


print("\n\n\n")

prompt = "Question: " +user_input +"? Answer:"
##inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)



generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

    
while True:
    print("\n")

    user_input = input("Enter question:\n")

    file_path = r"C:\Users\kyanzhe\Downloads\prompt-to-mask-main\image_path.txt" # Use raw string for file path    
    with open(file_path, 'r') as file:
        img_path = file.read().rstrip().replace('"', '')
    print(img_path)
    image = Image.open(img_path).convert('RGB') #doesnt work with .avif

    #get search object-------------------
    t0 = time.time()
    prompt = """
    Instruction: search this floor for people
    Answer: people
    Instruction: Go ahead until the next junction
    Answer: a junction
    Instruction: find the man in the blue shirt
    Answer: a man in a blue shirt
    Instruction: go to the red car in the car park
    Answer: a red car
    Instruction: """ + user_input + " Answer: "

    print(prompt)
    
    

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float32)

    generated_ids = model.generate(**inputs)
    search_object = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    

    print(search_object)
    print("extract search object "+ str(time.time()-t0))
    #finish getting search object-----------------------


    #generate verification question-------------------
    t0 = time.time()

    prompt = """
    Question: find the man in the blue shirt
    Answer: there is a man in a blue shirt
    Question: search this floor for people
    Answer: there are people
    Question: Go ahead until the next junction
    Answer: there is a traffic junction
    Question: go to the red car in the car park
    Answer: there is a red car
    Question: """ + user_input + " Answer: there "

    print(prompt)
    
    

    # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    inputs = processor(images=black_img, text=prompt, return_tensors="pt").to(device, torch.float16)


    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    

    print(generated_text)
    print("generate verification question "+ str(time.time()-t0))
    #finish generating verification question------------------------



    
    prompt = swap_first_two_words("there " + generated_text)
    prompt = "In this image, " + prompt +"? Answer: "
    





    # #----------------rephrase objects to find
    # pattern = re.compile(r'people', re.IGNORECASE)
    # generated_text = pattern.sub('a human', generated_text)

    # pattern = re.compile(r'next', re.IGNORECASE)
    # generated_text = pattern.sub('', generated_text)

    # # # generated_text = " "+generated_text
    # # pattern = r'\bthe\b(?! (?:left|right)\b)'
    # # generated_text = re.sub(pattern, 'a', generated_text)

    # search_object = generated_text

    # ##--------------integrate object into qn
    # if re.search(r'\bin\b', search_object, flags=re.IGNORECASE):
    #     prompt = 'Only answer yes if the entire sentence is correct. Sentence: "In this image there is '+search_object+'" Answer: '
    # else:
    #     # if search_object.startswith("a "):
    #     #     search_object = search_object[2:]
    #     clarify_object = re.sub(r'\bman\b', 'male', search_object)
    #     prompt = 'In this image, is there "' + clarify_object +    '"? Answer:'

    # print(prompt)
    # ##--------------finished making new prompt



    #answer verification question---------------------------
    t0 = time.time()

    print("\n")

    # prompt= "Question: In this image, " + generated_text[0].lower() + generated_text[1:] + " Answer: "
    # prompt = generated_text

    print(prompt)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
    # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float32)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    

    print(generated_text)
    print("answer verification question "+ str(time.time()-t0))
    #finished answering---------------------------

    print("\n")


    #clipseg--------------------------------------------
    if re.search(r'\byes\b', generated_text, flags=re.IGNORECASE):
        pattern = re.compile(r"\b(on the (left|right))\b", re.IGNORECASE)

        # Use the `sub` method of the pattern object to replace any matches with an empty string
        search_object = pattern.sub("", search_object)

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

        max_coordinates_pixel = np.unravel_index(np.argmax(interpolated_array), interpolated_array.shape)
        max_coordinates_percent = [max_coordinates_pixel[0]/interpolated_array.shape[0],max_coordinates_pixel[1]/interpolated_array.shape[1]]
        print(max_coordinates_percent)
        print("vertical, horizontal")

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

        y, x = max_coordinates_percent
        height, width = interpolated_array.shape
        max_coordinates = (int(y * height), int(x * width))

        # Plot the mask_img
        # plt.imshow(interpolated_array, cmap='gray')

        # Plot the point with a marker (e.g., red circle)
        plt.plot(max_coordinates[1], max_coordinates[0], 'ro')


        
        print("clipseg inference time "+str(time.time()-t0)) #1.4s on cpu, 2.3s on gpu

        # plt.imshow(mask_img)
        plt.rcParams['keymap.quit'].append(' ') #default is q. now you can close with spacebar

        plt.axis('off')

        # Save the plot without axes
        plt.savefig('annotated.png', bbox_inches='tight', pad_inches=0)

        # Show the plot (optional)
        plt.axis('on')

        plt.show()