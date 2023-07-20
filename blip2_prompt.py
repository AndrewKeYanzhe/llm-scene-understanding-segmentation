
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


img_path = data

image = Image.open(img_path)

user_input = "What is this a picture of"



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

    # generated_ids = model.generate(**inputs) # default for short prompt replies TODO
    generated_ids = model.generate(**inputs, num_beams=5, max_new_tokens=300, repetition_penalty=3.0, length_penalty=3, temperature=1) #penalty has to be a float. Decent values for great wall of china flant5xl. replicates blip2 paper

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0].strip() #this does not help
    

    print(generated_text)
    t1 = time.time()
    print(t1-t0)
    
