
import time
import sys

import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
from transformers import BlipProcessor, Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig


file_path = r"C:\Users\kyanzhe\Downloads\prompt-to-mask-main\image_path.txt"  # Use raw string for file path
with open(file_path, 'r') as file:
    img_path = file.read().rstrip().replace('"', '')
print(img_path)

t0 = time.time()




image = Image.open(img_path)

user_input = "What is this a picture of"



# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"



# # attempt at t5xxl
# model, vis_processors, _ = load_model_and_preprocess(
#    name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
# ) #lavis seems to not allow int8 https://github.com/salesforce/LAVIS/issues/294. Out of memory


##processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
##model = Blip2ForConditionalGeneration.from_pretrained(
####    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16 #this throws error on CPU
##    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32
##)


# this is working version-------------
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")


# # attempt at t5xxl https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
# processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", load_in_8bit=True,quantization_config=quantization_config, device_map=device) #out of memory on 12GB VRAM


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

    file_path = r"C:\Users\kyanzhe\Downloads\prompt-to-mask-main\image_path.txt"  # Use raw string for file path
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
    
    # default for short prompt replies TODO
    generated_ids = model.generate(**inputs) 



    # #attempt at longer replies, like the blip2 paper
    # generated_ids = model.generate(**inputs, num_beams=5, max_new_tokens=300, repetition_penalty=3.0, length_penalty=3, temperature=1, return_dict_in_generate=True, output_scores=True)  
    # #penalty has to be a float. Decent values for great wall of china flant5xl. replicates blip paper
    # # however penalty of 1.5 and above is not recommended https://discuss.huggingface.co/t/transformers-repetition-penalty-parameter/43638/3
    # #huggingface demo has penalty 0-5 slider
    # #huggingface demo has length penalty -1 to 2

    # #attempt to output confidence
    # generated_ids = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)  
    # transition_scores = model.compute_transition_scores(generated_ids.sequences, generated_ids.scores, normalize_logits=True)
    # transition_proba = np.exp(transition_scores)
    # print(transition_proba)
    # #error, blip2 has no vocab_size attribute

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0].strip() #this does not help
    

    print(generated_text)
    t1 = time.time()
    print(t1-t0)
    
