import time
t0 = time.time()
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation #this takes 13s
t1 = time.time()
print("import time "+str(time.time()-t0))
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2



from matplotlib import image as mpimg
import numpy as np

from scipy.ndimage import zoom

t0 = time.time()
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
t1 = time.time()
print("model load time "+str(time.time()-t0))
def process_image(image, prompt):
  t0 = time.time()

  print(image.size)
  w,h=image.size
  inputs = processor(text=prompt, images=image, padding="max_length", return_tensors="pt")
  
  # predict
  with torch.no_grad():
    outputs = model(**inputs)
    preds = outputs.logits
  
  filename = r"C:\Users\kyanzhe\Downloads\prompt-to-mask-main\mask.png"
  mask_array = torch.sigmoid(preds)

  original_h, original_w = mask_array.shape

  zoom_factors = (h / original_h, w / original_w)

  # Interpolate the array
  interpolated_array = zoom(mask_array, zoom_factors)

  # print(interpolated_array)

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

  

  print("inference time "+str(time.time()-t0)) #1.3s on cpu, 2.3s on gpu

  plt.rcParams['keymap.quit'].append(' ') #default is q. now you can close with spacebar


  # plt.imshow(mask_img)
  plt.show()
  
  # # img2 = cv2.imread(filename)
  # # gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  # # (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
  
  # # # fix color format
  # # cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
  
  # # return Image.fromarray(bw_image)

  # return Image.open("mask.png").convert("RGB")
  return True
  
title = "Interactive demo: zero-shot image segmentation with CLIPSeg"
description = "Demo for using CLIPSeg, a CLIP-based model for zero- and one-shot image segmentation. To use it, simply upload an image and add a text to mask (identify in the image), or use one of the examples below and click 'submit'. Results will show up in a few seconds."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2112.10003'>CLIPSeg: Image Segmentation Using Text and Image Prompts</a> | <a href='https://huggingface.co/docs/transformers/main/en/model_doc/clipseg'>HuggingFace docs</a></p>"

examples = [[r"C:\Users\kyanzhe\Downloads\download (3).jfif", "wood"]]
   
# interface = gr.Interface(fn=process_image, 
#                      inputs=[gr.Image(type="pil"), gr.Textbox(label="Please describe what you want to identify")],
#                      outputs=gr.Image(type="pil"),
#                      title=title,
#                      description=description,
#                      article=article,
#                      examples=examples)
                     
# interface.launch(debug=True)

process_image( Image.open(r"C:\Users\kyanzhe\Downloads\download (3).jfif").convert('RGB'),"man in blue shirt")
# process_image( Image.open(r"C:\Users\kyanzhe\Downloads\t688.jpg").convert('RGB'),"car on the left")