import time
t0 = time.time()
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation #this takes 13s
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



  max_coordinates_pixel = np.unravel_index(np.argmax(interpolated_array), interpolated_array.shape)
  max_coordinates_percent = [max_coordinates_pixel[0]/interpolated_array.shape[0],max_coordinates_pixel[1]/interpolated_array.shape[1]]
  print(max_coordinates_percent)
  print("vertical, horizontal")

  

  plt.imsave(filename, interpolated_array)


  mask_img = mpimg.imread(filename)



  plt.imshow(image)
  plt.imshow(mask_img, cmap='jet', alpha=0.5)

  y, x = max_coordinates_percent
  height, width = interpolated_array.shape
  max_coordinates = (int(y * height), int(x * width))


  # Plot the point with a marker (e.g., red circle)
  plt.plot(max_coordinates[1], max_coordinates[0], 'ro')

  

  print("inference time "+str(time.time()-t0)) #1.3s on cpu, 2.3s on gpu

  plt.rcParams['keymap.quit'].append(' ') #default is q. now you can close with spacebar


  # plt.imshow(mask_img)
  plt.show()

  return True
  

  
process_image( Image.open(r"C:\Users\kyanzhe\Downloads\download (3).jfif").convert('RGB'),"man in blue shirt")
