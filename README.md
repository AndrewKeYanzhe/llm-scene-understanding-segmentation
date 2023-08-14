# prompt-to-mask



#### Setup
1. Install dependencies in `requirements.txt`
2. Create `image_path.txt` and write the image path in the file
3. Create a black image (e.g. using Microsoft Paint) and save as `black.png` in the root folder
<br/><br/>

#### Tested on
* NVIDIA TITAN V 12GB (6.7GB VRAM was observed to be used by `blip2_clipseg.py`)
<br/><br/>

#### blip2_clipseg.py
1. Read user input for instruction
2. Verify that the object is in scene
3. Draw heatmap with CLIPSeg and save as `mask.png`
4. Overlay heatmap over source image and save as `annotated.png`

#### blip2_prompt.py
Use your own prompt. Recommended to use `Question: <insert question> Answer:` format

#### clipseg.py
Input keyword, CLIPSeg draws heatmap
