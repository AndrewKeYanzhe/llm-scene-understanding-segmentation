import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

## v2 models
model_path = 'openlm-research/open_llama_3b_v2'
# model_path = 'openlm-research/open_llama_7b_v2'

## v1 models
# model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

while True:
    user_input = input("Enter question:\n")
    # prompt = 'Q: What is the largest animal?\nA:'
    # prompt = '''Robot: Hi there, I’m a robot operating in an office kitchen.
    # You can ask me to do various tasks and I’ll tell you the sequence of actions I would do to accomplish your task.
    # The following objects are in the scene: 7up, apple, tea, multigrain chips, kettle chips, jalapeno chips, rice chips, coke, grapefruit soda, pepsi, redbull, energy bar, lime soda, sponge, and water bottle.
    # The following locations are in the scene: close counter, far counter, table, you, trash, bowl.
    # The actions that I can do are: find object, pick up object, go to location, put down object, done. I am unable to do other kinds of actions.
    # Human: Hold the sponge
    # Explanation: The user has asked for a single skill, I will just pick up the sponge.
    # Robot: 1. pick up the sponge, 2. done.
    # Human: Put the kettle chips here
    # Explanation: The user has asked for a type of chips, I will bring those to them.
    # Robot: 1. put down the kettle chips, 2. done.
    # Human: Move the grapefruit drink from the table to the close counter
    # Explanation: The user has asked me to move the grapefruit drink to the counter.
    # Robot: 1. find a grapefruit soda, 2. pick up the grapefruit soda, 3. go to the counter, 4. put down the grapefruit soda, 5. done.
    # Human: Bring me some snacks
    # Explanation: The user has asked for snacks, I will bring jalapeno chips and an apple.
    # Robot: 1. find a jalapeno chips, 2. pick up the jalapeno chips, 3. bring it to you, 4. put down the jalapeno chips, 5. find an apple, 6. pick up the apple, 7. bring it to you, 8. put down the apple, 9. done.
    # Human: Bring me something that isn’t a fruit
    # Explanation: The user has asked for something food that isn’t an fruit, I will bring an energy bar.
    # Robot: 1. find an energy bar, 2. pick up the energy bar, 3. bring it to you, 4. put down the energy bar, 5. done.
    # Human: Put the rice chips in the bowl and then move the tea to the table
    # Explanation: The user has asked me to do two tasks, I will do one and then the other.
    # Robot: 1. find the rice chips, 2. pick up the rice chips, 3. go to the bowl, 4. put down the rice chips, 5. pick up the tea, 6. go to the table, 7. put down the tea, 8. done.
    # Human: Throw away a redbull
    # Explanation: '''
    prompt = user_input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=300 #at this length, the model might ramble. 
    )
    print(tokenizer.decode(generation_output[0]))
