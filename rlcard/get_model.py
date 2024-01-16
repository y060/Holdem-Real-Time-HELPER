import torch
import numpy as np
import json


saved_agent = torch.load('./rlcard/model_agent_100W.pth', map_location=torch.device('cpu'))

extracted_model = saved_agent.q_estimator.qnet

with open('./rlcard/card2index.json', 'r') as file:
    card_to_index = json.load(file)

state_vector = np.zeros(54)

input_cards = ['SA', 'DA'] 
my_chips = 1
max_chips = 2

for card in input_cards:
    index = card_to_index[card]
    state_vector[index] = 1

state_vector[52] = my_chips
state_vector[53] = max_chips

print("Input Vector:\n", state_vector)

state_tensor = torch.from_numpy(state_vector).float()
state_tensor = state_tensor.unsqueeze(0)
output = extracted_model(state_tensor)

print(output)
