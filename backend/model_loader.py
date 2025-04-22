import torch
from gwnet_model import gwnet

def load_model(device, supports, aptinit, day):
    num_nodes = 35
    model = gwnet(
        device=device,
        num_nodes=num_nodes,
        supports=supports,
        aptinit=aptinit,
        in_dim=12,
        out_dim=12,
        pred_len=day  # Or your desired output length
    )
    # model.load_state_dict(torch.load("models/WaveNet_13_3.pt", map_location=device))
    # model.eval()
    return model
