import os
import torch
import sys

cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cur_dir, "../")))

from models import *
from utils.utils import *
from utils.datasets import *

def init_model(**opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt["model_def"], img_size=opt["img_size"]).to(device)

    if opt["weights_path"].endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt["weights_path"])
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt["weights_path"]))

    model.eval()  # Set in evaluation mode
    return model
    