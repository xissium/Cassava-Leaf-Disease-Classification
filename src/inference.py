# inference.py
import numpy as np
import torch
from tqdm import tqdm


def inference(model, states, test_loader, device):
    model.to(device)
    probs = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, (images) in pbar:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs
