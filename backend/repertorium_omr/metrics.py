import torch
import numpy as np

# -------------------------------------------- CTC DECODERS:

def ctc_greedy_decoder(y_pred, i2w):
    # Best path
    y_pred_decoded = torch.argmax(y_pred, dim=1)

    changes = torch.cat([
        torch.tensor([True], device=y_pred_decoded.device),
        y_pred_decoded[1:] != y_pred_decoded[:-1]
    ])
    y_pred_decoded = y_pred_decoded[changes].tolist()
    start_times = (torch.where(changes)[0] / y_pred.shape[0]).tolist()
    start_times_char = np.array([])

    start_times = [start_times[idx] for idx,i in enumerate(y_pred_decoded) if i != len(i2w)]
    y_pred_decoded = [i2w[i] for i in y_pred_decoded if i != len(i2w)]
    for idx,w in enumerate(y_pred_decoded):
        start_times_char = np.concatenate([start_times_char,
                                           np.full(len(w), start_times[idx])])
    y_pred_decoded = ''.join(y_pred_decoded)

    return y_pred_decoded, start_times_char

    
    # y_pred_decoded = torch.unique_consecutive(y_pred_decoded, dim=0).tolist()
    # # Convert to string; len(i2w) -> CTC-blank
    # y_pred_decoded = [i2w[i] for i in y_pred_decoded if i != len(i2w)]
    # return y_pred_decoded