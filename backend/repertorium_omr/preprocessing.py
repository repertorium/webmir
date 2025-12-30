import joblib
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

memory = joblib.memory.Memory("./cache", mmap_mode="r", verbose=0)

NUM_CHANNELS = 1
IMG_HEIGHT = 64
IMG_WIDTH = 256
toTensor = transforms.ToTensor()

################################# Image preprocessing:
def preprocess_image_from_file(path, unfolding=False, reduce=False):
    x = Image.open(path)
    x = preprocess_image_from_object(x, unfolding=unfolding, reduce=reduce)
    return x


def preprocess_image_from_object(x, unfolding=False, reduce=False):
    x = x.convert("L")  # Convert to grayscale

    if not unfolding:
        # Resize (preserving aspect ratio)
        new_width = int(
            IMG_HEIGHT * x.size[0] / x.size[1]
        )
        x = x.resize((new_width, IMG_HEIGHT))
    
    # If the boolean flag is True, rotate -90 degrees and flip horizontally
    else:
        reduction_factor = 2
        new_width = x.width // reduction_factor
        new_height = x.height // reduction_factor
        
        if reduce:
            reduce_factor = 0.75
            new_width = int(new_width * reduce_factor)
            new_height = int(new_height * reduce_factor)
        
        x = x.resize((new_width, new_height))
        x = x.rotate(-90, expand=True)  # Rotate by -90 degrees
    
    x = toTensor(x)  # Convert to tensor (normalizes to [0, 1])
    return x


def preprocess_image_from_object2(x, unfolding=False, reduce=False, image_height=128):
    x = x.convert("L")  # Convert to grayscale

    if not unfolding:
        # Resize (preserving aspect ratio)
        new_width = int(
            image_height * x.size[0] / x.size[1]
        )
        x = x.resize((new_width, image_height))

    # If the boolean flag is True, rotate -90 degrees and flip horizontally
    else:
        reduction_factor = 2
        new_width = x.width // reduction_factor
        new_height = x.height // reduction_factor
        
        if reduce:
            reduce_factor = 0.75
            new_width = int(new_width * reduce_factor)
            new_height = int(new_height * reduce_factor)
        
        x = x.resize((new_width, new_height))
        x = x.rotate(-90, expand=True)  # Rotate by -90 degrees
    
    x = toTensor(x)  # Convert to tensor (normalizes to [0, 1])
    return x


################################# CTC Preprocessing:

def pad_batch_images(x):
    max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[2])) for i in x], dim=0)
    return x

def ctc_batch_preparation(batch):
    x, xl = zip(*batch)
    # Zero-pad images to maximum batch image width
    x = pad_batch_images(x)
    xl = torch.tensor(xl, dtype=torch.int32)
    return x, xl