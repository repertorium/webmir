import os
import torch
from lightning.pytorch import LightningModule
from torch.nn import CTCLoss
from torchinfo import summary

from repertorium_omr.preprocessing import IMG_HEIGHT, NUM_CHANNELS
from repertorium_omr.metrics import ctc_greedy_decoder
from repertorium_omr.modules import CRNN
from repertorium_omr.modules import E2EScore_CRNN

import sys

class CTCTrainedCRNN(LightningModule):
    def __init__(self, w2i, i2w, ytest_i2w=None, ds_name=None):
        super(CTCTrainedCRNN, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w
        # Model
        self.model = CRNN(output_size=len(self.w2i) + 1)
        self.summary()
        self.compute_ctc_loss = CTCLoss(
            blank=len(self.w2i), zero_infinity=True
        )  # The target index cannot be blank!
        # Predictions
        self.YHat = []
        self.img_paths = []
        
        # To save the predictions in a file
        self.ds_name = ds_name

    def summary(self):
        summary(self.model, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def forward(self, x):
        return self.model(x)

    def transcribe(self, x):
        # Model prediction (decoded using the vocabulary on which it was trained)
        yhat = self.model(x)[0]
        yhat = yhat.log_softmax(dim=-1).detach().cpu()
        yhat, t_hat = ctc_greedy_decoder(yhat, self.i2w)
        return yhat, t_hat

    def test_step(self, batch):
        x, img_path = batch  # batch_size = 1
        # Model prediction (decoded using the vocabulary on which it was trained)
        yhat = self.model(x)[0]
        yhat = yhat.log_softmax(dim=-1).detach().cpu()
        yhat = ctc_greedy_decoder(yhat, self.i2w)
        # Append to later compute metrics
        self.YHat.append(yhat)
        self.img_paths.append(img_path)

    def on_test_epoch_end(self):
        # Save predictions     
        predictions_folder = os.path.join("predictions", self.ds_name)
        os.makedirs(predictions_folder, exist_ok=True)

        for pred, img_path in zip(self.YHat, self.img_paths):
            file_name = os.path.splitext(os.path.basename(img_path[0]))[0]
            pred_file_path = os.path.join(predictions_folder, f"{file_name}.txt")
            with open(pred_file_path, 'w') as f:
                pred_str = ''.join(pred)
                f.write(f"{pred_str}\n")
                        
        # Clear predictions
        self.YHat.clear()
        self.img_paths.clear()
        
class LightningE2EModelUnfolding(LightningModule):
    def __init__(self, w2i, i2w, ytest_i2w=None):
        super(LightningE2EModelUnfolding, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w
        self.model = E2EScore_CRNN(1, len(self.w2i) + 1)
        self.summary()
        # Loss
        self.compute_ctc_loss = CTCLoss(
            blank=len(self.w2i)
        )  # The target index cannot be blank!
        # Predictions
        self.YHat = []
        self.img_paths = []
        
    def summary(self):
        summary(self.model, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])
     
    # Checking if a bigger learning rate helps   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        x, img_path = batch  # batch_size = 1
        # Model prediction (decoded using the vocabulary on which it was trained)
        yhat = self.model(x)
        yhat = yhat.permute(1,0,2).contiguous()
        yhat = yhat[0]
        yhat = yhat.log_softmax(dim=-1).detach().cpu()
        yhat = ctc_greedy_decoder(yhat, self.i2w)    
        self.YHat.append(yhat)
        self.img_paths.append(img_path)
    
    def on_test_epoch_end(self):
        # Save predictions     
        predictions_folder = os.path.join("predictions", "aligned")
        os.makedirs(predictions_folder, exist_ok=True)

        for pred, img_path in zip(self.YHat, self.img_paths):
            file_name = os.path.splitext(os.path.basename(img_path[0]))[0]
            pred_file_path = os.path.join(predictions_folder, f"{file_name}.txt")
            with open(pred_file_path, 'w') as f:
                pred_str = ''.join(pred)
                f.write(f"{pred_str}\n")
                        
        # Clear predictions
        self.YHat.clear()
        self.img_paths.clear()
