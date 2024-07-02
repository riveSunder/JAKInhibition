import argparse
import os
import json

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from jak.common import seed_all
from jak.data import make_token_dict, \
        sequence_to_vectors, \
        one_hot_to_sequence, \
        tokens_to_one_hot

class XFormer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.batch_first = kwargs.get("batch_first", True) 

        encoder_layer = nn.TransformerEncoderLayer(d_model = 36, nhead=12, \
                dim_feedforward=512, batch_first=self.batch_first)
        decoder_layer = nn.TransformerDecoderLayer(d_model = 36, nhead=12, \
                dim_feedforward=512, batch_first=self.batch_first)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.my_device = kwargs.get("device", "cpu")
        self.mask_rate = kwargs.get("mask_rate", 0.2)
        self.lr = kwargs.get("lr", 3e-4)
        self.tag = kwargs.get("tag", "default_tag")

        # default vocab from Janus Kinase dataset
        self.vocab = kwargs.get("vocab", "#()+-1234567=BCFHINOPS[]cilnors")
        self.token_dict = make_token_dict(self.vocab)

        self.seq_length = kwargs.get("seq_length", 128)
        self.token_dim = kwargs.get("token_dim", 36)

    def add_optimizer(self):

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward_x(self, x):

        encoded = self.encoder(x)
        x = self.decoder(x, encoded)

        return x

    def seq2seq(self, sequence: str) -> str:
        """
        autoencode/decode a string
        """
        # convert string sequence to numerical vector
        tokens = sequence_to_vectors(sequence, self.token_dict, \
                pad_to = self.seq_length)

        one_hot = tokens_to_one_hot(tokens, pad_to = self.seq_length,\
                pad_classes_to = self.token_dim)

        one_hot = one_hot.to(self.my_device)

        if self.batch_first:
            pass
        else:
            one_hot = one_hot.permute(1,0,2)

        encoded = self.encoder(one_hot)

        decoded = self.decoder(one_hot, encoded)

        decoded_sequence = one_hot_to_sequence(decoded, self.token_dict)  

        return decoded_sequence


    def encode(self, sequence: str):

        # convert string sequence to numerical vector
        tokens = sequence_to_vectors(sequence, self.token_dict, \
                pad_to = self.seq_length)

        one_hot = tokens_to_one_hot(tokens, pad_to = self.seq_length,\
                pad_classes_to = self.token_dim)

        one_hot = one_hot.to(self.my_device)
        if self.batch_first:
            pass
        else:
            one_hot = one_hot.permute(1,0,2)

        encoded = self.encoder(one_hot)

        return encoded

    def forward(self, sequence: str):

        tokens = sequence_to_vectors(sequence, self.token_dict, \
                pad_to = self.seq_length)

        one_hot = tokens_to_one_hot(tokens, pad_to = self.seq_length,\
                pad_classes_to = self.token_dim)

        one_hot = one_hot.to(self.my_device)
        decoded = self.forward_x(one_hot)

        output_sequence = one_hot_to_sequence(decoded, self.token_dict)

        return output_sequence

    def calc_loss(self, masked, target): 

        predicted = self.forward_x(masked)

        loss = F.cross_entropy(predicted, target)

        return loss

    def training_step(self, batch): 

        one_hot = batch

        vector_mask = 1.0 * (torch.rand(*one_hot.shape[:-1],1).to(self.my_device) > self.mask_rate)

        masked_tokens = one_hot * vector_mask 

        loss = self.calc_loss(masked_tokens, one_hot)

        return loss
    
    def validation(self, validation_dataloader):

        sum_loss = 0.0



        with torch.no_grad():
            for ii, batch in enumerate(validation_dataloader):
                if self.batch_first:
                    validation_batch = batch[0]
                else:
                    validation_batch = batch[0].permute(1,0,2)

                sum_loss += self.training_step(validation_batch).cpu()

            mean_loss = sum_loss / (1+ii)

        return mean_loss

    def fit(self, dataloader, max_epochs, validation_dataloader=None, verbose=True):

        self.add_optimizer()
        display_every = 1
        save_every = 50 #pochs // 1008

        smooth_loss = None
        # exponential averaging coefficient
        alpha = 1.0 - 1e-3 #1.0 / (len(dataloader))

        t0 = time.time()

        print("epoch, wall_time, smooth_loss, train_loss, val_loss,")

        for epoch in range(max_epochs):
            t1 = time.time()
            sum_loss = 0.0
            for batch_number, batch in enumerate(dataloader):
                
                if self.batch_first:
                    training_batch = batch[0]
                else:
                    training_batch = batch[0].permute(1,0,2)

                self.optimizer.zero_grad()

                loss = self.training_step(training_batch)
                loss.backward()

                self.optimizer.step()

                sum_loss += loss.detach().cpu()

                if smooth_loss is None:
                    smooth_loss = loss.detach().cpu()
                else:
                    smooth_loss = alpha * smooth_loss + (1. - alpha) * loss.detach().cpu()

            mean_loss = sum_loss / batch_number
            t2 = time.time()

            save_filepath = f"parameters/{self.tag}/tag_{self.tag}_epoch{epoch}.pt"

            if epoch % save_every == 0 or epoch == max_epochs-1:

                if os.path.exists(os.path.split(save_filepath)[0]):
                    pass
                else:
                    os.mkdir(os.path.split(save_filepath)[0])

                torch.save(self.state_dict(), save_filepath)

            if epoch % display_every == 0 and verbose:


                elapsed_total = t2 - t0
                train_loss = self.validation(dataloader) 
                msg = f"{epoch}, {elapsed_total:.5e}, {smooth_loss:.5e}, {train_loss:.5e}, "

                if validation_dataloader is not None:
                    self.eval()
                    validation_loss = self.validation(validation_dataloader)
                    self.train()
                    msg += f"{validation_loss:.5e}," 
                else:
                    validation_loss, val_std_dev = None, None
                    msg += f" ,"

                print(msg)

def xformer_train(**kwargs):

    batch_size = kwargs["batch_size"]
    lr = kwargs["lr"]
    max_epochs = kwargs["max_epochs"]
    my_seed = kwargs["seed"]
    my_device = kwargs["device"]

    tag = f"xformer_{kwargs['tag']}_seed{my_seed}"

    x_data = np.load(kwargs["x_data"])

    in_dim = x_data.shape[-1]

    validation_size = int(x_data.shape[0] * 0.1) 

    # save model hyperparameters (all kwargs) to json
    json_filepath = f"parameters/{tag}/exp_tag_{tag}.json"
    if os.path.exists(os.path.split(json_filepath)[0]):
        pass
    else:
        os.mkdir(os.path.split(json_filepath)[0])
    with open(json_filepath, "w") as f:
        json.dump(kwargs, f)

    # seed rngs, and split training and validation dataset
    seed_all(my_seed)
    np.random.shuffle(x_data)

    x_data = torch.tensor(x_data, dtype=torch.float32).to(my_device)
    train_x = x_data[:-validation_size]
    val_x = x_data[-validation_size:]



    # set up dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_x)
    val_dataset = torch.utils.data.TensorDataset(val_x)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    for batch in train_dataloader:
        break

    smiles_vocab = "#()+-1234567=BCFHINOPS[]cilnors"

    xformer = XFormer(lr=lr, tag=tag, device=my_device, vocab=smiles_vocab)
    xformer.to(my_device)
    
    xformer.fit(train_dataloader, max_epochs, validation_dataloader=val_dataloader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int, default=64,\
            help="batch size")
    parser.add_argument("-d", "--device", type=str, default="cuda",\
            help="cpu or cuda")
    parser.add_argument("-l", "--lr", type=float, default=3e-5,\
            help="learning rate")
    parser.add_argument("-m", "--max_epochs", type=int, default=100,\
            help="number of epochs to train")
    parser.add_argument("-s", "--seed", type=int, default=42,\
            help="seed for pseudorandom number generators")
    parser.add_argument("-t", "--tag", type=str, default="default_tag",\
            help="string tag used to identify training run")
    parser.add_argument("-x", "--x_data", type=str, default="data/one_hot_smiles.npy",\
            help="relative filepath for training input data")

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    xformer_train(**kwargs)
