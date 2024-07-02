import argparse
import os
import json

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from jak.common import seed_all

class SimpleMLP(nn.Module):
        
    def __init__(self, in_dim, out_dim, h_dim=128, depth=1, dropout=0.2, device="cpu"):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.h_dim = h_dim
        self.depth = depth
        self.my_device = device

        self.dropout = dropout

        self.model = nn.Sequential(nn.Linear(self.in_dim, self.h_dim), nn.ReLU())

        for ii in range(depth):

            self.model.add_module(f"hid_{ii}",\
                    nn.Sequential(nn.Dropout(p=self.dropout),\
                        nn.Linear(self.h_dim, self.h_dim), nn.ReLU()))

        self.model.add_module("output_layer", \
                nn.Sequential(nn.Linear(self.h_dim, self.out_dim)))

    def forward(self, x):

        x = torch.tensor(np.array(x)) if type(x) is not torch.Tensor else x

        return self.model(x)
                            

class MLPCohort(nn.Module):

    def __init__(self, in_dim, out_dim,  cohort_size=3, h_dim=128, depth=3, dropout=0.2,\
            lr=3e-4, tag="no_tag", device="cpu"):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.h_dim = h_dim
        self.depth = depth
        self.cohort_size = cohort_size
        self.lr = lr
        self.my_device = device

        self.dropout = dropout
        self.tag = tag

        self.models = []
        for ii in range(self.cohort_size):
            self.add_model()

        self.add_optimizer()

        self.to(self.my_device)


    def add_model(self):

        new_model = SimpleMLP(self.in_dim, self.out_dim, self.h_dim, self.depth, \
                self.dropout, self.my_device) 
        
        self.models.append(new_model)

        for ii, param in enumerate(self.models[-1].parameters()):

            self.register_parameter(f"model{len(self.models)}_param{ii}", param)

        if len(self.models) > self.cohort_size:
            self.cohort_size += 1

    def add_optimizer(self):

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        """
        during training returns the prediction from a single (randomly selected) model, 
        during inference returns the mean and std. dev. of the model ensemble
        """
        if self.training:
            model_index = torch.randint(self.cohort_size, (1,)).item()
            return self.models[model_index](x)

        else:
            with torch.no_grad():
                y = torch.Tensor().to(self.my_device)
                for ii in range(self.cohort_size):

                    y = torch.cat([y, self.models[ii](x).unsqueeze(0)])

                y_mean = torch.mean(y, axis=0) #, keepdim=True)
                y_std_dev = torch.std(y, axis=0) #, keepdim=True)

            return y_mean, y_std_dev

    def forward_ensemble(self, x):
        with torch.no_grad():
            y = torch.Tensor()
            for ii in range(self.cohort_size):

                y = torch.cat([y, self.models[ii](x).unsqueeze(0)])

        return y

    def sparse_loss(self, pred_y, target_y, dont_care=None):

        if dont_care is None:

            dont_care = 1.0 * (target_y == 0)
            do_care = 1.0 - dont_care

        # sqrt of mean squared errer => mean absolute error. 
        loss = torch.sqrt(F.mse_loss(pred_y * do_care, target_y * do_care))

        return loss

    def training_step(self, batch_x, batch_y): 

        self.train()

        pred_y = self.forward(batch_x)

        loss = self.sparse_loss(pred_y, batch_y)
        
        return loss
    
    def validation(self, validation_dataloader):

        self.eval()

        sum_loss = 0.0
        sum_std_dev = 0.0

        with torch.no_grad():
            for ii, batch in enumerate(validation_dataloader):
                pred_y, std_dev = self.forward(batch[0])

                sum_loss += self.sparse_loss(pred_y, batch[1])

                sum_std_dev += torch.mean(std_dev)

            mean_std_dev = sum_std_dev / ii
            mean_loss = sum_loss / ii

        return mean_loss, mean_std_dev

    def fit(self, dataloader, max_epochs, validation_dataloader=None, verbose=True):

        display_every = 1
        save_every = max(1,max_epochs // 8)

        smooth_loss = None
        # exponential averaging coefficient
        alpha = 1.0 - 1e-3 #1.0 / (len(dataloader))
        t0 = time.time()

        print("epoch, wall_time, smooth_loss, train_loss, train_std_dev, val_loss, val_std_dev")

        for epoch in range(max_epochs):
            t1 = time.time()
            sum_loss = 0.0
            for batch_number, batch in enumerate(dataloader):

                self.optimizer.zero_grad()

                loss = self.training_step(batch[0], batch[1])
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
                msg = f"{epoch}, {elapsed_total:.5e}, {smooth_loss:.5e}, {mean_loss:.5e}, "
                

                if validation_dataloader is not None:
                    validation_loss, val_std_dev = self.validation(validation_dataloader)
                    train_loss, train_std_dev = self.validation(dataloader)
                    msg += f"{train_std_dev:.5e}, {validation_loss:.5e}, {val_std_dev:.5e}"
                else:
                    validation_loss, val_std_dev = None, None
                    msg += f" , , , "

                print(msg)


def mlp_train(**kwargs):

    batch_size = kwargs["batch_size"]
    cohort_size = kwargs["cohort_size"]
    depth = kwargs["depth"]
    lr = kwargs["lr"]
    max_epochs = kwargs["max_epochs"]
    my_seed = kwargs["seed"]
    my_device = kwargs["device"]

    tag = f"{kwargs['tag']}_seed{my_seed}"

    if kwargs["x_data"].endswith("pt"):
        x_data = torch.load(kwargs["x_data"])
        y_data = torch.load(kwargs["y_data"])
    else:
        x_data = np.load(kwargs["x_data"])
        y_data = np.load(kwargs["y_data"])

    in_dim = x_data.shape[-1]
    out_dim = y_data.shape[-1]

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

    seed_all(my_seed)
    np.random.shuffle(y_data)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(my_device)
    train_y = y_data[:-validation_size]
    val_y = y_data[-validation_size:]

    # set up dataloaders

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    ensemble = MLPCohort(in_dim, out_dim, \
            cohort_size=cohort_size,\
            depth=depth,\
            tag=tag,\
            lr=lr, device=my_device)

    ensemble.fit(train_dataloader, max_epochs, validation_dataloader=val_dataloader)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int, default=64,\
            help="batch size")
    parser.add_argument("-c", "--cohort_size", type=int, default=10,\
            help="number of mlps in ensemble/cohort")
    parser.add_argument("-d", "--device", type=str, default="cuda",\
            help="cpu or cuda")
    parser.add_argument("-p", "--depth", type=int, default=2,\
            help="depth of mlps in ensemble")
    parser.add_argument("-l", "--lr", type=float, default=3e-5,\
            help="learning rate")
    parser.add_argument("-m", "--max_epochs", type=int, default=100,\
            help="number of epochs to train")
    parser.add_argument("-s", "--seed", type=int, default=42,\
            help="seed for pseudorandom number generators")
    parser.add_argument("-t", "--tag", type=str, default="default_tag",\
            help="string tag used to identify training run")
    parser.add_argument("-x", "--x_data", type=str, default="data/kinase_x.pt",\
            help="relative filepath for training input data")
    parser.add_argument("-y", "--y_data", type=str, default="data/kinase_y.pt",\
            help="relative filepath for training target data")

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    mlp_train(**kwargs)
