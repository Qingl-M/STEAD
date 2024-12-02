import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from torch.nn.init import xavier_normal_, zeros_

from algorithm_utils import Algorithm, PyTorchUtils
from STE_GRU import GATGRUCell

class GraphGRU_VAE(nn.Module, PyTorchUtils):
    def __init__(self, nodes_num: int, input_dim: int, hidden_dim: int,
                 num_layers: tuple=(2,2), head: tuple=(1,1), dropout: tuple=(0,0), kind: str='GCN',
                 batch_first: bool=True, bias: tuple=(True, True), variational: bool=True, seed: int=0, gpu: int=None):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.nodes_num = nodes_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.head = head
        self.dropout = dropout
        self.kind = kind
        self.batch_first = batch_first
        self.bias = bias
        self.variational = variational
        self.encoder = GATGRUCell(self.nodes_num, self.input_dim, self.hidden_dim,
                                 num_layers = self.num_layers[0], head = self.head[0],
                                 dropout=self.dropout[0], kind = self.kind,
                                 batch_first = self.batch_first, bias=self.bias[0],
                                 seed = self.seed, gpu = self.gpu)
        self.to_device(self.encoder)
        self.decoder = GATGRUCell(self.nodes_num, self.input_dim, self.hidden_dim,
                                 num_layers = self.num_layers[1], head = self.head[1],
                                 dropout=self.dropout[1], kind = self.kind,
                                 batch_first = self.batch_first, bias=self.bias[1],
                                 seed = self.seed, gpu = self.gpu)
        self.to_device(self.decoder)
        if self.variational == True:
            self.mu = nn.ModuleList([nn.Linear(self.hidden_dim,
                                self.hidden_dim,
                                bias=self.bias[1]) for i in range(self.nodes_num)])
            self.logvar = nn.ModuleList([nn.Linear(self.hidden_dim,
                                    self.hidden_dim,
                                    bias=self.bias[1]) for i in range(self.nodes_num)])
            self.hidden2output_logvar = nn.ModuleList([nn.Linear(self.hidden_dim,
                                                  self.input_dim,
                                                  bias=self.bias[1]) for i in range(self.nodes_num)])
        self.hidden2output = nn.ModuleList([nn.Linear(self.hidden_dim,
                                       self.input_dim,
                                       bias=self.bias[1]) for i in range(self.nodes_num)])

    # The rest of the GraphGRU_VAE class remains the same as the GraphLSTM_VAE class
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, GCNConv) or isinstance(m, GATConv):
                xavier_normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def reparametrize(self, mu, logvar):
        return mu + torch.randn_like(logvar) * torch.exp(logvar)

    def deal_batch(self, conv, x, edge_index):
        batch = Batch.from_data_list([Data(x=x[i], edge_index=edge_index) for i in range(x.shape[0])])

        output = conv(batch.x, batch.edge_index)
        output = output.reshape(x.shape[0], x.shape[1], -1)
        return output
    def deal_list(self, module_list, x):
        x = x.permute(1, 0, 2).contiguous()
        outputs = []
        for i, l in enumerate(module_list):
            output = l(x[i])
            outputs.append(output)
        return torch.stack(outputs, 1)

    def forward(self, ts_batch, edge_index, use_teacher_forcing=True):
        if self.batch_first:
            ts_batch = ts_batch.permute(1, 0, 2, 3).contiguous()
        _, enc_hidden0 = self.encoder(ts_batch.float(), edge_index)
        enc_hidden = enc_hidden0[-1][0]
        if self.variational:
            mu = self.deal_list(self.mu, enc_hidden)
            logvar = self.deal_list(self.logvar, enc_hidden)
            enc_hidden = self.reparametrize(mu, logvar)
        dec_hidden = enc_hidden0
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        output[ts_batch.shape[0] - 1] = self.deal_list(self.hidden2output,
                                                       enc_hidden)
        if self.variational:
            output_logvar = self.to_var(torch.Tensor(ts_batch.size()).zero_())
            output_logvar[ts_batch.shape[0] - 1] = self.deal_list(self.hidden2output_logvar,
                                                                  enc_hidden)
            for i in reversed(range(ts_batch.shape[0] - 1)):
                if self.training and use_teacher_forcing:
                    _, dec_hidden = self.decoder(ts_batch[i + 1].unsqueeze(0).float(), edge_index, dec_hidden)

                else:
                    _, dec_hidden = self.decoder(output[i + 1].unsqueeze(0), edge_index, dec_hidden)
                output[i] = self.deal_list(self.hidden2output, dec_hidden[-1][0])
        else:
            for i in reversed(range(ts_batch.shape[0] - 1)):
                if self.training and use_teacher_forcing:
                    _, dec_hidden = self.decoder(ts_batch[i + 1].unsqueeze(0).float(), edge_index, dec_hidden)

                output[i] = self.deal_list(self.hidden2output, dec_hidden[-1][0])
        return (output.permute(1, 0, 2, 3).contiguous(), enc_hidden, mu, logvar,
                output_logvar.permute(1, 0, 2, 3).contiguous()) if self.variational else (
        output.permute(1, 0, 2, 3).contiguous(), enc_hidden)

    def loss_function(self, preds, labels, mu=None, logvar=None, output_logvar=None):
        if not self.variational:
            recon_loss = torch.mean((preds - labels) ** 2)
            return recon_loss
        else:
            recon_loss = 0.5 * torch.mean(
                torch.sum(torch.div((preds - labels) ** 2, output_logvar.exp()) + output_logvar, (1, 2, 3)))
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), (1, 2)))
            total_loss = recon_loss + kl_loss
            return total_loss, recon_loss, kl_loss