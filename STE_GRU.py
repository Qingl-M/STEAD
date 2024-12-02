import torch.nn as nn
from torch.autograd import Variable
import torch
from torch_geometric.nn import GCNConv, GATConv, GraphConv
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from algorithm_utils import Algorithm, PyTorchUtils


class STEGRUCell(nn.Module, PyTorchUtils):

    def __init__(self, nodes_num, input_dim, hidden_dim, head=1, dropout=0, bias=True, seed: int=0, gpu: int=None):
        super(STEGRUCell, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.nodes_num = nodes_num
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.head = head
        self.dropout = dropout
        self.bias = bias

        self.gconv = GATConv(in_channels=self.input_dim + self.hidden_dim,
                             out_channels=3 * self.hidden_dim,
                             heads=self.head,
                             concat=False,
                             dropout=self.dropout,
                             bias=self.bias)

    def forward(self, input_tensor, cur_state, edge_index):
        h_cur = cur_state[0]

        #    h_cur = torch.stack(h_cur)
        combined = torch.cat([input_tensor, h_cur], dim=2)  #将input和hidden在最后一维拼接
        batch = Batch.from_data_list([Data(x=combined[i], edge_index=edge_index) for i in range(combined.shape[0])])

        combined_conv = self.gconv(batch.x, batch.edge_index)
        combined_conv = combined_conv.reshape(combined.shape[0],combined.shape[1],-1)

        cc_r, cc_z, cc_n = torch.split(combined_conv, self.hidden_dim, dim=2)
        r = torch.sigmoid(cc_r)
        z = torch.sigmoid(cc_z)
        n = torch.tanh(cc_n)

        h_next = (1 - z) * n + z * h_cur

        return h_next

    def init_hidden(self, batch_size):
        return self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim)))
class GATGRUCell(nn.Module, PyTorchUtils):
    def __init__(self, nodes_num, input_dim, hidden_dim, head=1, dropout=0, bias=True, seed: int=0, gpu: int=None):
        super(GATGRUCell, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.nodes_num = nodes_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head = head
        self.dropout = dropout
        self.bias = bias

        self.attn_a = CrossModalAttention(input_dim, hidden_dim)
        self.attn_b = CrossModalAttention(input_dim, hidden_dim)
        self.attn_c = CrossModalAttention(input_dim, hidden_dim)

        self.gconv = GATConv(in_channels=self.hidden_dim,
                             out_channels=3 * self.hidden_dim,  # GRU has 3 gates
                             heads=self.head,
                             concat=False,
                             dropout=self.dropout,
                             bias=self.bias)

    def forward(self, input_tensor, cur_state, edge_index):
        h_cur = cur_state[0]

        # Split input_tensor into three types of channels
        metrics = input_tensor[:, :, :2]
        logs = input_tensor[:, :, 2:4]
        traces = input_tensor[:, :, 4:]

        # Apply cross-modal attention
        attn_a_output = self.attn_a(metrics, torch.cat([logs, traces], dim=2), torch.cat([logs, traces], dim=2))
        attn_b_output = self.attn_b(logs, torch.cat([metrics, traces], dim=2), torch.cat([metrics, traces], dim=2))
        attn_c_output = self.attn_c(traces, torch.cat([metrics, logs], dim=2), torch.cat([metrics, logs], dim=2))

        # Concatenate the attention outputs
        combined_features = torch.cat([attn_a_output, attn_b_output, attn_c_output], dim=2)

        combined = torch.cat([combined_features, h_cur], dim=2)  # concatenate along hidden axis
        batch = Batch.from_data_list([Data(x=combined[i], edge_index=edge_index) for i in range(combined.shape[0])])

        combined_conv = self.gconv(batch.x, batch.edge_index)
        combined_conv = combined_conv.reshape(combined.shape[0], combined.shape[1], -1)

        cc_r, cc_z, cc_n = torch.split(combined_conv, self.hidden_dim, dim=2)
        r = torch.sigmoid(cc_r)
        z = torch.sigmoid(cc_z)
        n = torch.tanh(cc_n)

        h_next = (1 - z) * n + z * h_cur

        return h_next

    def init_hidden(self, batch_size):
        return self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim)))

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CrossModalAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, Q, K, V):
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, V)
        return output