import torch
from torch import nn


class CDAE(nn.Module):
    def __init__(self, config):
        super(CDAE, self).__init__()
        self.num_users = config["num_users"]
        self.num_items = config["num_items"]
        self.hidden_units = config["hidden_units"]

        # random dropout
        self.dropout = nn.Dropout(p=config["dropout"])
        # ------- encoder -------
        self.h_item = nn.Linear(self.num_items, self.hidden_units)

        act1_type = config["activation_1"]
        if act1_type == "identity":
            self.act1 = nn.Identity()
        elif act1_type == "sigmoid":
            self.act1 = nn.Sigmoid()
        elif act1_type == "relu":
            self.act1 = nn.ReLU()
        else:
            raise ValueError("Invalid activation-1 name: `{}`".format(act1_type))

        self.h_user = nn.Embedding(self.num_users, self.hidden_units)

        # ------- decoder -------
        self.out = nn.Linear(self.hidden_units, self.num_items)
        act2_type = config["activation_2"]
        if act2_type == "identity":
            self.act2 = nn.Identity()
        elif act2_type == "sigmoid":
            self.act2 = nn.Sigmoid()
        elif act2_type == "relu":
            self.act2 = nn.ReLU()
        else:
            raise ValueError("Invalid activation-2 name: `{}`".format(act2_type))

    def forward(self, user_idx, x_items):
        # x_user: batch_size * 1, embed indices

        # z_u = h(W^T y_u + V_u + b)
        x_items = self.dropout(x_items)  # bs * num_items
        h_i = self.h_item(x_items)  # bs * num_f
        h_u = self.h_user(user_idx)  # bs * 1 -> bs * num_f
        z_u = self.act1(h_i + h_u)  # bs * num_f

        # y_{ui} = f(W^T_i z_u + b'_i)
        y = self.out(z_u)  # bs * num_items
        y = self.act2(y)  # bs * num_items
        return y
