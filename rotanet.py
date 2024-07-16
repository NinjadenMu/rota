import torch
import torch.nn as nn


def encode_input(pieces):
    inp = torch.zeros(18)
    for i, side in enumerate(pieces):
        for piece in side:
            if piece != -1:
                inp[9 * i + piece] = 1.0

    return inp


class RotaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dl1 = nn.Linear(18, 16)
        self.dl2 = nn.Linear(16, 16)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)

        x = self.dl2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
  
        return x
    

if __name__ == '__main__':
    model = RotaNet()
    model.load_state_dict(torch.load('./weights.pt'))

    print(model(encode_input([[-1, -1, 0], [1, 8, -1]])))