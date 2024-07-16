import torch
import torch.nn as nn
import torch.optim as optim
import game
import numpy as np
import random
import copy

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

def sample_game(pieces, moves_in):
    to_move = 0
    move_counter = 0
    while move_counter < moves_in and not game.check_for_win(pieces):
        legal_moves = game.get_legal_moves(to_move, pieces)
        move = random.choice(legal_moves)

        game.make_move(to_move, pieces, move)
        to_move = not to_move

        move_counter += 1

    if game.check_for_win(pieces):
        to_move = not to_move
        game.undo_move(to_move, pieces, move)

    if to_move:
        temp = copy.copy(pieces[0])
        pieces[0] = copy.copy(pieces[1])
        pieces[1] = temp

data = []

while len(data) < 1000:
    pieces1 = [[-1, -1, -1], [-1, -1, -1]]
    pieces2 = [[-1, -1, -1], [-1, -1, -1]]

    moves_in = random.randint(1, 25)

    sample_game(pieces1, moves_in)
    sample_game(pieces2, moves_in)

    legal_moves1 = game.get_legal_moves(0, pieces1)
    legal_moves2 = game.get_legal_moves(0, pieces2)
    result1 = game.search(0, pieces1, legal_moves1, 6, [])
    result2 = game.search(0, pieces2, legal_moves2, 6, [])

    if result1 != result2:
        data.append(sorted([(pieces1, result1), (pieces2, result2)], key = lambda x: -x[1]))

model = RotaNet()

def loss_func(reward1, reward2):
    return -torch.log(torch.sigmoid(reward1 - reward2))

def encode_input(pieces):
    inp = torch.zeros(18)
    for i, side in enumerate(pieces):
        for piece in side:
            if piece != -1:
                inp[9 * i + piece] = 1.0

    return inp

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
losses = []

#print(loss_func(torch.tensor([1.]), torch.tensor([3.])))
for epoch in range(50):
    for elem in data:
        optimizer.zero_grad()

        inp1 = encode_input(elem[0][0])
        inp2 = encode_input(elem[1][0])

        reward1 = model(inp1)
        reward2 = model(inp2)
        
        if random.random() < 0.05:
            print('r1')
            print(reward1)
            print(reward2)
        
        loss = loss_func(reward1, reward2)
        loss.backward()
        losses.append(loss.item())
        
        optimizer.step()

print(encode_input([[-1, -1, -1], [-1, -1, -1]]))
print(model(torch.tensor(encode_input([[3, 7, 6], [2, 4, 5]]))))
print(model(torch.tensor(encode_input([[0, 6, 3], [1, 4, 7]]))))
torch.save(model.state_dict(), './weights.pt')



    