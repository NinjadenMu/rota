import torch
import torch.nn as nn
import torch.optim as optim

import random
import copy

import game

from rotanet import RotaNet, encode_input


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

def generate_data():
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

        if result1 != result2 or random.random() < 0.3:
            data.append(sorted([(pieces1, result1), (pieces2, result2)], key = lambda x: -x[1]))

    return data


def loss_func(reward1, reward2, elem):
    if elem[0][1] == elem[1][1]:
        return torch.square(reward1 - reward2)
    
    return -torch.log(torch.sigmoid(reward1 - reward2))

def train():
    data = generate_data()

    model = RotaNet()

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    losses = []

    for epoch in range(50):
        for elem in data:
            optimizer.zero_grad()

            inp1 = encode_input(elem[0][0])
            inp2 = encode_input(elem[1][0])

            reward1 = model(inp1)
            reward2 = model(inp2)
            
            loss = loss_func(reward1, reward2, elem)
            loss.backward()
            losses.append(loss.item())
            
            optimizer.step()

    print(model(torch.tensor(encode_input([[5, 6, 8], [3, 4, 7]]))))
    print(model(torch.tensor(encode_input([[0, 6, 3], [1, 4, 7]]))))
    torch.save(model.state_dict(), './weights.pt')
    
    return model


if __name__ == '__main__':
    model = train()



    