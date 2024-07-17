import random
import rotanet
import copy


NUM_SPOTS = 9

ADJACENCY_MATRIX = [
    [0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0]
]

WINNING_COMBOS = [[0, 4, 8], [1, 5, 8], [2, 6, 8], [3, 7, 8]]

pieces = [[-1, -1, -1], [-1, -1, -1]]
to_move = 0

nn_eval = True
if nn_eval:
    model = rotanet.create_model()

def get_legal_moves(to_move, pieces):
    player_pieces = pieces[to_move]

    moves = []

    all_placed = True

    for i, start in enumerate(player_pieces):
        if all_placed and start == -1:
            all_placed = False
            for stop in range(NUM_SPOTS):
                if stop not in pieces[0] and stop not in pieces[1]:
                    moves.append((start, stop))

        else:
            pseudolegal_moves = ADJACENCY_MATRIX[start]

            for stop in range(NUM_SPOTS):
                if pseudolegal_moves[stop]:
                    if stop not in pieces[0] and stop not in pieces[1]:
                        moves.append((start, stop))    

    return moves

def check_for_win(pieces):
    pieces[0].sort()
    pieces[1].sort()

    if pieces[0][-1] == 8:
        if pieces[0] in WINNING_COMBOS:
            return 100

    if pieces[1][-1] == 8:
        if pieces[1] in WINNING_COMBOS:
            return -100

    return 0

def make_move(to_move, pieces, move):
    pieces[to_move][pieces[to_move].index(move[0])] = move[1]

def undo_move(to_move, pieces, move):
    pieces[to_move][pieces[to_move].index(move[1])] = move[0]

def search(to_move, pieces, moves, depth, pv, transposition_table = {}, alpha = -300, beta = 300):
    pieces_hash = (''.join(list(map(str, sorted(pieces[0])))) + ''.join(list(map(str, sorted(pieces[1])))) + str(to_move)).ljust(13, '0')
    if pieces_hash in transposition_table and transposition_table[pieces_hash][1]:
        return 0
    
    result = check_for_win(pieces)

    if result:
        transposition_table[pieces_hash] = [result, 0]

        if result < 0:
            result = result - depth

        else:
            result = result + depth

        return result

    if depth == 0:
        if nn_eval:
            if pieces_hash in transposition_table and transposition_table[pieces_hash] != 300:
                result = transposition_table[pieces_hash][0]

            else:
                if to_move:
                    eval_pieces = [[], []]
                    temp = copy.copy(pieces[0])
                    eval_pieces[0] = copy.copy(pieces[1])
                    eval_pieces[1] = temp

                else:
                    eval_pieces = pieces

                inp = rotanet.encode_input(eval_pieces)
                result = model(inp).item() - 0.5

        transposition_table[pieces_hash] = [result, 0]

        return result
    
    if pieces_hash in transposition_table:
        transposition_table[pieces_hash][1] = 1
    else:
        transposition_table[pieces_hash] = [300, 1]

    best_result = -300
    worst_result = 300
    for move in moves:
        make_move(to_move, pieces, move)
        to_move = not to_move

        opp_moves = get_legal_moves(to_move, pieces)
        child_pv = []

        result = search(to_move, pieces, opp_moves, depth - 1, child_pv, transposition_table, alpha, beta)

        best_result = max(result, best_result)
        worst_result = min(result, worst_result)

        to_move = not to_move
        undo_move(to_move, pieces, move)


        if not to_move:
            if best_result > alpha and best_result < beta:
                pv.clear()
                pv.append(move)
                pv += child_pv

            alpha = max(alpha, best_result)
            if beta <= alpha:
                print(1)
                break

        else:
            if worst_result > alpha and worst_result < beta:
                pv.clear()
                pv.append(move)
                pv += child_pv
                
            beta = min(beta, worst_result)
            if beta <= alpha:
                print(1)
                break

    transposition_table[pieces_hash][1] = 0

    if not to_move:
        return best_result

    return worst_result


if __name__ == '__main__':
    #print(search(0, [[0, 6, 3], [1, 4, 7]], get_legal_moves(0, [[0, 6, 3], [1, 4, 7]]), 8, []))
    while not check_for_win(pieces):
        print(pieces)

        legal_moves = get_legal_moves(to_move, pieces)
        if not to_move:
            while True:
                try:
                    print('Spots are numbered 0-8.  0-7 run clockwise along the outside circle.  8 is the center.  An unplaced spot can be represented as -1.')
                    move = tuple(map(int, input('Please input your move in format {start spot}->{end spot}: ').split('->')))

                    if move in legal_moves:
                        break

                    else:
                        print('Illegal Move!')

                except:
                    print('Invalid Input!')

        else:
            pv = []
            depth = 9
            if (pieces[0].count(-1) + pieces[1].count(-1)) > 4:
                depth = 6
            print(depth)
            result = search(to_move, pieces, legal_moves, 5, pv)
            move = pv[0]
            print(result)
            print(pv)

        make_move(to_move, pieces, move)

        to_move = not to_move

    print(check_for_win(pieces))
    print('Game Over!')

"""to_move = 0
pieces = [[3, 1, 6], [4, 0, 7]]
pv = []

#pieces_hash = (''.join(list(map(str, sorted(pieces[0])))) + ''.join(list(map(str, sorted(pieces[1])))) + str(to_move)).ljust(13, '0')
#print(pieces_hash)

print(search(to_move, pieces, get_legal_moves(to_move, pieces), 6, pv))
print(pv)"""

