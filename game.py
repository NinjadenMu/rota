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


def get_legal_moves(to_move, pieces):
    player_pieces = pieces[to_move]

    moves = []
    all_tiles_placed = True

    for i, start in enumerate(player_pieces):
        if all_tiles_placed and start == -1:
            all_tiles_placed = False

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
            return 1

    if pieces[1][-1] == 8:
        if pieces[1] in WINNING_COMBOS:
            return -1

    return 0

def make_move(to_move, pieces, move):
    pieces[to_move][pieces[to_move].index(move[0])] = move[1]

def undo_move(to_move, pieces, move):
    pieces[to_move][pieces[to_move].index(move[1])] = move[0]

def search(to_move, pieces, moves, depth, alpha, beta):
    result = check_for_win(pieces)
    if result:
        if result < 0:
            return result - (depth / (depth + 2))

        return result + (depth / (depth + 2))

    if depth == 0:
        return result

    if not len(moves):
        return result

    best_result = -3
    worst_result = 3
    for move in moves:
        make_move(to_move, pieces, move)
        to_move = not to_move

        opp_moves = get_legal_moves(to_move, pieces)
        result = search(to_move, pieces, opp_moves, depth - 1, alpha, beta)
        best_result = max(result, best_result)
        worst_result = min(result, worst_result)

        to_move = not to_move
        undo_move(to_move, pieces, move)

        if not to_move:
            alpha = max(alpha, best_result)
            if beta <= alpha:
                break

        else:
            beta = min(beta, worst_result)
            if beta <= alpha:
                break

    if not to_move:
        return best_result

    return worst_result

#print(search(to_move, pieces, get_legal_moves(to_move, pieces), 8, -1, 1))
while not check_for_win(pieces):
    print(pieces)

    moves = get_legal_moves(to_move, pieces)
    results = []

    for move in moves:
        results.append(search(to_move, pieces, [move], 7, -1, 1))

    if not to_move:
        make_move(to_move, pieces, moves[results.index(max(results))])

    else:
        make_move(to_move, pieces, moves[results.index(min(results))])

    to_move = not to_move


print(pieces)
print(check_for_win(pieces))

while not check_for_win(pieces):
    print(pieces)

    legal_moves = get_legal_moves(to_move, pieces)
    if to_move == 0:
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
        move = legal_moves[0]

    make_move(to_move, pieces, move)

    to_move = not to_move

print(check_for_win(pieces))
print('Game Over!')

