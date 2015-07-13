import time
import constants


class NonRandomPlayer(object):
    """
    good things:
    - multiple heuristics (weighted sum)
    - automatic search depth calibration for slow computers (works even with cpu limited to 1% nominal speed)
    - enforced time limit per move
    - virtually no memory use
    - pep 8 compliant code style
    - documented and commented and nicely named
    - self-trained ai (played with itself with varying heuristic weights to improve weights)
    - perfect endgame (complete search)
    - pretty prints board and diagnostic info each step
    - modular code
    - negamax search (variation on alpha-beta)
    """

    def __init__(self, color):
        # sanity checks
        assert color in 'bwBW', 'ERROR: color must be B or W'
        assert constants.BRD_SIZE == 8, 'ERROR: heuristics only configured to run on 8x8 board'
        # assert constants.MEMORY_LIMIT_MB == 82, 'ERROR: memory limit has been modified'
        # assert constants.TIME_LIMIT_SEC >= 30 * 60, 'ERROR: time limit should be at least 30 minutes'

        # init variables
        self.board_evaluation_count = None
        self.player_color = color.upper()
        self.other_color = {'B': 'W', 'W': 'B'}[self.player_color]
        self.start_time = None
        self.move = 0

        # self-calibration
        cpu_speed = self.cpu_benchmark()
        self.exhaustive_threshold = 10 + cpu_speed  # worst case seen: 12 minutes for 14-ply search
        self.time_limiter = [[],[],[], [54], [54, 59], [54, 57, 59], [54, 57, 58, 59]][cpu_speed] + [65, 65]
        self.search_depth = len(self.time_limiter) - 1

        # verbose
        print ' init: NonRandomPlayer'
        print 'color:', {'W': 'white', 'B': 'black'}[self.player_color]
        print 'depth: %d ply' % (self.search_depth + 1)
        print 'final: %d ply exhaustive search' % self.exhaustive_threshold

    def chooseMove(self, board, previous_move):
        """
        uses camelcase because calling code requires it

        """
        # diagnostics
        self.start_time = time.time()
        self.board_evaluation_count = 0
        self.move += 1

        # pretty-print board
        print ''
        print ' move: %d (%s)' % (self.move, {'W': 'white', 'B': 'black'}[self.player_color])
        print 'board:', ' '.join(board[0]).replace('B', 'X').replace('W', 'O').replace('G', '.')
        for row in board[1:]:
            print '      ', ' '.join(row).replace('B', 'X').replace('W', 'O').replace('G', '.')

        # get next move
        blanks = self.count_unclaimed_cells(board)
        if blanks < self.exhaustive_threshold:
            print ' type: exhaustive search (%d ply)' % blanks
            move = self.exhaustive_negamax(board, self.player_color, self.other_color, 2 * blanks)
        elif blanks == self.exhaustive_threshold:
            # and if time left >= 15 * 60 (seconds)
            print ' type: exhaustive search (%d ply)' % blanks
            move = self.exhaustive_negamax(board, self.player_color, self.other_color, 2 * blanks)
        else:
            print ' type: heuristic search'
            move = self.heuristic_negamax(board, self.player_color, self.other_color, self.search_depth)

        # print diagnostics
        print 'value:', move[0]
        print ' seen:', self.board_evaluation_count
        print ' time:', time.time() - self.start_time
        print ' move:', move[1]
        return move[1]

    """""""""""""""""""""""""""""""""
            SEARCH FUNCTIONS
    """""""""""""""""""""""""""""""""

    def heuristic_negamax(self, board, player_color, other_color, search_depth, v_max=99999, v_min=-99999):
        # recursion termination
        if search_depth < 0 or (time.time() - self.start_time) > self.time_limiter[search_depth]:
            self.board_evaluation_count += 1
            return self.othello_heuristic(board, player_color), None

        # find next moves and play them
        moves = self.get_valid_moves(board, player_color, other_color) or [None]
        next_boards = (self.apply_move(board, move, player_color, other_color) for move in moves)

        # recursion
        values = []
        for next_board in next_boards:
            value = -self.heuristic_negamax(next_board, other_color, player_color, search_depth - 1, -v_min, -v_max)[0]
            values.append(value)
            v_min = max(v_min, value)
            if v_min > v_max:
                break
        return max(zip(values, moves))

    def exhaustive_negamax(self, board, player_color, other_color, search_depth, v_max=64, v_min=-64):
        """
        :param search_depth: 2x remaining blank cells
        """
        # recursion termination
        if search_depth < 0 or self.count_unclaimed_cells(board) == 0:
            self.board_evaluation_count += 1
            return self.heuristic_helper(1, board, player_color, self.heuristic_count), None

        # find and apply available moves
        moves = self.get_valid_moves(board, player_color, other_color) or [None]
        # next_boards = (self.apply_move(board, move, player_color, other_color) for move in moves)

        # recursion
        values = []
        # for next_board in next_boards:
        for move in moves:
            next_board = self.apply_move(board, move, player_color, other_color)
            value = -self.exhaustive_negamax(next_board, other_color, player_color, search_depth - 1, -v_min, -v_max)[0]
            values.append(value)
            v_min = max(v_min, value)
            if v_min > v_max:
                break
        return max(zip(values, moves))

    """""""""""""""""""""""""""""""""
            HEURISTIC FUNCTIONS
    """""""""""""""""""""""""""""""""

    def othello_heuristic(self, board, player_color):
        """
        computes a weighted sum of multiple heuristics
        the static-weight heuristic is treated as a default
            against which all other heuristics are scaled appropriately
        note: this heuristic is neither admissible nor consistent
            that said, it does work pretty dang well

        there's a zero added at the bottom (and a blank return on top)
            to make all the lines consistent
                to make the heuristics easier to read
        """
        return \
            self.heuristic_helper(20, board, player_color, self.heuristic_corners_captured)   + \
            self.heuristic_helper(20, board, player_color, self.heuristic_partial_edge)       + \
            self.heuristic_helper( 7, board, player_color, self.heuristic_stability)          + \
            self.heuristic_helper( 6, board, player_color, self.heuristic_semi_stability)     + \
            self.heuristic_helper( 4, board, player_color, self.heuristic_mobility)           + \
            self.heuristic_helper( 1, board, player_color, self.heuristic_static_weight)      + \
            self.heuristic_helper( 0, board, player_color, self.heuristic_count)              + \
            0

    @staticmethod
    def heuristic_helper(multiplier, board, player_color, heuristic_function):
        if not multiplier:
            return 0

        white_score, black_score = heuristic_function(board)
        if player_color == 'W':
            return (white_score - black_score) * multiplier
        else:
            return (black_score - white_score) * multiplier

    @staticmethod
    def heuristic_count(board):
        """
        simple count of pieces on board

        expected output range: 0 to 64
        """
        white_pieces = sum(row.count('W') for row in board)
        black_pieces = sum(row.count('B') for row in board)
        return white_pieces, black_pieces

    def heuristic_mobility(self, board):
        """
        counts moves left to self and opponent
        strong negative response if 0 moves exist

        expected output range: -9, or 1 to 10
        """
        white_moves = len(self.get_valid_moves(board, 'W', 'B')) or -9
        black_moves = len(self.get_valid_moves(board, 'B', 'W')) or -9
        return white_moves, black_moves

    @staticmethod
    def heuristic_stability(board):
        """
        counts cells forming triangles at the corner
            which can never be captured
            these cells permanently belong to whomever claimed them

        expected output range: 0 to 20
        """
        corner_sequences = [[[(0, 0)],
                             [(0, 1), (1, 0)],
                             [(0, 2), (1, 1), (2, 0)],
                             [(0, 3), (1, 2), (2, 1), (3, 0)],
                             [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)],
                             [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)],
                             [(0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0)],
                             [(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)],
                             [(1, 7), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (7, 1)],
                             [(2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2)],
                             [(3, 7), (4, 6), (5, 5), (6, 4), (7, 3)],
                             [(4, 7), (5, 6), (6, 5), (7, 4)],
                             [(5, 7), (6, 6), (7, 5)],
                             [(6, 7), (7, 6)],
                             [(7, 7)]],
                            [[(0, 7)],
                             [(0, 6), (1, 7)],
                             [(0, 5), (1, 6), (2, 7)],
                             [(0, 4), (1, 5), (2, 6), (3, 7)],
                             [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7)],
                             [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)],
                             [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
                             [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
                             [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6)],
                             [(2, 0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5)],
                             [(3, 0), (4, 1), (5, 2), (6, 3), (7, 4)],
                             [(4, 0), (5, 1), (6, 2), (7, 3)],
                             [(5, 0), (6, 1), (7, 2)],
                             [(6, 0), (7, 1)],
                             [(7, 0)]],
                            [[(7, 7)],
                             [(7, 6), (6, 7)],
                             [(7, 5), (6, 6), (5, 7)],
                             [(7, 4), (6, 5), (5, 6), (4, 7)],
                             [(7, 3), (6, 4), (5, 5), (4, 6), (3, 7)],
                             [(7, 2), (6, 3), (5, 4), (4, 5), (3, 6), (2, 7)],
                             [(7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7)],
                             [(7, 0), (6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6), (0, 7)],
                             [(6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6)],
                             [(5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5)],
                             [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)],
                             [(3, 0), (2, 1), (1, 2), (0, 3)],
                             [(2, 0), (1, 1), (0, 2)],
                             [(1, 0), (0, 1)],
                             [(0, 0)]],
                            [[(7, 0)],
                             [(7, 1), (6, 0)],
                             [(7, 2), (6, 1), (5, 0)],
                             [(7, 3), (6, 2), (5, 1), (4, 0)],
                             [(7, 4), (6, 3), (5, 2), (4, 1), (3, 0)],
                             [(7, 5), (6, 4), (5, 3), (4, 2), (3, 1), (2, 0)],
                             [(7, 6), (6, 5), (5, 4), (4, 3), (3, 2), (2, 1), (1, 0)],
                             [(7, 7), (6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1), (0, 0)],
                             [(6, 7), (5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1)],
                             [(5, 7), (4, 6), (3, 5), (2, 4), (1, 3), (0, 2)],
                             [(4, 7), (3, 6), (2, 5), (1, 4), (0, 3)],
                             [(3, 7), (2, 6), (1, 5), (0, 4)],
                             [(2, 7), (1, 6), (0, 5)],
                             [(1, 7), (0, 6)],
                             [(0, 7)]]]

        # count for white
        white_cells = set()
        for corner_seq in corner_sequences:
            for diagonal_seq in corner_seq:
                temp = [cell for cell in diagonal_seq if board[cell[0]][cell[1]] == 'W']
                if temp == diagonal_seq:
                    white_cells.update(diagonal_seq)
                else:
                    break

        # count for black
        black_cells = set()
        for corner_seq in corner_sequences:
            for diagonal_seq in corner_seq:
                temp = [cell for cell in diagonal_seq if board[cell[0]][cell[1]] == 'B']
                if temp == diagonal_seq:
                    white_cells.update(diagonal_seq)
                else:
                    break
        return len(white_cells), len(black_cells)

    @staticmethod
    def heuristic_semi_stability(board):
        """
        stable moves cannot be taken by the opponent
        this counts semi-stable moves, which are difficult for the opponent to take
        these are cells where everything in a rectangle behind it to the corner is taken

        expected output range: 0 to 20
        """
        corners = [(0, 0), (0, 7), (7, 7), (7, 0)]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 1)]

        # count for white
        white_cells = set()
        for i, corner in enumerate(corners):
            temp = []
            while (0 <= corner[0] <= 7) and (0 <= corner[1] <= 7) and board[corner[0]][corner[1]] == 'W':
                temp.append(corner)
                corner = (corner[0] + directions[i][0], corner[1] + directions[i][1])
            limit = 8
            for cell in temp:
                j = 0
                while (0 <= cell[0] <= 7) and (0 <= cell[1] <= 7) and board[cell[0]][cell[1]] == 'W' and limit > j:
                    j += 1
                    white_cells.add(cell)
                    cell = (cell[0] + directions[i + 1][0], cell[1] + directions[i + 1][1])
                limit = min(limit, j)

        # count for black
        black_cells = set()
        for i, corner in enumerate(corners):
            temp = []
            while (0 <= corner[0] <= 7) and (0 <= corner[1] <= 7) and board[corner[0]][corner[1]] == 'B':
                temp.append(corner)
                corner = (corner[0] + directions[i][0], corner[1] + directions[i][1])
            limit = 8
            for cell in temp:
                j = 0
                while (0 <= cell[0] <= 7) and (0 <= cell[1] <= 7) and board[cell[0]][cell[1]] == 'B' and limit > j:
                    j += 1
                    black_cells.add(cell)
                    cell = (cell[0] + directions[i + 1][0], cell[1] + directions[i + 1][1])
                limit = min(limit, j)

        return len(white_cells), len(black_cells)

    @staticmethod
    def heuristic_corners_captured(board):
        """
        counts corners

        expected output range: 0 to 4
        """
        corners = [board[0][0], board[0][-1], board[-1][0], board[-1][-1]]
        white_corners = corners.count('W')
        black_corners = corners.count('B')
        return white_corners, black_corners

    @staticmethod
    def heuristic_partial_edge(board):
        """
        counts partial edges
        these are the middle 4 cells along each edge
        they should be weighted enough to encourage taking the corner-adjacent cell to gain an edge

        expected output range: 0 to 4
        """
        edges = [board[0][2:6], board[7][2:6], [col[0] for col in board[2:6]], [col[7] for col in board[2:6]]]
        white_partial_edges = sum(edge.count('W') == 4 for edge in edges)
        black_partial_edges = sum(edge.count('B') == 4 for edge in edges)
        return white_partial_edges, black_partial_edges

    @staticmethod
    def heuristic_static_weight(board):
        """
        counts a weighted sum of occupied cells encoding several preferences:
        - strong preference not to play adjacent to corners
        - sufficient corner preference to neutralize the surrounding 3 cells
        - preference take edge if possible
        - mild preference not to to play adjacent to edge
        - slight preference to take center diagonals

        expected output range: -50 to 150
        """
        weights = [ 40, -13, 2.7, 2.6, 2.6, 2.7, -13,  40,
                   -13, -13, -.8, -.8, -.8, -.8, -13, -13,
                   2.7, -.8, 1.5, 1.3, 1.3, 1.5, -.8, 2.7,
                   2.6, -.8, 1.3,   1,   1, 1.3, -.8, 2.6,
                   2.6, -.8, 1.3,   1,   1, 1.3, -.8, 2.6,
                   2.7, -.8, 1.5, 1.3, 1.3, 1.5, -.8, 2.7,
                   -13, -13, -.8, -.8, -.8, -.8, -13, -13,
                    40, -13, 2.7, 2.6, 2.6, 2.7, -13,  40]
        flat_board = [item for row in board for item in row]
        white_score = sum(weight for weight, cell in zip(weights, flat_board) if cell == 'W')
        black_score = sum(weight for weight, cell in zip(weights, flat_board) if cell == 'B')
        return white_score, black_score

    """""""""""""""""""""""""""""""""
            UTILITY FUNCTIONS
    """""""""""""""""""""""""""""""""

    @staticmethod
    def cpu_benchmark():
        """
        tests computer speed
        smaller is slower

        expected output range: 0 to 3
        """
        # speed test
        time_taken = time.time()
        for i in range(1500):
            _ = i ** i
        time_taken = time.time() - time_taken

        # scaling
        score = 0
        for score, threshold in enumerate([1.6, 0.8, 0.4, 0.2]):
            if time_taken > threshold:
                return score
        return score + 1

    @staticmethod
    def count_unclaimed_cells(board):
        return sum(row.count('G') for row in board)

    @staticmethod
    def is_move_valid(board, cell_position, direction_vector, player_color, other_color):
        """
        this is an amazingly important function
        it's called millions of times per turn (worst seen: 226,146,909 calls = 286,737 ms own time)
        caching is hard because its called with many irrelevant boards

        """
        # init
        row, col = cell_position
        delta_row, delta_col = direction_vector
        other_color_seen = False

        # step
        row += delta_row
        col += delta_col

        # other color pieces
        while 0 <= row <= 7 and 0 <= col <= 7 and board[row][col] == other_color:
            other_color_seen = True
            row += delta_row
            col += delta_col

        # own color piece
        return 0 <= row <= 7 and 0 <= col <= 7 and board[row][col] == player_color and other_color_seen

    def apply_move(self, board, cell_position, player_color, other_color):
        # clone
        board_copy = [row[:] for row in board]

        # check invalid move
        if cell_position is None:
            return board_copy

        # set target cell
        board_copy[cell_position[0]][cell_position[1]] = player_color

        # flip nearby cells
        for direction in constants.DIRECTIONS:
            if self.is_move_valid(board_copy, cell_position, direction, player_color, other_color):
                next_position = (cell_position[0] + direction[0], cell_position[1] + direction[1])
                while board_copy[next_position[0]][next_position[1]] == other_color:
                    board_copy[next_position[0]][next_position[1]] = player_color
                    next_position = (next_position[0] + direction[0], next_position[1] + direction[1])

        return board_copy

    def get_valid_moves(self, board, player_color, other_color):
        moves = []
        for i in xrange(8):
            for j in xrange(8):
                if board[i][j] != 'G':
                    continue  # background is green, i.e., empty cell
                for direction in constants.DIRECTIONS:
                    if self.is_move_valid(board, (i, j), direction, player_color, other_color):
                        moves.append((i, j))
                        break
        return moves

    """""""""""""""""""""""""""""""""
           NON-USEFUL FUNCTIONS
    """""""""""""""""""""""""""""""""

    def gameEnd(self, board):
        """
        uses camelcase because calling code requires it
        """
        pass

    def getColor(self):
        """
        uses camelcase because calling code requires it
        """
        return self.player_color


if __name__ == '__main__':

    def test():
        player_white = NonRandomPlayer('w')
        player_black = NonRandomPlayer('b')
        player_white.board_evaluation_count = 0

        test_board = [['G', 'W', 'W', 'B', 'G', 'W', 'B', 'W'],
                      ['G', 'B', 'B', 'G', 'W', 'B', 'W', 'G'],
                      ['B', 'B', 'B', 'B', 'B', 'B', 'G', 'B'],
                      ['G', 'G', 'G', 'W', 'W', 'G', 'B', 'G'],
                      ['B', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                      ['G', 'W', 'B', 'B', 'W', 'G', 'W', 'W'],
                      ['B', 'B', 'B', 'W', 'W', 'W', 'W', 'W'],
                      ['B', 'W', 'W', 'W', 'W', 'G', 'W', 'W']]
        time_start = time.time()

        print player_white.count_unclaimed_cells(test_board)
        # print bot.exhaustive_negamax(board_, 'W', 20)
        # print bot.exhaustive_negamax(board_, 'B', 20)
        player_white.chooseMove(test_board, 0)
        player_black.chooseMove(test_board, 0)
        print time.time() - time_start

    test()
