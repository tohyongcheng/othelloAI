import random, memory, constants,copy
import sched, time
from threading import Thread

openingWeights = []
openingWeights.append([0, 0, 0, 0, 0, 0, 0, 0])
openingWeights.append([0, -0.02231, 0.05583, 0.02004, 0.02004, 0.05583, -0.02231, 0])
openingWeights.append([0, 0.05583, 0.10126, -0.10927, -0.10927, 0.10126, 0.05583, 0])
openingWeights.append([0, 0.02004, -0.10927, -0.10155, -0.10155, -0.10927, 0.02004, 0])
openingWeights.append([0, 0.02004, -0.10927, -0.10155, -0.10155, -0.10927, 0.02004, 0])
openingWeights.append([0, 0.05583, 0.10126, -0.10927, -0.10927, 0.10126, 0.05583, 0])
openingWeights.append([0, -0.02231, 0.05583, 0.02004, 0.02004, 0.05583, -0.02231, 0])
openingWeights.append([0, 0, 0, 0, 0, 0, 0, 0])
middleWeights = []
middleWeights.append([ 6.32711, -3.32813, 0.33907, -2.00512, -2.00512, 0.33907, -3.32813, 6.32711])
middleWeights.append([-3.32813, -1.52928, -1.87550, -0.18176, -0.18176, -1.87550, -1.52928, -3.32813])
middleWeights.append([ 0.33907, -1.87550, 1.06939, 0.62415, 0.62415, 1.06939, -1.87550, 0.33907])
middleWeights.append([-2.00512, -0.18176, 0.62415, 0.10539, 0.10539, 0.62415, -0.18176, -2.00512])
middleWeights.append([-2.00512, -0.18176, 0.62415, 0.10539, 0.10539, 0.62415, -0.18176, -2.00512])
middleWeights.append([ 0.33907, -1.87550, 1.06939, 0.62415, 0.62415, 1.06939, -1.87550, 0.33907])
middleWeights.append([-3.32813, -1.52928, -1.87550, -0.18176, -0.18176, -1.87550, -1.52928, -3.32813])
middleWeights.append([ 6.32711, -3.32813, 0.33907, -2.00512, -2.00512, 0.33907, -3.32813, 6.32711])
endWeights = []
endWeights.append([5.50062, -0.17812, -2.58948, -0.59007, -0.59007, -2.58948, -0.17812, 5.50062])
endWeights.append([-0.17812, 0.96804, -2.16084, -2.01723, -2.01723, -2.16084, 0.96804, -0.17812])
endWeights.append([ -2.58948, -2.16084, 0.49062, -1.07055, -1.07055, 0.49062, -2.16084, -2.58948])
endWeights.append([ -0.59007, -2.01723, -1.07055, 0.73486, 0.73486, -1.07055, -2.01723, -0.59007])
endWeights.append([ -0.59007, -2.01723, -1.07055, 0.73486, 0.73486, -1.07055, -2.01723, -0.59007])
endWeights.append([ -2.58948, -2.16084, 0.49062, -1.07055, -1.07055, 0.49062, -2.16084, -2.58948])
endWeights.append([-0.17812, 0.96804, -2.16084, -2.01723, -2.01723, -2.16084, 0.96804, -0.17812])
endWeights.append([5.50062, -0.17812, -2.58948, -0.59007, -0.59007, -2.58948, -0.17812, 5.50062])
tryWeights = []
tryWeights.append([20, -3, 11, 8, 8, 11, -3, 20])
tryWeights.append([-3, -7, -4, 1, 1, -4, -7, -3])
tryWeights.append([11, -4, 2, 2, 2, 2, -4, 11])
tryWeights.append([8, 1, 2, -3, -3, 2, 1, 8])
tryWeights.append([8, 1, 2, -3, -3, 2, 1, 8])
tryWeights.append([11, -4, 2, 2, 2, 2, -4, 11])
tryWeights.append([-3, -7, -4, 1, 1, -4, -7, -3])
tryWeights.append([20, -3, 11, 8, 8, 11, -3, 20])

# b_w_score + d_score + frontier_score + mobility_score + corners_score + diagonal_stability_score + horizontal_stability_score + corner_closeness_score + partial_edge_score
weights = { "b_w_score": 0.1,
            "d_score": 1.0,
            "frontier_score": 4.0,
            "mobility_score": 4.0,
            "corners_score": 20.0,
            "diagonal_stability_score": 6.0,
            "horizontal_stability_score": 6.0,
            "corner_closeness_score": 0.0,
            "partial_edge_score": 20.0
          }

class YcSmartestPlayer:

  def __init__(self,color):
      self.color = color
      self.player_color = color
      self.other_color = self.oppositeColor(color)
      self.table = {}
      self.savedMoves = {}
      self.board_evaluation_count = None
      self.depth = 6

      ### Memory Management
      self.manager = None
      self.scheduler = sched.scheduler(time.time, time.sleep)
      self.event = None
      #### profiling
      self.calls = 0.0
      self.hits = 0.0


  def chooseMove(self,board,prevMove):
      color = self.color
      if   color == 'W': oppColor = 'B'
      elif color == 'B': oppColor = 'W'
      else: assert False, 'ERROR: Current player is not W or B!'

      if sum(self.computeScore(board)) >= 52:
        self.depth = 12

      result = self.alphabeta(board, self.depth, -constants.INFINITY, constants.INFINITY, True)

      return result[1]


  def gameEnd(self,board):
      return

  def getColor(self):
      return self.color

  def getMemoryUsedMB(self):
      return 0.0

  def toBitBoard(self,board):
      #helper function that turns a board to a bitboard
      whiteBoard = 0 # dw about starting 0s... can handle automatically what!
      blackBoard = 0
      for i in xrange(constants.BRD_SIZE):
          for j in xrange(constants.BRD_SIZE):
              whiteBoard = whiteBoard << 1
              blackBoard = blackBoard << 1
              if board[i][j] == 'W':
                 whiteBoard += 1
              if board[i][j] == 'B':
                 blackBoard += 1

      return (whiteBoard,blackBoard)

  def evaluate(self,board, bitboard, endGame = False):
      score = 0.0

      oppColor = self.oppositeColor(self.color)
      w,b = self.computeScore(board)
      

      # weights["b_w_score"] = 1.0 + 3.0*(b+w)/64
      # weights["d_score"] = 0.0
      # weights["frontier_score"] = 5.0 + (-3.0)*(b+w)/64
      # weights["mobility_score"] = 5.0 + 3.0*(b+w)/64
      # weights["corners_score"] = 25.0 + -10.0*(b+w)/64
      # weights["diagonal_stability_score"] = 1.0 + 5.0*(b+w)/64
      # weights["horizontal_stability_score"] = 1.0 + 5.0*(b+w)/64
      # weights["corner_closeness_score"] = 10.0 + -20.0*(b+w)/64
      # weights["partial_edge_score"] = 15.0 + -10.0*(b+w)/64

      b_w_score = 0.0
      # Coin Parity
      if self.color == "W":
        b_w_score += (w-b)
      else:
        b_w_score += (b-w)

      b_w_score *= weights["b_w_score"]

      if endGame:
        return b_w_score*10000
      # else:
      #   b_w_score *= -1
      
      # Mobility
      mobility_score = 0.0
      my_moves = len(self.findAllMovesHelper(board, self.color, oppColor, bitboard)) or -9
      opp_moves = len(self.findAllMovesHelper(board, oppColor, self.color, bitboard)) or -9
      mobility_score += weights["mobility_score"] * (my_moves - opp_moves) 

      # Feature Weights on squares dependent on game stage
      my_frontier_tiles = opp_frontier_tiles = 0.0
      my_d = 0.0
      opp_d = 0.0
      
      # Game Stage Dependent Weights
      # currentStageWeights = openingWeights
      # if self.no_of_corners_occupied(board) >= 2:
      #   currentStageWeights = endWeights
      # elif self.disc_present_on_edges(board):
      #   currentStageWeights = middleWeights
      # currentStageWeights = anotherWeights
      currentStageWeights = tryWeights

      for i in xrange(8):
        for j in xrange(8):
          if board[i][j] == self.color:
            my_d += currentStageWeights[i][j]
          elif board[i][j] == oppColor:
            opp_d += currentStageWeights[i][j]

          if board[i][j] != "G":
            for ddir in constants.DIRECTIONS:
              if self.validPos((i+ddir[0],j+ddir[1])) and board[i+ddir[0]][j+ddir[1]] == "G":
                if board[i][j] == self.color:
                  my_frontier_tiles += 1
                else:
                  opp_frontier_tiles += 1

      d_score = weights["d_score"] * ( my_d - opp_d )

      # Frontier Score - Potential Mobility
      frontier_score = 0.0
      frontier_score += weights["frontier_score"] * -(my_frontier_tiles - opp_frontier_tiles)

      # Corners Capture + Horizontal Stability (+ Edge Stability)
      my_corners = 0.0
      opp_corners = 0.0
      corners = [[0,0],[0,7],[7,7],[7,0]]
      directions = [[0,1], [1,0], [0,-1], [-1,0],[0,1]]

      idx = 0
      my_semi_stable_sq = set()
      opp_semi_stable_sq = set()
      for corner in corners:
        if board[corner[0]][corner[1]] == self.color:
          my_corners += 1
          edge_squares = []
          sq = (corner[0], corner[1])
          while(self.validPos((sq[0],sq[1])) and board[sq[0]][sq[1]] == self.color):
            sq = (sq[0]+directions[idx][0],sq[1]+directions[idx][1])
            edge_squares.append(sq)
          height = 8
          for sq in edge_squares:
            idx_2 = 0
            while(self.validPos((sq[0],sq[1])) and board[sq[0]][sq[1]] == self.color):
              idx_2 += 1
              sq = (sq[0] + directions[idx+1][0], sq[1] + directions[idx+1][1])
              my_semi_stable_sq.add(sq)
            height = min(height, idx_2)

        elif board[corner[0]][corner[1]] == oppColor:
          opp_corners += 1
          edge_squares = []
          sq = (corner[0], corner[1])
          while(self.validPos((sq[0],sq[1])) and board[sq[0]][sq[1]] == oppColor):
            sq = (sq[0]+directions[idx][0],sq[1]+directions[idx][1])
            edge_squares.append(sq)
          height = 8
          for sq in edge_squares:
            idx_2 = 0
            while(self.validPos((sq[0],sq[1])) and board[sq[0]][sq[1]] == oppColor):
              idx_2 += 1
              sq = (sq[0] + directions[idx+1][0], sq[1] + directions[idx+1][1])
              opp_semi_stable_sq.add(sq)
            height = min(height, idx_2)
        idx += 1
      
      corners_score = 0.0
      corners_score += weights["corners_score"]* (my_corners - opp_corners)

      horizontal_stability_score = 0.0
      horizontal_stability_score += weights["horizontal_stability_score"] * (len(my_semi_stable_sq) - len(opp_semi_stable_sq))
      
      # Diagonal Stability
      my_diagonal_sq = set()
      opp_diagonal_sq = set()
      #   diagonal 1 - own
      starters = ((7,0), (6,0), (5,0), (4,0), (3,0), (2,0), (1,0), (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7))
      for starter in starters:
        isFilled = True
        i,j = starter
        while self.validPos((i,j)):
          if board[i][j] != self.color:
            isFilled = False
            break
          i += 1
          j += 1
        if isFilled:
          i,j = starter
          while self.validPos((i,j)):
            my_diagonal_sq.add((i,j))
            i+=1
            j+=1

      #   diagonal 1 - opp
      for starter in starters:
        isFilled = True
        i,j = starter
        while self.validPos((i,j)):
          if board[i][j] != oppColor:
            isFilled = False
            break
          i += 1
          j += 1
        if isFilled:
          i,j = starter
          while self.validPos((i,j)):
            opp_diagonal_sq.add((i,j))
            i+=1
            j+=1

      #   diagonal 2 - own
      starters = ((7,0), (6,0), (5,0), (4,0), (3,0), (2,0), (1,0), (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7))
      for starter in starters:
        isFilled = True
        i,j = starter
        while self.validPos((i,j)):
          if board[i][j] != self.color:
            isFilled = False
            break
          i -= 1
          j += 1
        if isFilled:
          i,j = starter
          while self.validPos((i,j)):
            my_diagonal_sq.add((i,j))
            i-=1
            j+=1

      #   diagonal 2 - own
      starters = ((7,0), (6,0), (5,0), (4,0), (3,0), (2,0), (1,0), (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7))
      for starter in starters:
        isFilled = True
        i,j = starter
        while self.validPos((i,j)):
          if board[i][j] != oppColor:
            isFilled = False
            break
          i -= 1
          j += 1
        if isFilled:
          i,j = starter
          while self.validPos((i,j)):
            opp_diagonal_sq.add((i,j))
            i-=1
            j+=1

      diagonal_stability_score = 0.0
      diagonal_stability_score += weights["diagonal_stability_score"]  * (len(my_diagonal_sq) - len(opp_diagonal_sq))


      edges = [board[0][2:6], board[7][2:6], [col[0] for col in board[2:6]], [col[7] for col in board[2:6]]]
      white_partial_edges = sum(edge.count('W') == 4 for edge in edges)
      black_partial_edges = sum(edge.count('B') == 4 for edge in edges)

      partial_edge_score = 0.0
      if (white_partial_edges + black_partial_edges) > 0:
        if self.color == "W":
          partial_edge_score = weights["partial_edge_score"] * (white_partial_edges - black_partial_edges)
        else:
          partial_edge_score = weights["partial_edge_score"] * (black_partial_edges - white_partial_edges)
    
      # Corner Closeness
      corner_closeness_score = 0.0
      my_tiles = opp_tiles = 0
      
      if board[0][0] == "G":
          if board[0][1] == self.color: my_tiles += 1
          elif board[0][1] == oppColor: opp_tiles += 1
          if board[1][1] == self.color: my_tiles += 1
          elif board[1][1] == oppColor: opp_tiles += 1
          if board[1][0] == self.color: my_tiles += 1
          elif board[1][0] == oppColor: opp_tiles += 1

      if board[0][7] == "G":
          if board[0][6] == self.color: my_tiles += 1
          elif board[0][6] == oppColor: opp_tiles += 1
          if board[1][6] == self.color: my_tiles += 1
          elif board[1][6] == oppColor: opp_tiles += 1
          if board[1][7] == self.color: my_tiles += 1
          elif board[1][7] == oppColor: opp_tiles += 1

      if board[7][0] == "G":
          if board[6][0] == self.color: my_tiles += 1
          elif board[6][0] == oppColor: opp_tiles += 1
          if board[7][1] == self.color: my_tiles += 1
          elif board[7][1] == oppColor: opp_tiles += 1
          if board[6][1] == self.color: my_tiles += 1
          elif board[6][1] == oppColor: opp_tiles += 1

      if board[7][7] == "G":
          if board[6][7] == self.color: my_tiles += 1
          elif board[6][7] == oppColor: opp_tiles += 1
          if board[6][6] == self.color: my_tiles += 1
          elif board[6][6] == oppColor: opp_tiles += 1
          if board[7][6] == self.color: my_tiles += 1
          elif board[7][6] == oppColor: opp_tiles += 1

      corner_closeness_score += weights["corner_closeness_score"] * -(my_tiles - opp_tiles) 

      score = b_w_score + d_score + frontier_score + mobility_score + corners_score + diagonal_stability_score + horizontal_stability_score + corner_closeness_score + partial_edge_score
      
      # # Print board for score evaluation
      # print 
      # print "Current Board: "
      # for row in board:
      #     print '      ', ' '.join(row).replace('B', 'X').replace('W', 'O').replace('G', '.')
      # print
      # print "Current Player", self.color, self.color.replace('B', 'X').replace('W', 'O')
      # print "Opponent Player", oppColor, oppColor.replace('B', 'X').replace('W', 'O')
      # print "BW Score: ", b_w_score
      # print "Weighted Square: ",d_score
      # print "Frontier: ", frontier_score
      # print "Mobility:", mobility_score
      # print "Corners Captire: ", corners_score
      # print "Diagonal Stability: ", diagonal_stability_score
      # print "Horizontal Stability: ", horizontal_stability_score
      # print "Corner Closeness: ", corner_closeness_score
      # print "Partial Edge: ", partial_edge_score
      # print "Total score: ", score
      # print "##########################################"
      # print

      
      return score


  def no_of_corners_occupied(self, board):
      corners = [[0,0],[0,7],[7,7],[7,0]]
      b = 0
      w = 0
      for corner in corners:
        if board[corner[0]][corner[1]] == "B":
          b += 1
        elif board[corner[0]][corner[1]] == "W":
          w += 1

      return max(b,w)

  def disc_present_on_edges(self,board):
    for i in xrange(8):
      if (board[i][0] != "G" or board[i][7] != "G" or board[0][i] != "G" or board[7][i] != "G" ): return True
    return False

  def oppositeColor(self, color):
      if color == 'W': return 'B'
      if color == 'B': return 'W'
      assert False, 'Color is neither W or B'

  def computeScore(self, board):
      w = b = 0
      for i in range(constants.BRD_SIZE):
          for j in range(constants.BRD_SIZE):
              color = board[i][j]
              if   color == 'W': w += 1
              elif color == 'B': b += 1
      return w, b

  # invariant list = toMoveList(toMoveBoard(list))
  def toMoveList(self,moveboard):
      moves = []
      for i in range(8):
        for j in range(8):
          b = moveboard & 1
          if b == 1: moves.append((7-i,7-j))
          moveboard >>= 1
      moves.reverse()
      return moves # we dont want to change the heursitics

  def toMoveBoard(self,moves):
      moveBoard = 0
      for i in range(8):
        for j in range(8):
          moveBoard <<= 1
          if (i,j) in moves: moveBoard+=1
      moveBoard
      return moveBoard

  def findAllMovesHelper(self, board, color, oppColor, bitboard, checkHasMoveOnly=False):
      moves = []
      for i in xrange(constants.BRD_SIZE):
          for j in xrange(constants.BRD_SIZE):
              if board[i][j] != 'G': continue
              for ddir in constants.DIRECTIONS:
                  if self.validMove(board, (i,j), ddir, color, oppColor):
                      moves.append((i,j))
                      if checkHasMoveOnly: return moves
                      break
      return moves


  def validMove(self, board, pos, ddir, color, oppColor):
      def addToPos(pos,ddir):
          return (pos[0]+ddir[0], pos[1]+ddir[1])

      def validPos(pos):
          return pos[0] >= 0 and pos[0] < constants.BRD_SIZE and \
                 pos[1] >= 0 and pos[1] < constants.BRD_SIZE

      newPos = addToPos(pos, ddir)
      if not validPos(newPos):                   return False
      if board[newPos[0]][newPos[1]] != oppColor: return False

      while board[newPos[0]][newPos[1]] == oppColor:
          newPos = addToPos(newPos, ddir)
          if not validPos(newPos): break

      if validPos(newPos) and board[newPos[0]][newPos[1]] == color:
          return True
      return False

  def validPos(self,pos):
      return pos[0] >= 0 and pos[0] < constants.BRD_SIZE and \
             pos[1] >= 0 and pos[1] < constants.BRD_SIZE


  def addToPos(self,pos,ddir):
          return (pos[0]+ddir[0], pos[1]+ddir[1])

  def makeMove(self,board, pos, color):
      board[pos[0]][pos[1]] = color
      oppColor = self.oppositeColor(color)

      for ddir in constants.DIRECTIONS:
          if self.validMove(board, pos, ddir, color, oppColor):
              newPos = self.addToPos(pos, ddir)
              while board[newPos[0]][newPos[1]] == oppColor:
                  board[newPos[0]][newPos[1]] = color
                  newPos = self.addToPos(newPos, ddir)

  def alphabeta(self,board, depth, alpha, beta, maximizingPlayer, skip=False):
    # check if we've seen this board before
    self.calls += 1
    bitboard = self.toBitBoard(board)
  
    if maximizingPlayer:
      currentColor = self.color
    else:
      currentColor = self.oppositeColor(self.color)

    moves = self.findAllMovesHelper(board, currentColor, self.oppositeColor(currentColor), bitboard)

    if sum(self.computeScore(board)) == 64:
        return (self.evaluate(board,bitboard,True),None)
    if depth == 0:
        v = self.evaluate(board, bitboard)
        # self.memoize(bitboard, depth, v , None)
        return (v, None)

    if len(moves) == 0:
      maximizingPlayer = not maximizingPlayer
      if skip == True:
        return (self.evaluate(board,bitboard, True), None)
      else:
        return self.alphabeta(board, depth-1, alpha, beta, maximizingPlayer, True)



    bestMove = None
    printMoves = {}
    if maximizingPlayer:
        v = -constants.INFINITY
        for move in moves:
            boardCopy = copy.deepcopy(board)
            # make move
            # board[move[0]][move[1]] = currentColor
            self.makeMove(boardCopy, move, self.color)
            score = self.alphabeta(boardCopy, depth -1, alpha, beta, False)
            
            if depth == self.depth:
              printMoves[move] = score[0]

            if score[0] > v:
              bestMove = move
              v = score[0]

            alpha = max(alpha, v)
            if beta <= alpha:
                break
    else:
        v = constants.INFINITY
        for move in moves:
            boardCopy = copy.deepcopy(board)
            #make move
            self.makeMove(boardCopy, move, self.oppositeColor(self.color))
            score = self.alphabeta(boardCopy, depth -1, alpha, beta, True)
            if score[0] < v:
              bestMove = move
              v = score[0]

            beta = min(beta, v)
            if beta <= alpha:
                break


    if depth == self.depth:
      print printMoves
      print "Current Board: "
      for row in board:
          print '      ', ' '.join(row).replace('B', 'X').replace('W', 'O').replace('G', '.')
      print
      print
      print
      
    return (v, bestMove)