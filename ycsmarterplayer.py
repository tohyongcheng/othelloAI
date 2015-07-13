import random, memory, constants

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

class YcSmarterPlayer:

  def __init__(self,color):
      self.color = color
      self.table = {}
      self.savedMoves = {}

      #### profiling
      self.calls = 0.0
      self.hits = 0.0

  def memoize(self, bitboard, depth, v, bestMove):
      self.table[(bitboard, depth)] = (v, bestMove)

  def chooseMove(self,board,prevMove):
      memUsedMB = memory.getMemoryUsedMB()
      if memUsedMB > constants.MEMORY_LIMIT_MB - 10: #If I am close to memory limit
          #don't allocate memory, limit search depth, etc.
          # we just flush the table
          print 'flushing'
          self.table = {}
          self.savedMoves = {}

      color = self.color
      if   color == 'W': oppColor = 'B'
      elif color == 'B': oppColor = 'W'
      else: assert False, 'ERROR: Current player is not W or B!'

      result = self.alphabeta(board, 6, -constants.INFINITY, constants.INFINITY, True)
      print "Mb Used:", memory.getMemoryUsedMB()
      print "Move found, score:", result[0]
      print "Hit Percentage:", self.hits / self.calls * 100
      return result[1]


  def gameEnd(self,board):
      return

  def getColor(self):
      return self.color

  def getMemoryUsedMB(self):
      return 0.0

  def toBitBoard(self,board):
      #helper function that turns a board to a bitboard
      whiteBoard = 0
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

  def evaluate(self,board, bitboard):
      score = 0.0
      oppColor = self.oppositeColor(self.color)

      # Feature Weights on squares dependent on game stage
      d = 0
      currentStageWeights = openingWeights
      if self.no_of_corners_occupied(board) >= 2:
        currentStageWeights = endWeights
      elif self.disc_present_on_edges(board):
        currentStageWeights = middleWeights

      for i in xrange(8):
        for j in xrange(8):
          if board[i][j] == self.color:
            d += currentStageWeights[i][j]
          elif board[i][j] == oppColor:
            d -= currentStageWeights[i][j]
      # print("D:", d)
      score += 300.0 * d

      # Coin Parity
      b,w = self.computeScore(board)
      if self.color == "W":
        score += 100.0 * (w-b) / (b+w)
      else:
        score += 100.0 * (b-w) / (b+w)

      # Mobility
      my_moves = len(self.findAllMovesHelper(board, self.color, oppColor, bitboard))
      opp_moves = len(self.findAllMovesHelper(board, oppColor, self.color, bitboard))
      if (my_moves + opp_moves) != 0:
        score += 300.0 * (my_moves - opp_moves) / (my_moves + opp_moves)

      # Corners Capture
      my_corners = 0
      opp_corners = 0
      corners = [[0,0],[0,7],[7,7],[7,0]]
      for corner in corners:
        if board[corner[0]][corner[1]] == self.color:
          my_corners += 1
        elif board[corner[0]][corner[1]] == oppColor:
          opp_corners += 1
      if (my_corners + opp_corners) > 0:
        score += 300.0 * (my_corners - opp_corners) / (my_corners + opp_corners)


      # Corner Closeness
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
      if (my_tiles + opp_tiles) > 0:
        score += (-200) * (my_tiles - opp_tiles) / (my_tiles + opp_tiles)

      # print("Score:", score)


      # Pattern Recognition

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

  def findAllMovesHelper(self, board, color, oppColor, bitboard, checkHasMoveOnly=False):
      #code.interact(local=locals())
      if (bitboard,color) in self.savedMoves:
        return self.savedMoves[(bitboard,color)]

      moves = []
      for i in xrange(constants.BRD_SIZE):
          for j in xrange(constants.BRD_SIZE):
              if board[i][j] != 'G': continue
              for ddir in constants.DIRECTIONS:
                  if self.validMove(board, (i,j), ddir, color, oppColor):
                      moves.append((i,j))
                      if checkHasMoveOnly: return moves
                      break
      self.savedMoves[(bitboard,color)] = moves
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

    
  def alphabeta(self,board, depth, alpha, beta, maximizingPlayer):
    # check if we've seen this board before
    self.calls += 1
    bitboard = self.toBitBoard(board)
    if (bitboard, depth) in self.table:
      self.hits += 1
      return self.table[(bitboard,depth)]

    if maximizingPlayer:
      currentColor = self.color
    else:
      currentColor = self.oppositeColor(self.color)

    moves = self.findAllMovesHelper(board, currentColor, self.oppositeColor(currentColor), bitboard)

    if depth == 0 or len(moves) == 0:
        v = self.evaluate(board, bitboard)
        self.memoize(bitboard, depth, v , None)
        return (v, None)

    bestMove = None
    if maximizingPlayer:
        v = -constants.INFINITY
        for move in moves:
            board[move[0]][move[1]] = currentColor
            score = self.alphabeta(board, depth -1, alpha, beta, False)
            if score[0] > v:
              bestMove = move
              v = score[0]

            alpha = max(alpha, v)
            board[move[0]][move[1]] = 'G'
            if beta <= alpha:
                break
    else:
        v = constants.INFINITY
        for move in moves:
            board[move[0]][move[1]] = currentColor
            score = self.alphabeta(board, depth -1, alpha, beta, True)
            if score[0] < v:
              bestMove = move
              v = score[0]

            beta = min(beta, v)
            board[move[0]][move[1]] = 'G'
            if beta <= alpha:
                break

    self.memoize(bitboard,depth, v, bestMove)
    return (v, bestMove, moves)