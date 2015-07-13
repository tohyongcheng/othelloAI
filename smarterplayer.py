import random, memory, constants, code

class SmarterPlayer:

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
      if memUsedMB > constants.MEMORY_LIMIT_MB - 10000: #If I am close to memory limit
          #don't allocate memory, limit search depth, etc.
          # we just flush the table
          self.table = {}
          self.savedMoves = {}

      color = self.color
      if   color == 'W': oppColor = 'B'
      elif color == 'B': oppColor = 'W'
      else: assert False, 'ERROR: Current player is not W or B!'

      result = self.alphabeta(board, 7, -constants.INFINITY, constants.INFINITY, True)
      print "Mb Used:", getMemoryUsedMB()
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
        score += 100.0 * (my_moves - opp_moves) / (my_moves + opp_moves)

      # Corners Capture
      my_corners = 0
      opp_corners = 0
      corners = [[0,0],[0,7],[7,7],[7,0]]
      for corner in corners:
        if board[corner[0]][corner[1]] == self.color:
          my_corners += 1
        elif board[corner[0]][corner[1]] == oppColor:
          opp_corners += 1
      if (my_corners + opp_corners) != 0:
        score += 300.0 * (my_corners - opp_corners) / (my_corners + opp_corners)

      # Stability not implemneted yet...

      return score

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
      if bitboard in self.savedMoves:
        return self.savedMoves[bitboard]

      moves = []
      for i in xrange(constants.BRD_SIZE):
          for j in xrange(constants.BRD_SIZE):
              if board[i][j] != 'G': continue
              for ddir in constants.DIRECTIONS:
                  if self.validMove(board, (i,j), ddir, color, oppColor):
                      moves.append((i,j))
                      if checkHasMoveOnly: return moves
                      break
      self.savedMoves[bitboard] = moves
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