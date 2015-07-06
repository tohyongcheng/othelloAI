import random, memory, constants

class SmartPlayer:

  def __init__(self,color):
      self.color = color
      self.table = {}

  def memoize(self, board, v, bestMove):
      self.table[str(board)] = [v, bestMove]

  def chooseMove(self,board,prevMove):
      memUsedMB = memory.getMemoryUsedMB()
      if memUsedMB > constants.MEMORY_LIMIT_MB - 100: #If I am close to memory limit
          #don't allocate memory, limit search depth, etc.
          self.table = {}
          # we just flush the table
             
      dirs = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))
      color = self.color
      if   color == 'W': oppColor = 'B'
      elif color == 'B': oppColor = 'W'
      else: assert False, 'ERROR: Current player is not W or B!'

      return self.alphabeta(board, 8, -constants.INFINITY, constants.INFINITY, True)[1]


  def gameEnd(self,board):
      return

  def getColor(self):
      return self.color

  def getMemoryUsedMB(self):
      return 0.0


  def evaluate(self,board):
      score = 0.0
      oppColor = self.oppositeColor(self.color)

      # Coin Parity
      b,w = self.computeScore(board)
      if self.color == "W":
        score += 100.0 * (w-b) / (b+w)
      else:
        score += 100.0 * (b-w) / (b+w)

      # Mobility
      my_moves = len(self.findAllMovesHelper(board, self.color, oppColor))
      opp_moves = len(self.findAllMovesHelper(board, oppColor, self.color))
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

  def findAllMovesHelper(self, board, color, oppColor, checkHasMoveOnly=False):
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

    
  def alphabeta(self,board, depth, alpha, beta, maximizingPlayer):
    # check if we've seen this board before
    boardString = str(board)
    if boardString in self.table:
      return self.table[boardString]

    if maximizingPlayer:
      currentColor = self.color
    else:
      currentColor = self.oppositeColor(self.color)

    moves = self.findAllMovesHelper(board, currentColor, self.oppositeColor(currentColor))

    if depth == 0 or len(moves) == 0:
        v = self.evaluate(board)
        self.memoize(board, v , None)
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

    self.memoize(board, v, bestMove)
    return (v, bestMove)
