import random, memory, constants

class SmartPlayer:

  def __init__(self,color):
      self.color = color

  def chooseMove(self,board,prevMove):
      memUsedMB = memory.getMemoryUsedMB()
      if memUsedMB > constants.MEMORY_LIMIT_MB - 100: #If I am close to memory limit
          #don't allocate memory, limit search depth, etc.
          #RandomPlayer uses very memory so it does nothing
          pass
             
      dirs = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))
      color = self.color
      if   color == 'W': oppColor = 'B'
      elif color == 'B': oppColor = 'W'
      else: assert False, 'ERROR: Current player is not W or B!'

      moves = []
      for i in xrange(len(board)):
          for j in xrange(len(board[i])):
              if board[i][j] != 'G': continue #background is green, i.e., empty square
              for ddir in dirs:
                  if self.validMove(board, (i,j), ddir, color, oppColor):
                      moves.append((i,j))
                      break
      if len(moves) == 0: return None #no valid moves
      i = random.randint(0,len(moves)-1) #randomly pick a valid move
      return moves[i]


  def gameEnd(self,board):
      return

  def getColor(self):
      return self.color

  def getMemoryUsedMB(self):
      return 0.0


  def evaluate(self,board):
      score = 0
      oppColor = oppositeColor(color)

      # Coin Parity
      b,w = self.computeScore(board)
      if color == "W":
        score += 100 * (abs(w-b)) / (b+w)
      else:
        score += 100 * (abs(b-w)) / (b+w)

      # Mobility
      my_moves = len(self.findAllMovesHelper(board, color, oppColor))
      opp_moves = len(self.findAllMovesHelper(board, oppColor, color))
      if (my_moves + opp_moves) != 0:
        score += 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

      # Corners Capture
      my_corners = 0
      opp_corners = 0
      corners = [(0,0),[0,7],[7,7],[7,0]]
      for corner in corners:
        if board[corner[0]][corner[1]] == color:
          my_corners += 1
        elif board[corner[0]][corner[1]] == oppColor:
          opp_corners += 1
      if (my_corners + opp_corners) != 0:
        score += 100 * (my_corners - opp_corners) / (my_corners + opp_corners)

      # Stability not implemneted yet...

      return score

  def oppositeColor(color):
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
              for ddir in self.dirs:
                  if self.validMove((i,j), ddir, color, oppColor):
                      moves.append((i,j))
                      if checkHasMoveOnly: return moves
                      break
      return moves
