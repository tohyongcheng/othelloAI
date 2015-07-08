
      # Stability not implemneted yet...
      stability_score = 0
      stable_corners = [[0,0],[0,7],[7,7],[7,0],[0,1],[1,0],[1,7],[0,6],[7,6],[6,7],[7,1],[6,0]]
      stable_corners_occupied = False

      for c in stable_corners:
        if board[c[0]][c[1]] != 'G'
          stable_corners_occupied = True
          break

      if stable_corners_occupied:
        squares = {}
        for i in xrange(8):
          for j in xrange(8):
            squares[i,j] = 0
        
        # horizontal
        for i in xrange(8):
          isFilled = True
          for j in xrange(8):
            if board[i][j] == "G":
              isFilled = False
              break

          if isFilled:
            for j in xrange(8):
              squares[i,j] += 1

        # vertical
        for i in xrange(8):
          isFilled = True
          for j in xrange(8):
            if board[j][i] == "G":
              isFilled = False
              break

          if isFilled:
            for j in xrange(8):
              squares[j,i] += 1

        # diagonal 1
        starters = ((7,0), (6,0), (5,0), (4,0), (3,0), (2,0), (1,0), (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7))
        for starter in starters:
          isFilled = True
          i,j = starter
          while i < 8 and j < 8:
            if board[i][j] == "G"
            isFilled = False
            break
            i += 1
            j += 1
          i,j = starter
          if isFilled:
            while i<8 and j<8:
              squares[i,j] += 1
              i+=1
              j+=1

        # diagonal 2
        starters = ((7,0), (6,0), (5,0), (4,0), (3,0), (2,0), (1,0), (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7))
        for starter in starters:
          isFilled = True
          i,j = starter
          while i < 8 and j < 8:
            if board[i][j] == "G"
            isFilled = False
            break
            i += 1
            j += 1
          i,j = starter
          if isFilled:
            while i>=0 and j<8:
              squares[i,j] += 1
              i-=1
              j+=1

        # decide qualifying 
        qualified_squares = []
        for square_key in squares:
          if squares[square_key] == 4:
            qualified_squares.append(square_key)

        # check for adjacent ones...
        squares_adjacent_to_stable = {}
        for i in xrange(8):
          for j in xrange(8):
            squares_adjacent_to_stable[i,j] = 0

        for square in qualified_squares:
          i,j = square
          