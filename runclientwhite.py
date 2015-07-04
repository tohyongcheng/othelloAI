import client, randomplayer, smartplayer

#replace randomPlayer.RandomPlayer with your player
#make sure to specify the color of the player to be 'W'
whitePlayer = randomplayer.RandomPlayer('W')

whiteClient = client.Client(whitePlayer)
whiteClient.run()
