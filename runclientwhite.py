import client, randomplayer, smartplayer, ycsmarterplayer

#replace randomPlayer.RandomPlayer with your player
#make sure to specify the color of the player to be 'W'
# whitePlayer = randomplayer.RandomPlayer('W')
# whitePlayer = smartplayer.SmartPlayer('W')
whitePlayer = ycsmarterplayer.YcSmarterPlayer('W')

whiteClient = client.Client(whitePlayer)
whiteClient.run()
