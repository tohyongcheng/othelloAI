import client, randomplayer, smartplayer, ycsmarterplayer, ycsmartestplayer

#replace randomPlayer.RandomPlayer with your player
#make sure to specify the color of the player to be 'W'
# whitePlayer = randomplayer.RandomPlayer('W')
# whitePlayer = smartplayer.SmartPlayer('W')
whitePlayer = ycsmarterplayer.YcSmarterPlayer('W')
# whitePlayer = ycsmartestplayer.YcSmartestPlayer('W')

whiteClient = client.Client(whitePlayer)
whiteClient.run()
