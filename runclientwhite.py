import client, randomplayer, smartplayer, ycsmarterplayer, ycsmartestplayer, ycsmartestplayer_no_memo, nonrandomplayer

#replace randomPlayer.RandomPlayer with your player
#make sure to specify the color of the player to be 'W'
# whitePlayer = randomplayer.RandomPlayer('W')
# whitePlayer = smartplayer.SmartPlayer('W')
# whitePlayer = ycsmarterplayer.YcSmarterPlayer('W')
# whitePlayer = ycsmartestplayer_no_memo.YcSmartestPlayer('W')
# whitePlayer = ycsmartestplayer.YcSmartestPlayer('W')
whitePlayer = nonrandomplayer.NonRandomPlayer('W')

whiteClient = client.Client(whitePlayer)
whiteClient.run()
