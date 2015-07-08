import client, randomplayer, smartplayer, ycsmarterplayer

#replace randomplayer.RandomPlayer with your player
#make sure to specify the color of the player to be 'B'
# blackPlayer = randomplayer.RandomPlayer('B')
# blackPlayer = smartplayer.SmartPlayer('B')
blackPlayer = ycsmarterplayer.YcSmarterPlayer('B')

blackClient = client.Client(blackPlayer)
blackClient.run()
