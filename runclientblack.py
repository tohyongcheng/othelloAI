import client, randomplayer, smartplayer

#replace randomplayer.RandomPlayer with your player
#make sure to specify the color of the player to be 'B'
# blackPlayer = randomplayer.RandomPlayer('B')
blackPlayer = smartplayer.SmartPlayer('B')

blackClient = client.Client(blackPlayer)
blackClient.run()
