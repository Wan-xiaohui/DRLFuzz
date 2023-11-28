from pong_game import PingPong

class TestPong(PingPong):
    def _init(self, playerY, playerVel, ballX, ballY, ballVerX, ballVerY):
        super().init()
        playerY = int(playerY)
        playerVel = int(playerVel)
        ballX = int(ballX)
        ballY = int(ballY)
        ballVerX = int(ballVerX)
        ballVerY = int(ballVerY)
        self.agentPlayer.pos.y = playerY
        self.agentPlayer.vel.y = playerVel
        self.ball.pos.x = ballX
        self.ball.pos.y = ballY
        self.ball.vel.x = ballVerX
        self.ball.vel.y = ballVerY