from ple.games.catcher import Catcher

class TestCatcher(Catcher):
    def _init(self, vel, playerX, fruitX, fruitY):
        super().init()
        vel = int(vel)
        playerX = int(playerX)
        fruitX = int(fruitX)
        fruitY = int(fruitY)
        self.player.vel = vel
        self.player.rect.center = (
            playerX - self.paddle_width / 2,
            self.height - self.paddle_height - 3)
        self.fruit.rect.center = (fruitX, fruitY)


