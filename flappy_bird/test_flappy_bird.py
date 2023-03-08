from ple.games.flappybird import FlappyBird

class TestFlappyBird(FlappyBird):
    def _init(self, pipe1_h, pipe2_h, dist, vel):
        super().init()
        pipe1_h = int(pipe1_h)
        pipe2_h = int(pipe2_h)
        dist = int(dist)
        vel = int(vel)
        self.player.vel = vel
        pipe3_h = self.rng.random_integers(self.pipe_min,self.pipe_max)
        pipe_hight = [pipe1_h, pipe2_h, pipe3_h]
        for i, p in enumerate(self.pipe_group):
            p.init(pipe_hight[i], self.pipe_gap, self.pipe_offsets[i]+dist, self.pipe_color)
        
        