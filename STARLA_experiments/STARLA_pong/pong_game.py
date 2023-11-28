from ple.games.pong import *

class PingPong(Pong):
    def init(self):
        self.score_counts = {
            "agent": 0.0,
            "cpu": 0.0
        }

        self.score_sum = 0.0
        self.ball = Ball(
            self.ball_radius,
            self.ball_speed_ratio * self.height,
            self.rng,
            (self.width / 2, self.height / 2),
            self.width,
            self.height
        )

        self.agentPlayer = Player(
            self.players_speed_ratio * self.height,
            self.paddle_width,
            self.paddle_height,
            (self.paddle_dist_to_wall, self.height / 2),
            self.width,
            self.height)

        self.cpuPlayer = Player(
            self.cpu_speed_ratio * self.height,
            self.paddle_width,
            self.height,
            (self.width - self.paddle_dist_to_wall, self.height / 2),
            self.width,
            self.height)

        self.players_group = pygame.sprite.Group()
        self.players_group.add(self.agentPlayer)
        self.players_group.add(self.cpuPlayer)

        self.ball_group = pygame.sprite.Group()
        self.ball_group.add(self.ball)