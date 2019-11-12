import pygame
import neat
import os
import random
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800
gen_count = 0

BIRDS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
         pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
         pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png"))),
         pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png")))]

PIPE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
VEL = 5

STAT_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMGS = BIRDS
    MAX_ROT = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0

    def move(self):
        self.tick_count += 1

        displacement = self.vel*self.tick_count + 1.5*self.tick_count**2

        # limit up speed to 16
        if displacement >= 16:
            displacement = 16

        # free fall acceleration
        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0:
            if self.tilt < self.MAX_ROT:
                self.tilt = self.MAX_ROT
            else:
                if self.tilt > -90:
                    self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        # animation cycle
        self.img = self.IMGS[self.img_count % 4]

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 180

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE, False, True)
        self.PIPE_BOTTOM = PIPE

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(40, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, int(self.top - round(bird.y)))
        bottom_offset = (self.x - bird.x, int(self.bottom - round(bird.y)))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True

        return False


class Base:
    WIDTH = BASE.get_width()
    IMG = BASE

    def __init__(self, y):
        self.y = y
        self.first = 0
        self.second = self.WIDTH

    def move(self):
        self.first -= VEL
        self.second -= VEL

        if self.first + self.WIDTH < 0:
            self.first = self.second + self.WIDTH

        if self.second + self.WIDTH < 0:
            self.second = self.first + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.first, self.y))
        win.blit(self.IMG, (self.second, self.y))


def draw_window(win, birds, pipes, base, score, alive, speed):
    win.blit(BG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    text = STAT_FONT.render("Generation: " + str(gen_count), 1, (255, 255, 255))
    win.blit(text, (10, 10))
    text = STAT_FONT.render("Alive: " + str(alive), 1, (255, 255, 255))
    win.blit(text, (10, 50))
    text = STAT_FONT.render("FPS: " + str(speed), 1, (255, 0, 0))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), WIN_HEIGHT - 100))

    base.draw(win)
    for bird in birds:
        bird.draw(win)

    pygame.display.update()


def main(genomes, config):
    global gen_count
    gen_count += 1
    speed = 30

    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(600)]
    score = 0

    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

    clock = pygame.time.Clock()

    while True:
        clock.tick(speed)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                quit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_m and speed < 240:
                    speed = speed*2
                if e.key == pygame.K_k:
                    birds = []

        pipe_index = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_index = 1
        else:
            break
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y,
                                       abs(bird.y - pipes[pipe_index].height),
                                       abs(bird.y - pipes[pipe_index].bottom)))

            if output[0] > 0.7:
                bird.jump()

        add_pipe = False
        removed = []
        base.move()
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_BOTTOM.get_width() < 0:
                removed.append(pipe)
            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(600))

        for p in removed:
            pipes.remove(p)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        draw_window(win, birds, pipes, base, score, len(birds), speed)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
