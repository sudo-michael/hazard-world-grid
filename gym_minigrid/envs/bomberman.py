from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.roomgrid import reject_next_to

import itertools as itt, random

# TODO: reverse mapping is potentially confusing here. Redo?
AVOID_OBJ = {
    0: 'lava',
    1: 'grass',
    2: 'water'
}

# from encoding in minigrid.py
IDX_TO_AVOID = {
    'lava': 9,
    'grass': 11,
    'water': 12
}

NEG = ['do not', 'don\'t', 'never']
VNOP = ['cross', 'touch']
VPROP = ['move', 'go', 'travel', 'pass', 'walk']
VNOP_GERUND = ['crossing', 'touching']
VPROP_GERUND = ['moving', 'going', 'traveling', 'passing', 'walking']
PROP = ['through', 'on', 'upon']
VGOAL = ['go to', 'reach', 'move to']
NUM = {
    1: 'once',
    2: 'twice',
    3: 'three times'
    # 4: 'four times',
    # 5: 'five times'
}


def constraint1(avoid, hc, nu, avoid_idx):
    ne = random.choice(NEG)
    vn = random.choice(VNOP)
    return f'{ne} {vn} {avoid} more than {nu}, {hc}, {avoid_idx}'
    

def constraint2(avoid, hc, nu, avoid_idx):
    ne = random.choice(NEG)
    vp = random.choice(VPROP)
    pr = random.choice(PROP)
    return f'{ne} {vp} {pr} {avoid} more than {nu}, {hc}, {avoid_idx}'


def constraint3(avoid, hc, nu, avoid_idx):
    vn = random.choice(VNOP)
    return f'{vn} {avoid} less than {nu}, {hc}, {avoid_idx}'


def constraint4(avoid, hc, nu, avoid_idx):
    vp = random.choice(VPROP)
    pr = random.choice(PROP)
    return f'{vp} {pr} {avoid} less than {nu}, {hc}, {avoid_idx}'


CONSTRAINTS = {
    0: constraint1,
    1: constraint2,
    2: constraint3,
    3: constraint4
}

def make_mission(avoid_obj, hc):
    avoid_idx = IDX_TO_AVOID[avoid_obj]
    if hc == 1:
        ne = random.choice(NEG)
        vn = random.choice(VNOP)
        return f'{ne} {vn} {avoid_obj}, 0, {avoid_idx}'
    opt = np.random.choice(4)
    nu = NUM[hc-1]
    return CONSTRAINTS[opt](avoid_obj, hc-1, nu, avoid_idx)
    
class BombermanEnv(MiniGridEnv):
    """
    Environment inspired by Bomberman. Exploring 'ruins' while picking
    up a series of rewards.
    """

    def __init__(self, hc=3, size=13, seed=None):
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            see_through_walls=True,
            seed=None,
        )

    def isEmpty(self, i, j):
        return self.grid.get(i, j) == None

    def step(self, action):
        obs, reward, _, info = super().step(action)
        done = False

        if action == self.actions.pickup:
            if self.carrying != None:
                if self.carrying.type == 'ball':
                    reward += self._reward()
                    self.objs_collected += 1 
                elif self.carrying.type == 'box':
                    reward += self._reward() * 2
                    self.objs_collected += 1 
                elif self.carrying.type == 'key':
                    reward += self._reward() * 3
                    self.objs_collected += 1
                self.carrying = None
        curr_cell = self.grid.get(*self.agent_pos)
        if curr_cell != None and curr_cell.type == self.avoid_obj:
            # print(curr_cell.type)
            self.violations += 1

        if self.objs_collected == 3:
            done = True 
            # print('d1')
        if self.violations == self.hc:
            done = True
            # print('d2')
        if self.step_count >= self.max_steps:
            done = True 
            # print('d3')
        obs = self.gen_obs() 

        # print(obs['image'])

        return obs, reward, done, {}
    
    def _gen_grid(self, width, height, wall_sparsity=0.1, sparsity=0.3):
        assert width % 2 == 1 and height % 2 == 1
        # bomberman grid must be odd
        self.grid = Grid(width, height)

        # pick a constrained object
        rand = random.randint(0, 2) 
        self.avoid_obj = AVOID_OBJ[rand]

        self.objs_collected = 0
        self.hc = random.randint(1, 4)
        self.violations = 0

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # add bomberman walls
        for i in range(2, height-2, 2):
            for j in range(2, width-2, 2):
                self.put_obj(Wall(), i, j)
        
        for i in range(2, height-2, 2):
            for j in range(3, width-2, 2):
                if random.random() < wall_sparsity:
                    self.put_obj(Wall(), i, j)

        for i in range(3, height-2, 2):
            for j in range(2, width-2, 2):
                if random.random() < wall_sparsity:
                    self.put_obj(Wall(), i, j)

        # add lava
        for i in range(1, height-1):
            for j in range(1, width-1):
                if np.array_equal([i, j], self.agent_pos):
                    continue
                elif random.random() < sparsity and self.isEmpty(i, j):
                    val = random.random()
                    if val < 0.33:
                        self.put_obj(Lava(), i, j)
                    elif val < 0.66:
                        self.put_obj(Water(), i, j)
                    else:
                        self.put_obj(Grass(), i, j)
        
        self.place_agent()

        # add 3 objects
        self.place_obj(Ball('red'), reject_fn=reject_next_to)
        self.place_obj(Box('yellow'), reject_fn=reject_next_to)
        self.place_obj(Key('blue'), reject_fn=reject_next_to)
        
        self.mission = make_mission(self.avoid_obj, self.hc)

register(
    id='MiniGrid-Bomberman-v0',
    entry_point='gym_minigrid.envs:BombermanEnv'
)


class BombermanTurkEnv(BombermanEnv):
    def _gen_grid(self):
        # TODO: call decode() to return a new grid. Should be free.
        # take obs and mission from a pickle file to set self.obs, self.mission
        pass

register(
    id='MiniGrid-Bomberman-Turk-v0',
    entry_point='gym_minigrid.envs:BombermanTurkEnv'
)
