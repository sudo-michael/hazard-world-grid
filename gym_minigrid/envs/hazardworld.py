from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.roomgrid import reject_next_to

import itertools as itt, random
import numpy as np

from gym import spaces

################################################################################
# Heleper functions
################################################################################


def reject_dist_3(env, pos):
    """
    Function to filter out object positions that are 3 away from
    the agent's starting positon
    """

    sx, sy = env.agent_pos
    x, y = pos
    d = abs(sx - x) + abs(sy - y)
    return d <= 3


def too_close(pos, entities, min_dist):
    """
    Function to filter out object positions that are right next to
    the agent's starting point
    """
    for entity_pos in entities:
        sx, sy = entity_pos
        x, y = pos
        d = abs(sx - x) + abs(sy - y)
        if d <= min_dist:
            return True
    return False


################################################################################
# Synthetic instruction generation
################################################################################

AVOID_OBJ = {0: "lava", 1: "grass", 2: "water"}

AVOID_OBJ_VALS = list(AVOID_OBJ.values())

IDX_TO_AVOID = {"lava": 9, "grass": 11, "water": 12}

NEG = ["do not", "don't", "never"]
VNOP = ["cross", "touch"]
VPROP = ["move", "go", "travel", "pass", "walk"]
VNOP_GERUND = ["crossing", "touching"]
VPROP_GERUND = ["moving", "going", "traveling", "passing", "walking"]
PROP = ["through", "on", "upon"]
VGOAL = ["go to", "reach", "move to"]
NUM = {1: "once", 2: "twice", 3: "three times", 4: "four times", 5: "five times"}


def constraint1(avoid, nu, avoid_idx):
    ne = random.choice(NEG)
    vn = random.choice(VNOP)
    return f"{ne} {vn} {avoid} more than {nu}"


def constraint2(avoid, nu, avoid_idx):
    ne = random.choice(NEG)
    vp = random.choice(VPROP)
    pr = random.choice(PROP)
    return f"{ne} {vp} {pr} {avoid} more than {nu}"


def constraint3(avoid, nu, avoid_idx):
    vn = random.choice(VNOP)
    return f"{vn} {avoid} less than {nu}"


def constraint4(avoid, nu, avoid_idx):
    vp = random.choice(VPROP)
    pr = random.choice(PROP)
    return f"{vp} {pr} {avoid} less than {nu}"


CONSTRAINTS = {0: constraint1, 1: constraint2, 2: constraint3, 3: constraint4}


def make_budgetary_constraint(avoid_obj, hc):
    avoid_idx = IDX_TO_AVOID[avoid_obj]
    if hc == 1:
        ne = random.choice(NEG)
        vn = random.choice(VNOP)
        return f"{ne} {vn} {avoid_obj}"
    opt = np.random.choice(4)
    nu = NUM[hc - 1]
    return CONSTRAINTS[opt](avoid_obj, nu, avoid_idx)


def make_sequential_constraint(first_obj, avoid_obj, isBefore):
    if isBefore:
        return f"Before walking on {first_obj} do not walk on {avoid_obj}"
    return f"After walking on {first_obj} do not walk on {avoid_obj}"


def make_relational_constraint(avoid_obj, dist):
    return f"Stay {dist} steps away from {avoid_obj}"


################################################################################
# HazardWorld base
################################################################################


class HazardWorldBase(MiniGridEnv):
    """
    Pick up 3 objects while avoiding many potential hazards.
    Potential hazards are specified by the constraint stored in the mission
    field. The base HazardWorld environment contains budgetary constraints.
    """

    def __init__(self, size=13, seed=None):
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            see_through_walls=True,
            seed=None,
        )
        self.action_space = spaces.Discrete(4)

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space_img = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 2),
            dtype="uint8",
        )

        # TODO change text length
        self.observation_space_text = spaces.Box(
            low=0, high=40, shape=(10,), dtype="uint8"
        )

        self.observation_space_hc = spaces.Box(
            low=0, high=255, shape=(1,), dtype="int64"
        )

        self.observation_space_violations = spaces.Box(
            low=0, high=255, shape=(1,), dtype="int64"
        )

        self.observation_space = spaces.Dict(
            {
                "image": self.observation_space_img,
                "mission": self.observation_space_text,
                "violations": self.observation_space_violations,
                "hc": self.observation_space_hc,
            }
        )

    def isEmpty(self, i, j):
        return self.grid.get(i, j) == None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        obs, reward, _, info = super().step(action)
        done = False

        if action == self.actions.pickup:
            if self.carrying != None:
                if self.carrying.type == "ball":
                    reward += self._reward()
                    self.objs_collected += 1
                elif self.carrying.type == "box":
                    reward += self._reward() * 2
                    self.objs_collected += 1
                elif self.carrying.type == "key":
                    reward += self._reward() * 3
                    self.objs_collected += 1
                self.carrying = None
        curr_cell = self.grid.get(*self.agent_pos)
        if curr_cell != None and curr_cell.type == self.avoid_obj:
            self.violations += 1

        if self.objs_collected == 3:
            done = True
        if self.step_count >= self.max_steps:
            done = True
        obs = self.gen_obs()
        obs["violations"] = np.array([self.violations])
        obs["hc"] = np.array([self.hc - 1])

        return obs, reward, done, {}

    def reset(self):
        obs = super().reset()
        obs["violations"] = np.array([self.violations]).astype(np.int64)
        obs["hc"] = np.array([self.hc - 1]).astype(np.int64)
        return obs

    def _gen_grid(self, width, height, sparsity=0.25):
        assert width % 2 == 1 and height % 2 == 1
        # HazardWorld grid size must be odd
        self.grid = Grid(width, height)

        # pick a cost entity
        self.avoid_obj = random.choice(AVOID_OBJ_VALS)

        self.objs_collected = 0
        self.hc = random.randint(1, 6)
        self.violations = 0

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # The child class must place reward entities


################################################################################
# Budgetary Constraints
################################################################################


class HazardWorldBudgetary(HazardWorldBase):
    def _gen_grid(self, width, height, sparsity=0.5):
        super()._gen_grid(width, height, sparsity)
        # add obstacles
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if random.random() < sparsity and self.isEmpty(i, j):
                    val = random.random()
                    if val < 0.33:
                        self.put_obj(Lava(), i, j)
                    elif val < 0.66:
                        self.put_obj(Water(), i, j)
                    else:
                        self.put_obj(Grass(), i, j)

        self.place_agent()

        # add 3 reward entities
        self.place_obj(Ball("red"), reject_fn=reject_next_to)
        self.place_obj(Box("yellow"), reject_fn=reject_next_to)
        self.place_obj(Key("blue"), reject_fn=reject_next_to)

        self.mission = make_budgetary_constraint(self.avoid_obj, self.hc)


register(
    id="MiniGrid-HazardWorld-B-v0", entry_point="gym_minigrid.envs:HazardWorldBudgetary"
)

################################################################################
# Sequential Constraints
################################################################################


class HazardWorldSequential(HazardWorldBase):
    def _gen_grid(self, width, height, sparsity=0.5):
        super()._gen_grid(width, height, sparsity)
        self.isBefore = random.choice([True, False])
        rand = random.randint(0, 2)

        if self.isBefore:
            # first_obj here deactivates avoid_obj
            self.first_obj, self.avoid_obj = random.sample(AVOID_OBJ_VALS, 2)
            self.mission = make_sequential_constraint(
                self.first_obj, self.avoid_obj, self.isBefore
            )
            self.hc = 1
        else:
            # avoid object is chosen for free in super class
            self.first_obj = random.choice(AVOID_OBJ_VALS)
            self.mission = make_sequential_constraint(
                self.first_obj, self.avoid_obj, self.isBefore
            )
            self.second_obj = self.avoid_obj
            self.avoid_obj = None
            self.hc = 42  # arbitrary

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if random.random() < sparsity and self.isEmpty(i, j):
                    val = random.random()
                    if val < 0.33:
                        self.put_obj(Lava(), i, j)
                    elif val < 0.66:
                        self.put_obj(Water(), i, j)
                    else:
                        self.put_obj(Grass(), i, j)

        self.place_agent()

        # add 3 reward entities
        self.place_obj(Ball("red"), reject_fn=reject_next_to)
        self.place_obj(Box("yellow"), reject_fn=reject_next_to)
        self.place_obj(Key("blue"), reject_fn=reject_next_to)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        curr_cell = self.grid.get(*self.agent_pos)

        # handle conditions
        if curr_cell != None:
            if self.isBefore and curr_cell == first_obj:
                self.avoid_obj == None
            if not self.isBefore and self.avoid_obj == None:
                # ignore constraint violations if after condition has not been met
                self.violations = 0
                if curr_cell.type == self.first_obj:
                    self.avoid_obj = self.second_obj
                    self.hc = 1
        return obs, reward, done, info


register(
    id="MiniGrid-HazardWorld-S-v0",
    entry_point="gym_minigrid.envs:HazardWorldSequential",
)

################################################################################
# Relational Constraints
################################################################################


class HazardWorldRelational(HazardWorldBase):
    def safe_put(self, obj, i, j, entities):
        """
        Put an object at a specific position in the grid
        """
        # Don't place the object on top of another object
        if self.grid.get(i, j) != None:
            return
        # Don't place the object where the agent is
        if np.array_equal([i, j], self.agent_pos):
            return
        # Check if there is a filtering criterion
        if too_close((i, j), entities, self.min_dist):
            return

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def _gen_grid(self, width, height, sparsity=0.5):
        super()._gen_grid(width, height, sparsity)

        self.min_dist = random.randint(0, 3)
        self.hc = 1

        entities = []
        entities.append(self.place_agent())
        entities.append(self.place_obj(Ball("red"), reject_fn=reject_dist_3))
        entities.append(self.place_obj(Box("yellow"), reject_fn=reject_dist_3))
        entities.append(self.place_obj(Key("blue"), reject_fn=reject_dist_3))

        # add obstacles
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if random.random() < sparsity and self.isEmpty(i, j):
                    val = random.random()
                    if val < 0.33:
                        self.safe_put(Lava(), i, j, entities)
                    elif val < 0.66:
                        self.safe_put(Water(), i, j, entities)
                    else:
                        self.safe_put(Grass(), i, j, entities)

        self.mission = make_relational_constraint(self.avoid_obj, self.min_dist)


register(
    id="MiniGrid-HazardWorld-R-v0",
    entry_point="gym_minigrid.envs:HazardWorldRelational",
)
