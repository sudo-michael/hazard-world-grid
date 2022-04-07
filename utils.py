import numpy as np
import ast

TEXT_LENGTH = 10

STOI = {
    "do": 0,
    "not": 1,
    "don't": 2,
    "never": 3,
    "cross": 4,
    "touch": 5,
    "move": 6,
    "go": 7,
    "travel": 8,
    "pass": 9,
    "walk": 10,
    "crossing": 11,
    "touching": 12,
    "moving": 13,
    "going": 14,
    "traveling": 15,
    "passing": 16,
    "walking": 17,
    "through": 18,
    "on": 19,
    "upon": 20,
    "to": 21,
    "reach": 22,
    "move to": 23,
    "once": 24,
    "twice": 25,
    "three": 26,
    "lava": 27,
    "water": 28,
    "grass": 29,
    "more": 30,
    "less": 31,
    "four": 32,
    "five": 32,
    "times": 33,
    "than": 34,
}

OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
    "grass": 11,
    "water": 12,
}


def encode_mission(mission):
    encoding = []
    for word in mission.split():
        word = word.lower()
        if word not in STOI:
            print(word)
            STOI[word] = len(STOI)
        encoding.append(STOI[word])

    encoding = np.array(encoding)
    vec = np.zeros((TEXT_LENGTH,))
    vec[: len(encoding)] = encoding
    return vec.astype(np.uint8)


def create_constraint_mask(obs, env):
    idx_to_avoid = OBJECT_TO_IDX[env.avoid_obj]
    objects = obs["image"][:, :, 0]
    constraint_mask = objects == idx_to_avoid
    return constraint_mask


def from_np_array(array_string):
    # from https://stackoverflow.com/a/42756309
    array_string = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(array_string))
