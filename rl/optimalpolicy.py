import numpy as np
import copy

FILED_TYPE = {
    "N": 0,
    "G": 1,
    }

ACTIONS = {
    "UP": 0,
    "DOWN": 1, 
    "LEFT": 2, 
    "RIGHT": 3
    }

class GridWorld:

    def __init__(self):

        self.map = [[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]]

    def step(self, pos, action):
        to_x, to_y = copy.deepcopy(pos)

        if self._is_possible_action(to_x, to_y, action) is False:
            return pos, 0 

        if action == ACTIONS["UP"]:
            to_y += -1
        elif action == ACTIONS["DOWN"]:
            to_y += 1
        elif action == ACTIONS["LEFT"]:
            to_x += -1
        elif action == ACTIONS["RIGHT"]:
            to_x += 1

        reward = self._compute_reward(to_x, to_y)
        next_pos = to_x, to_y

        return next_pos, reward

    def _is_possible_action(self, x, y, action):
        """ 
            実行可能な行動かどうかの判定
        """
        to_x, to_y = x, y

        if action == ACTIONS["UP"]:
            to_y += -1
        elif action == ACTIONS["DOWN"]:
            to_y += 1
        elif action == ACTIONS["LEFT"]:
            to_x += -1
        elif action == ACTIONS["RIGHT"]:
            to_x += 1

        if len(self.map) <= to_y or 0 > to_y:
            return False
        elif len(self.map[0]) <= to_x or 0 > to_x:
            return False

        return True

    def _compute_reward(self, x, y):
        if self.map[y][x] == FILED_TYPE["N"]:
            return 0
        elif self.map[y][x] == FILED_TYPE["G"]:
            return 1

    def get_all_state(self):
        """
            すべての状態を取得する 
        """
        all_state = []
        for y in range(len(self.map)):
            for x in range(len(self.map[0])):
                state = x, y
                if self.map[y][x] == FILED_TYPE["G"]:
                    continue
                else:
                    all_state.append(state)

        return all_state


def evaluate_policy(pi, v, gamma, all_states, grid_env):
    theta = .01
    delta = np.inf
    while delta >= theta:
        delta = 0.
        # v_dot = copy.deepcopy(v)
        for s in all_states:
            v_tmp = copy.deepcopy(v[s])
            a = pi[s]
            s2, r = grid_env.step(s, a)
            if s2 in v.keys():
                v[s] = r + gamma * v[s2]
            else:
                v[s] = r

            delta = max(delta, abs(v_tmp-v[s]))

def improvement_policy(pi, v, gamma, all_states, grid_env):
    is_convergence = True
    for s in all_states:
        b = copy.deepcopy(pi[s])
        argmax_action = None
        max_a_value = -1
        for a in ACTIONS.values():
            s2, r = grid_env.step(s, a)
            if s2 in v:
                a_value = r + gamma * v[s2]
            else:
                a_value = r

            if a_value > max_a_value:
                max_a_value = a_value
                argmax_action = a
                pi[s] = argmax_action
        if b != argmax_action:
            is_convergence = False
            return is_convergence

    return is_convergence

def print_result(result, len_x, len_y):
    result_formatted = np.full((len_y, len_x), np.inf)

    for y in range(len_y):
        for x in range(len_x):
            if (x, y) in result.keys():
                result_formatted[y][x] = float(result[(x, y)])

    print(result_formatted)


if __name__ == '__main__':
    grid_env = GridWorld()  # grid worldの環境の初期化
    all_states = grid_env.get_all_state()
    len_y = len(grid_env.map)
    len_x = len(grid_env.map[0])
    gamma = .99

    # v(s), pi(s)の初期化
    v, pi = {}, {}
    for s in all_states:
        v[s] = 0.
        pi[s] = 0

    is_convergence = False
    count = 0

    # 収束するまで繰り返す
    while is_convergence is False:
        # 方策評価
        evaluate_policy(pi, v, gamma, all_states, grid_env)

        # 方策改善
        is_convergence = improvement_policy(pi, v, gamma, all_states, grid_env)

    print("v")
    print_result(v, len_x, len_y)
    print("pi")
    print_result(pi, len_x, len_y)