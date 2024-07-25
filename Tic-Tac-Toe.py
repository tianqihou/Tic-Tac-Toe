import os
import numpy as np


class Agent:

    def __init__(self, OOXX_index, Epsilon=0.1, LearningRate=0.1) -> None:
        self.value = np.zeros((3, 3, 3, 3, 3, 3, 3, 3, 3))
        self.current_state = np.zeros(9)
        self.previous_state = np.zeros(9)
        self.index = OOXX_index
        self.epsilon = Epsilon
        self.alpha = LearningRate

    def reset(self):
        self.current_state = np.zeros(9)
        self.previous_state = np.zeros(9)

    def actionTake(self, State: np.ndarray):
        state = State.copy()
        availabel = np.where(state == 0)[0]
        length = len(availabel)
        if length != 0:
            self.previous_state = self.current_state
            random = np.random.uniform(0, 1)
            if random < self.epsilon:
                choose = np.random.randint(length)
                state[availabel[choose]] = self.index
            else:
                tempValue = np.zeros(length)
                for i in range(length):
                    tempState = state.copy()
                    tempState[availabel[i]] = self.index
                    tempValue[i] = self.value[tuple(tempState.astype(int))]

                choose = np.argmax(tempValue)
                state[availabel[choose]] = self.index
            self.current_state = state.copy()
        return state

    def valueUpdate(self, value):
        if value != 0:
            self.value[tuple(self.current_state.astype(int))] = value
        self.value[tuple(self.previous_state.astype(int))] += self.alpha * (
            self.value[tuple(self.current_state.astype(int))]
            - self.value[tuple(self.previous_state.astype(int))]
        )

    def save(self, path):
        self.value.dump(path)

    def load(self, path):
        self.value = np.load(path, allow_pickle=True)


class Player:

    def __init__(self, OOXX_index) -> None:
        self.index = OOXX_index

    @staticmethod
    def parse_tuple(input_string: str):
        input_string = input_string.strip("()")
        input_string = input_string.replace("，", ",")
        elements = input_string.split(",")
        return tuple(int(element.strip()) - 1 for element in elements)

    def actionTake(self, State: np.ndarray):
        try:
            input_string = input("请落子(eg: 2,2): ")
            index = self.parse_tuple(input_string)
            if State.reshape(-1, 3)[index] != 0:
                print("该位置已经有棋子了，请重新落子")
                return self.actionTake(State)
            State.reshape(-1, 3)[index] = self.index
            return State.reshape(-1)
        except Exception:
            print("错误，棋盘大小为3x3，请重新落子")
            return self.actionTake(State)


def checkWin(State, indexes=[1, -1]):
    for index in indexes:
        win_value = 3 * index
        for i in range(3):
            if (
                State[i * 3 : i * 3 + 3].sum() == win_value
                or State[i::3].sum() == win_value
            ):
                return index
        if State[0::4].sum() == win_value or State[2:7:2].sum() == win_value:
            return index

    # 检查是否平局
    if np.count_nonzero(State) == 9:
        return 0

    # 游戏继续
    return None


import matplotlib.pyplot as plt


def plot(result, indexes=[1, -1]):
    avg = len(result) // 100
    result_np = np.array(result).reshape(-1, avg)
    draw = np.count_nonzero(result_np == 0, axis=1) / avg
    plt.plot(range(100), draw, label="draw")
    for index in indexes:
        plt.plot(
            range(100),
            np.count_nonzero(result_np == index, axis=1) / avg,
            label=f"{index}_win",
        )
    plt.legend()
    plt.show()


def train(agent1, agent2, epoch=30000):
    result = []
    for i in range(epoch):

        # 初始化游戏状态
        state = np.zeros(9)
        agent1.reset()
        agent2.reset()

        while True:
            if i % 2 == 0:
                state = agent2.actionTake(state)
                win = checkWin(state, [agent1.index, agent2.index])
                if win is not None:
                    break
                state = agent1.actionTake(state)
                win = checkWin(state, [agent1.index, agent2.index])
                if win is not None:
                    break
            else:
                state = agent1.actionTake(state)
                win = checkWin(state, [agent1.index, agent2.index])
                if win is not None:
                    break
                state = agent2.actionTake(state)
                win = checkWin(state, [agent1.index, agent2.index])
                if win is not None:
                    break

            agent1.valueUpdate(0)
            agent2.valueUpdate(0)

        if win == agent1.index:
            agent1.valueUpdate(10)
            agent2.valueUpdate(-10)
        elif win == agent2.index:
            agent1.valueUpdate(-10)
            agent2.valueUpdate(10)
        result.append(win)
    plot(result)
    return result


def display_state(state):
    print(". 1 2 3")
    for index, i in enumerate(state.reshape(3, 3)):
        print(index + 1, end=" ")
        for j in i:
            print("O" if j == 1 else "X" if j == -1 else ".", end=" ")
        print()


def play(player, agent):
    agent.reset()
    state = np.zeros(9)
    input_string = int(input("请选择（1.先手 2.后手）:"))

    while True:
        if input_string == 1:
            state = player.actionTake(state)
            win = checkWin(state)
            if win is not None:
                break
            state = agent.actionTake(state)
            display_state(state)
            win = checkWin(state)
            if win is not None:
                break
        else:
            state = agent.actionTake(state)
            display_state(state)
            win = checkWin(state)
            if win is not None:
                break
            state = player.actionTake(state)
            win = checkWin(state)
            if win is not None:
                break
    print(f"游戏结束！")
    if win == agent.index:
        print(f"AI胜利！")
    elif win == player.index:
        print(f"玩家胜利！")
    else:
        print("平局！")
    input_string = int(input("是否重新开始（1.是 2.否）:"))
    if input_string == 1:
        play(player, agent)


def main(path="Tic-Tac-Toe.npy"):
    player = Player(OOXX_index=-1)
    agent = Agent(OOXX_index=1, Epsilon=0)
    agent.load(path)

    win = play(player, agent)


if __name__ == "__main__":
    if not os.path.exists("Tic-Tac-Toe.npy"):
        agent1 = Agent(OOXX_index=1)
        agent2 = Agent(OOXX_index=-1)

        agent1.epsilon = 0.1
        agent2.epsilon = 0.1
        result = train(agent1, agent2, epoch=50000)
        agent1.save("Tic-Tac-Toe.npy")

    main()
