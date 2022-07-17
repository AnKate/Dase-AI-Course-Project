import numpy as np
import heapq

# 记录目标状态各元素值对应的下标
final_state_index = [(1, 1), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
final_state = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
# 已访问过的节点队列
history_nodes = []
# 未访问结点的优先级队列
priority_nodes = []


# 节点的类定义
# 需要记录每个节点的父亲节点, 便于复原路径
class Node:
    def __init__(self, state):
        self.state = state
        self.father = None
        self.heuristic = None
        self.cost = 0
        self.F = None

    def __lt__(self, other):
        return self.F < other.F

    def set_heuristic(self):
        sum = 0
        for i in range(0, 9):
            element_index = tuple(np.argwhere(self.state == i)[0])
            sum += abs(element_index[0] - final_state_index[i][0]) + abs(element_index[1] - final_state_index[i][1])
        self.heuristic = sum

    def set_father(self, node):
        self.father = node

    def set_cost(self, cost):
        self.cost = cost

    def F_evaluation(self):
        self.F = self.cost + self.heuristic


# 将每一步的移动视作元素0的移动
# 上移元素0
def up(state, zero_index):
    state[zero_index[0]][zero_index[1]] = state[zero_index[0] - 1][zero_index[1]]
    state[zero_index[0] - 1][zero_index[1]] = 0
    return state


# 下移元素0
def down(state, zero_index):
    state[zero_index[0]][zero_index[1]] = state[zero_index[0] + 1][zero_index[1]]
    state[zero_index[0] + 1][zero_index[1]] = 0
    return state


# 左移元素0
def left(state, zero_index):
    state[zero_index[0]][zero_index[1]] = state[zero_index[0]][zero_index[1] - 1]
    state[zero_index[0]][zero_index[1] - 1] = 0
    return state


# 右移元素0
def right(state, zero_index):
    state[zero_index[0]][zero_index[1]] = state[zero_index[0]][zero_index[1] + 1]
    state[zero_index[0]][zero_index[1] + 1] = 0
    return state


# 向历史节点队列和优先级队列中加入新节点
def add_nodes(state, father: Node):
    # print(state)
    new_node = Node(state)  # 构建新的节点
    new_node.set_father(father)  # 记录父亲节点, 便于路径回溯
    new_node.set_heuristic()  # 计算本节点的启发
    new_node.set_cost(father.cost + 1)
    new_node.F_evaluation()
    # 判断是否在已访问列表中
    for i in history_nodes:
        temp_state = i.state
        # 如果在, 则进行其优先级与父节点的更新
        if (temp_state == state).all():
            if i.F > new_node.F:
                i.F = new_node.F
                i.set_cost(new_node.cost)
                i.set_father(father)
                # 将该节点从已访问列表中移回优先级队列中
                history_nodes.remove(i)
                heapq.heappush(priority_nodes, i)
            return
    # 否则, 判断是否已在优先级队列中
    # 若在, 则判断其是否需要更新优先级与父亲节点
    for i in priority_nodes:
        temp_state = i.state
        if (temp_state == state).all():
            if i.F > new_node.F:
                i.F = new_node.F
                i.set_cost(new_node.cost)
                i.set_father(father)
                # 维护堆
                heapq.heapify(priority_nodes)
            return
    # 若不在, 则直接加入优先级队列
    heapq.heappush(priority_nodes, new_node)    # 存储当前节点到优先级队列中


# 决策函数
def decision(current_node: Node):
    # 加入已访问过的节点队列
    history_nodes.append(current_node)
    # 获取元素0所在的位置, 用于判断可以移动的方向
    zero_index = tuple(np.argwhere(current_node.state == 0)[0])
    temp_state = np.copy(current_node.state).astype('int32')
    if zero_index[1] >= 1:
        temp_state = left(temp_state, zero_index)
        add_nodes(temp_state, current_node)
        temp_state = np.copy(current_node.state).astype('int32')
    if zero_index[1] <= 1:
        temp_state = right(temp_state, zero_index)
        add_nodes(temp_state, current_node)
        temp_state = np.copy(current_node.state).astype('int32')
    if zero_index[0] >= 1:
        temp_state = up(temp_state, zero_index)
        add_nodes(temp_state, current_node)
        temp_state = np.copy(current_node.state).astype('int32')
    if zero_index[0] <= 1:
        temp_state = down(temp_state, zero_index)
        add_nodes(temp_state, current_node)


# 回溯路径
def get_path(node: Node):
    cnt = 0
    while node.father is not None:
        node = node.father
        cnt += 1
    print(cnt)


if __name__ == "__main__":
    # 处理输入数据
    data = input()
    start = np.zeros((1, 9))
    for i in range(len(data)):
        start[0][i] = int(data[i])
    start = start.reshape((3, 3))
    # 将起始节点放入优先级队列
    start_node = Node(start)
    start_node.set_heuristic()
    start_node.F_evaluation()
    heapq.heappush(priority_nodes, start_node)
    # 优先级队列不为空时进行循环
    while len(priority_nodes) != 0:
        # 将当前节点从优先级队列中弹出
        current = heapq.heappop(priority_nodes)
        # print(current.state)
        # 若当前节点为最终节点, 则结束循环
        if (current.state == final_state).all():
            break
        decision(current)

    get_path(current)

