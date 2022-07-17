import numpy as np
import heapq

# 已访问过的节点
# history_status = []
# 已展开而未访问的节点, 使用heapq维护的小顶堆实现优先级队列
priority_status = []


# 记录每个位置的状态
class Status:
    def __init__(self, state):
        self.state = state
        self.cost = 0
        self.father = None

    def __lt__(self, other):
        return self.cost < other.cost


# 展开子节点, 添加到队列里
def add_status(child_state, father: Status, distance):
    # 实例化一个新的节点
    new_status = Status(child_state)
    new_status.cost = father.cost + distance
    new_status.father = father
    # 对于本问题, 不需要记录历史信息, 也不需要更新优先级队列中的信息
    # 优先级队列中, 相同位置的状态父节点不同, 即视作两条路径中相互独立的节点
    # 所以无需更新
    # 判断当前状态是否在优先级队列中
    # for i in priority_status:
    #     temp_state = i.state
    #     if temp_state == child_state:
    #         if i.cost > new_status.cost:
    #             i.cost = new_status.cost
    #             i.father = new_status.father
    #             heapq.heapify(priority_status)
    #         return
    heapq.heappush(priority_status, new_status)


# 决策函数
# 获取后继子节点
def decision(data, status: Status):
    current_state = status.state
    for i in range(len(data[current_state - 1])):
        distance = data[current_state - 1, i]
        if distance != 0:
            next_state = i + 1
            add_status(next_state, status, distance)


if __name__ == "__main__":
    # 数据读取
    N, M, K = [int(i) for i in input().split(' ')[:3]]
    # 以矩阵形式存储各点位之间的路径关系
    data = np.zeros((N, N)).astype('int32')
    for i in range(M):
        m, n, k = [int(i) for i in input().split(' ')[:3]]
        data[m - 1, n - 1] = k

    cnt = 0
    heapq.heappush(priority_status, Status(N))
    while len(priority_status) != 0:
        current_status = heapq.heappop(priority_status)
        if current_status.state == 1:
            cnt += 1
            # 第几次输出就代表是第几短的路径
            print(current_status.cost)
            # 若输出K次, 则说明已经找到
            if cnt == K:
                break
        decision(data, current_status)
    # 若不到K次, 则-1补齐剩下的
    while cnt != K:
        cnt += 1
        print(-1)
