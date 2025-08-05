import numpy as np
import re
from scipy import sparse

from config import *


def parse_matpower_case(file_path):
    """解析MATPOWER的.m案例文件（改进版）"""
    with open(file_path, 'r') as f:
        content = f.read()

    # 提取baseMVA
    baseMVA = float(re.search(r'mpc\.baseMVA\s*=\s*(\d+)', content).group(1))

    def parse_array(data_str):
        """改进的数组解析函数，处理分号和注释"""
        lines = []
        for line in data_str.split('\n'):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            # 移除行末分号和其他非数字字符
            line = re.sub(r'[^\d\.\s-]', '', line)
            if line:
                lines.append(list(map(float, line.split())))
        return np.array(lines)

    # 提取并解析bus数据
    bus_data = re.search(r'mpc\.bus\s*=\s*\[([^\]]+)\]', content, re.DOTALL).group(1)
    bus = parse_array(bus_data)

    # 提取并解析gen数据
    gen_data = re.search(r'mpc\.gen\s*=\s*\[([^\]]+)\]', content, re.DOTALL).group(1)
    gen = parse_array(gen_data)

    # 提取并解析branch数据
    branch_data = re.search(r'mpc\.branch\s*=\s*\[([^\]]+)\]', content, re.DOTALL).group(1)
    branch = parse_array(branch_data)

    return {
        'baseMVA': baseMVA,
        'bus': bus,
        'gen': gen,
        'branch': branch
    }


def makeYbus(bus, branch):  # 未考虑变压器和移相器
    nb = bus.shape[0]  # 节点数量
    # 计算支路导纳矩阵元素
    stat = branch[:, BR_STATUS]  # 支路状态
    Ys = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])  # 串联导纳
    Bc = stat * branch[:, BR_B]  # 线路充电电纳
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA  # 节点对地导纳 仅对自导纳生效

    # 计算变比
    tap = np.ones(len(branch), dtype=complex)
    for i in range(len(branch)):
        if branch[i, TAP] != 0:
            tap[i] = branch[i, TAP]
    tap = tap * np.exp(1j * np.pi / 180 * branch[:, SHIFT])
    print(tap)
    # tap
    # [1. + 0.j 1. + 0.j 1. + 0.j 1. + 0.j 1. + 0.j 1. + 0.j 1. + 0.j
    #  0.978 + 0.j 0.969 + 0.j 0.932 + 0.j 1. + 0.j 1. + 0.j 1. + 0.j 1. + 0.j
    #  1. + 0.j 1. + 0.j 1. + 0.j 1. + 0.j 1. + 0.j 1. + 0.j]
    # 获取支路连接关系
    j = branch[:, F_BUS].astype(int) - 1
    i = branch[:, T_BUS].astype(int) - 1
    # 构建Ybus矩阵
    Yii = Ys + 1j * Bc / 2
    # print("Yii", Yii)
    Yij = -Ys / tap
    # print("Yij", Yij)
    Yji = -Ys / np.conj(tap)
    # print("Yji", Yji)
    Yjj = Yii / tap ** 2
    # print("Yjj", Yjj)
    Ybus = sparse.csr_matrix(
        (np.concatenate([Yii, Yij, Yji, Yjj]),
         (np.concatenate([i, i, j, j]), np.concatenate([i, j, i, j]))),
        shape=(nb, nb)  # 注意这里括号的闭合
    ) + sparse.diags(Ysh, 0, shape=(nb, nb))
    # print(Ybus)

    return Ybus


def getBusType(bus):  # bus type (1 - PQ bus, 2 - PV bus, 3 - reference bus, 4 - isolated bus)
    pv = []
    pq = []
    ref = []
    types = bus[:, BUS_TYPE]
    # print(types)
    for i in range(len(types)):
        if types[i] == 1:
            pq.append(i)
        elif types[i] == 2:
            pv.append(i)
        elif types[i] == 3:
            ref.append(i)

    return pq, pv, ref


def newtonpf_I_polar(Ybus, P0, Q0, U0, theta0, pv, pq, max_it=10, tol=1e-8):
    pq_pv = np.concatenate([pq, pv])
    # print("pq_pv", pq_pv)

    P0_cut = P0[pq_pv]
    Q0_pq = Q0[pq]
    Q0_cut = Q0_pq

    n_pq = len(pq)
    n_pv = len(pv)
    n = n_pq + n_pv + 1

    G = np.real(Ybus)
    B = np.imag(Ybus)

    for epoch in range(max_it):
        theta0_ij = np.zeros((len(theta0), len(theta0)))
        for i in range(len(theta0)):
            for j in range(len(theta0)):
                theta0_ij[i][j] = theta0[i] - theta0[j]

        P = np.zeros(n)
        Q = np.zeros(n)
        for i in range(n):
            for j in range(n):
                theta_ij = theta0[i] - theta0[j]
                P[i] += U0[i] * U0[j] * (G[i, j] * np.cos(theta_ij) + B[i, j] * np.sin(theta_ij))
                Q[i] += U0[i] * U0[j] * (G[i, j] * np.sin(theta_ij) - B[i, j] * np.cos(theta_ij))

        d_P = P0_cut - P[pq_pv]
        # print("d_P", d_P)
        d_Q = Q0_cut - Q[pq]
        # print("d_Q", d_Q)

        loss = abs(max(np.max(d_P), np.max(d_Q)))
        print("当前迭代次数：", epoch, "当前误差：", loss)
        if loss < tol:
            print("当前迭代次数：", epoch, "计算成功！")
            break
        # 构建H
        H = np.zeros((n, n))
        for i in pq_pv:
            for j in pq_pv:
                theta_ij = theta0[i] - theta0[j]
                if i != j:
                    H[i, j] = -U0[i] * U0[j] * (G[i, j] * np.sin(theta_ij) - B[i, j] * np.cos(theta_ij))
                else:
                    H[i, j] = Q[i] + (U0[j] ** 2) * B[i, j]
        H = H[pq_pv][:, pq_pv]
        # 构建N
        N = np.zeros((n, n))
        for i in pq_pv:
            for j in pq:
                theta_ij = theta0[i] - theta0[j]
                if i != j:
                    N[i, j] = -U0[i] * U0[j] * (G[i, j] * np.cos(theta_ij) + B[i, j] * np.sin(theta_ij))
                else:
                    N[i, j] = -P[i] - (U0[j] ** 2) * G[i, j]
        N = N[pq_pv][:, pq]
        # 构建M
        M = np.zeros((n, n))
        for i in pq:
            for j in pq_pv:
                theta_ij = theta0[i] - theta0[j]
                if i != j:
                    M[i, j] = U0[i] * U0[j] * (G[i, j] * np.cos(theta_ij) + B[i, j] * np.sin(theta_ij))
                else:
                    M[i, j] = -P[i] + (U0[j] ** 2) * G[i, j]
        M = M[pq][:, pq_pv]
        # 构建L
        L = np.zeros((n, n))
        for i in pq:
            for j in pq:
                theta_ij = theta0[i] - theta0[j]
                if i != j:
                    L[i, j] = -U0[i] * U0[j] * (G[i, j] * np.sin(theta_ij) - B[i, j] * np.cos(theta_ij))
                else:
                    L[i, j] = -Q[i] + (U0[j] ** 2) * B[i, j]
        L = L[pq][:, pq]
        # 组合！
        J = np.block([
            [H, N],
            [M, L]
        ])
        # print("J", J)

        F = np.concatenate([d_P, d_Q])
        # print("J", J.shape, "F", F.shape)
        dx = np.linalg.solve(-J, F)
        # print("dx", dx)

        theta0[pq] += dx[0:n_pq]
        theta0[pv] += dx[n_pq:n_pq + n_pv]
        U0[pq] += (dx[n_pq + n_pv:])
        # print("dU", (dx[n_pq + n_pv:]))

    return U0, theta0, P, Q


if __name__ == "__main__":
    try:
        case = parse_matpower_case('case14.m')
        print("解析成功！")
        print("\nBus数据示例:")
        print(len(case['bus']))  # 打印前两行bus数据
    except Exception as e:
        print(f"解析失败: {str(e)}")

    baseMVA = case['baseMVA']
    bus = case['bus']
    gen = case['gen']
    branch = case['branch']
    pq, pv, ref = getBusType(bus)
    nn = bus.shape[0]  # 节点数量
    # 初始化P、Q、U、theta
    P0 = np.zeros(nn)
    Q0 = np.zeros(nn)
    for i in range(len(gen)):
        P0[int(gen[i, GEN_BUS] - 1)] += gen[i, PG]
    for i in range(len(bus)):
        P0[int(bus[i, BUS_I] - 1)] -= bus[i, PD]

    for i in range(len(gen)):
        Q0[int(gen[i, GEN_BUS] - 1)] += gen[i, QG]
    for i in range(len(bus)):
        Q0[int(bus[i, BUS_I] - 1)] -= bus[i, QD]

    P0 = P0 / baseMVA
    Q0 = Q0 / baseMVA
    print("P0", P0)
    print("Q0", Q0)
    U0 = np.zeros(nn)
    theta0 = np.zeros(nn)
    U0 = bus[:, VM]
    U0[gen[:, GEN_BUS].astype(int) - 1] = gen[:, VG]
    theta0 = np.deg2rad(bus[:, VA])
    print("U0", U0)
    print("theta0", theta0)
    U0_cal = U0.copy()
    theta0_cal = theta0.copy()

    Ybus = makeYbus(bus, branch)
    print("_____________YMatrix_______________")
    print(Ybus)

    U0_cal, theta0_cal, P, Q = newtonpf_I_polar(Ybus, P0, Q0, U0_cal, theta0_cal, pv, pq)

    print("_____________P_______________")
    for i in range(len(P)):
        print(i, P[i])
    print("_____________Q_______________")
    for i in range(len(Q)):
        print(i, Q[i])
    print("_____________U_______________")
    for i in range(len(U0_cal)):
        print(i, U0_cal[i])
    print("_____________theta_______________")
    for i in range(len(theta0_cal)):
        print(i, theta0_cal[i])
