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
    print(Ys)
    Bc = stat * branch[:, BR_B]  # 线路充电电纳
    print(Bc)

    YMatrix = Ys + 1j * Bc / 2
    print(YMatrix)
    # 获取支路连接关系 (转换为0-based索引)
    f = branch[:, F_BUS].astype(int) - 1
    t = branch[:, T_BUS].astype(int) - 1
    # 构建Ybus矩阵
    Ybus = sparse.csr_matrix(
        (np.concatenate([YMatrix, YMatrix, -YMatrix, -YMatrix]),
         (np.concatenate([f, t, f, t]), np.concatenate([f, t, t, f]))),
        shape=(nb, nb)  # 注意这里括号的闭合
    )
    print(Ybus)

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


def newtonpf_I_polar(Ybus, P0, Q0, U0, theta0, pv, pq, max_it=50, tol=1e-10):
    pq_pv = np.concatenate([pq, pv])
    print("pq_pv", pq_pv)

    P0_pq = P0[pq]
    P0_pv = P0[pv]
    P0_cut = np.concatenate([P0_pq, P0_pv])
    Q0_pq = Q0[pq]
    Q0_cut = Q0_pq

    n_pq = len(pq)
    n_pv = len(pv)
    n = n_pq + n_pv + 1

    G = np.real(Ybus)
    B = np.imag(Ybus)
    print("G", G)
    print("B", B)

    for epoch in range(max_it):
        theta0_ij = np.zeros((len(theta0), len(theta0)))
        for i in range(len(theta0)):
            for j in range(len(theta0)):
                theta0_ij[i][j] = theta0[i] - theta0[j]
        print("theta0_ij", theta0_ij)
        cos_theta_ij = np.cos(theta0_ij)
        print("cos_theta_ij", cos_theta_ij)
        sin_theta_ij = np.sin(theta0_ij)
        print("sin_theta_ij", sin_theta_ij)

        P = np.zeros(n)
        Q = np.zeros(n)
        for i in range(n):
            for j in range(n):
                theta_ij = theta0[i] - theta0[j]
                P[i] += U0[i] * U0[j] * (G[i, j] * np.cos(theta_ij) + B[i, j] * np.sin(theta_ij))
                Q[i] += U0[i] * U0[j] * (G[i, j] * np.sin(theta_ij) - B[i, j] * np.cos(theta_ij))

        d_P = P0_cut - P[pq_pv]
        print("d_P", d_P)
        d_Q = Q0_cut - Q[pq]
        print("d_Q", d_Q)

        loss = abs(max(np.max(d_P), np.max(d_Q)))
        print("当前迭代次数：", epoch, "当前误差：", loss)
        if loss < tol:
            print("当前迭代次数：", epoch, "计算成功！")
            break
        # 构建H
        H_pq = -U0[pq].reshape(-1, 1) * U0[pq_pv] * (sin_theta_ij[pq] * G[pq].toarray() - cos_theta_ij[pq] * B[
            pq].toarray())[:, pq_pv]
        # print("H_pq", H_pq)
        H_pv = -U0[pv].reshape(-1, 1) * U0[pq_pv] * (sin_theta_ij[pv] * G[pv].toarray() - cos_theta_ij[pv] * B[
            pv].toarray())[:, pq_pv]
        # print("H_pv", H_pv)
        H = np.concatenate([H_pq, H_pv])
        # print("H", H.shape)
        for i in range(len(pq_pv)):
            if i >= len(Q):
                Q_t = 0
            else:
                Q_t = Q[i]
            H[i][i] = Q_t + (U0[pq_pv[i]] ** 2) * B[pq_pv[i], pq_pv[i]]
        print("H", H)

        # 构建N
        N_pq = -U0[pq].reshape(-1, 1) * U0[pq] * (cos_theta_ij[pq] * G[pq].toarray() + sin_theta_ij[pq] * B[
            pq].toarray())[:, pq]
        print("N_pq", N_pq)
        N_pv = -U0[pv].reshape(-1, 1) * U0[pq] * (cos_theta_ij[pv] * G[pv].toarray() + sin_theta_ij[pv] * B[
            pv].toarray())[:, pq]
        print("N_pv", N_pv)
        N = np.concatenate([N_pq, N_pv])
        print("N", N.shape)
        for i in range(len(pq)):
            N[i][i] = -P[i] - (U0[pq[i]] ** 2) * G[pq[i], pq[i]]
        # print(Q0[1])

        # 构建M L
        M = U0[pq].reshape(-1, 1) * U0[pq_pv] * (cos_theta_ij[pq] * G[pq].toarray() + sin_theta_ij[pq] * B[
            pq].toarray())[:, pq_pv]
        print("M", M.shape)
        for i in range(len(pq)):
            M[i][i] = -P[i] + (U0[pq[i]] ** 2) * G[pq[i], pq[i]]
        L = -U0[pq].reshape(-1, 1) * U0[pq] * (sin_theta_ij[pq] * G[pq].toarray() - cos_theta_ij[pq] * B[pq].toarray())[
                                              :, pq]
        print("L", L.shape)
        for i in range(len(pq)):
            L[i][i] = -Q[i] + (U0[pq[i]] ** 2) * B[pq[i], pq[i]]
        # print(Q0[1])

        # 组合！
        J = np.block([
            [H, N],
            [M, L]
        ])
        print("J", J)

        F = np.concatenate([d_P, d_Q])
        print("J", J.shape, "F", F.shape)
        dx = np.linalg.solve(-J, F)
        print("dx", dx)

        theta0[pq] += dx[0:n_pq]
        theta0[pv] += dx[n_pq:n_pq + n_pv]
        U0[pq] += (dx[n_pq + n_pv:])

    return U0, theta0


if __name__ == "__main__":
    try:
        case = parse_matpower_case('case5.m')
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
    Ybus = makeYbus(bus, branch)
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
    P0 = P0/baseMVA
    Q0 = P0/baseMVA

    print("P0", P0)
    print("Q0", Q0)
    U0 = np.zeros(nn)
    theta0 = np.zeros(nn)
    U0 = bus[:, VM]
    theta0 = np.deg2rad(bus[:, VA])
    print("U0", U0)
    print("theta0", theta0)
    U0_cal = U0.copy()
    theta0_cal = theta0.copy()

    U0_cal, theta0_cal = newtonpf_I_polar(Ybus, P0, Q0, U0_cal, theta0_cal, pv, pq)

    print("P0", P0)
    print("Q0", Q0)
    print("Ybus", Ybus)
    print("U0_cl", U0_cal)
    print("theta0_cl", theta0_cal)

    # pq_pv = np.concatenate([pq, pv])
    # print("pq_pv", pq_pv)
    # P0_pq = P0[pq]
    # P0_pv = P0[pv]
    # P0_cut = np.concatenate([P0_pq, P0_pv])
    # Q0_pq = Q0[pq]
    # Q0_cut = Q0_pq
    # n_pq = len(pq)
    # n_pv = len(pv)
    # G = np.real(Ybus)
    # B = np.imag(Ybus)
    # print("G", G)
    # print("B", B)
    #
    # theta0 = np.array([0.0571, -0.0133, -0.0086, 0, 0.0718])
    # U0 = np.array([1.0000, 0.9893, 1.0000, 1.0000, 1.0000])
    # theta0_ij = np.zeros((len(theta0), len(theta0)))
    # for i in range(len(theta0)):
    #     for j in range(len(theta0)):
    #         theta0_ij[i][j] = theta0[i] - theta0[j]
    # print("theta0_ij", theta0_ij)
    # cos_theta_ij = np.cos(theta0_ij)
    # print("cos_theta_ij", cos_theta_ij)
    # sin_theta_ij = np.sin(theta0_ij)
    # print("sin_theta_ij", sin_theta_ij)
    # P_pq = U0[pq_pv] * (cos_theta_ij[pq] * G[pq].toarray() + sin_theta_ij[pq] * B[pq].toarray())[:, pq_pv]
    # P_pq = U0[pq] * np.sum(P_pq, axis=1)
    # print("P_pq", P_pq)
    # P_pv = U0[pq_pv] * (cos_theta_ij[pv] * G[pv].toarray() + sin_theta_ij[pv] * B[pv].toarray())[:, pq_pv]
    # P_pv = U0[pv] * np.sum(P_pv, axis=1)
    # print("P_pq", P_pv)
    # P = np.concatenate([P_pq, P_pv])
    # print("P", P)
    # Q = U0[pq_pv] * (sin_theta_ij[pq] * G[pq].toarray() - cos_theta_ij[pq] * B[pq].toarray())[:, pq_pv]
    # Q = U0[pq] * np.sum(Q, axis=1)
    # print("Q", Q)
    #
    # d_P = P0_cut - P
    # print("d_P", d_P)
    # d_Q = Q0_cut - Q
    # print("d_Q", d_Q)
    #
    # print("P0", P0)
    # print("Q0", Q0)
    # loss = max(np.max(d_P), np.max(d_Q))
    # print("loss", loss)
    # print("P", P)
    # print("Q", Q)
    #
    # theta = (np.array([0.0571, -0.0133, -0.0086, 0, 0.0718]))
    # U = np.array([1.0000, 0.9893, 1.0000, 1.0000, 1.0000])
    # n = len(U)
    # G = np.real(Ybus)  # 电导矩阵
    # B = np.imag(Ybus)  # 电纳矩阵
    # P = np.zeros(n)
    # Q = np.zeros(n)
    # for i in range(n):
    #     for j in range(n):
    #         theta_ij = theta[i] - theta[j]
    #         P[i] += U[i] * U[j] * (G[i, j] * np.cos(theta_ij) + B[i, j] * np.sin(theta_ij))
    #         Q[i] += U[i] * U[j] * (G[i, j] * np.sin(theta_ij) - B[i, j] * np.cos(theta_ij))
    #
    # print("P", P)
    # print("Q", Q)
