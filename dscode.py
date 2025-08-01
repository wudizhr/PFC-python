import numpy as np
import re


def parse_matpower_case(file_path):
    """解析MATPOWER的.m案例文件（改进版）"""
    with open(file_path, 'r') as f:
        content = f.read()

    # 提取baseMVA
    baseMVA = float(re.search(r'mpc\.baseMVA\s*=\s*(\d+);', content).group(1))

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


def build_YMatrix(case):
    bus = case['bus']
    branches = case['branch']
    n = len(bus)
    Y = np.zeros((n, n), dtype=complex)  # 直接构建复数导纳矩阵

    for branch in branches:
        f = int(branch[0] - 1)
        t = int(branch[1] - 1)
        r, x, b = branch[2], branch[3], branch[4]

        # 串联导纳
        y_series = 1 / (r + 1j * x) if (r != 0 or x != 0) else 0

        # 并联电纳（均分到两端节点）
        y_shunt = 1j * b / 2

        # 更新导纳矩阵
        Y[f, f] += y_series + y_shunt
        Y[t, t] += y_series + y_shunt
        Y[f, t] -= y_series
        Y[t, f] -= y_series

    return {'Y': Y}


def build_initial_conditions(case):
    bus = case['bus']
    gen = case['gen']
    n = len(bus)

    # 初始化节点功率和电压
    P = np.zeros(n)
    Q = np.zeros(n)
    V = np.ones(n, dtype=complex)  # 复数电压

    # 处理负荷（注入为负）
    P -= bus[:, 2]  # Pd
    Q -= bus[:, 3]  # Qd

    # 处理发电机（注入为正）
    for g in gen:
        bus_idx = int(g[0] - 1)
        P[bus_idx] += g[1]  # Pg
        Q[bus_idx] += g[2]  # Qg
        if bus[bus_idx, 1] == 2:  # PV节点
            V[bus_idx] = g[5] * np.exp(1j * np.deg2rad(bus[bus_idx, 8]))  # Vg∠θ

    return {'P': P, 'Q': Q, 'V': V}


def build_jacobian(Y, V, bus):
    n = len(bus)
    J = np.zeros((2 * n, 2 * n))
    V_complex = V['V']

    for i in range(n):
        if bus[i, 1] == 3:  # Slack节点跳过
            continue

        for j in range(n):
            # 公共部分
            Vj = V_complex[j]
            Yij = Y['Y'][i, j]
            G, B = Yij.real, Yij.imag

            # H/N (有功对θ/V)
            J[2 * i, 2 * j] = -V_complex[i].imag * Yij * Vj  # ∂P/∂θ
            J[2 * i, 2 * j + 1] = V_complex[i].real * Yij * Vj  # ∂P/∂V

            # J/L (无功对θ/V)
            J[2 * i + 1, 2 * j] = -V_complex[i].real * Yij * Vj  # ∂Q/∂θ
            J[2 * i + 1, 2 * j + 1] = V_complex[i].imag * Yij * Vj  # ∂Q/∂V

        # PV节点特殊处理：替换Q方程为V方程
        if bus[i, 1] == 2:
            J[2 * i + 1, :] = 0
            J[2 * i + 1, 2 * i] = 2 * V_complex[i].real  # ∂|V|²/∂e
            J[2 * i + 1, 2 * i + 1] = 2 * V_complex[i].imag  # ∂|V|²/∂f

    # 删除平衡节点对应的行列
    slack = np.where(bus[:, 1] == 3)[0][0]
    mask = np.ones(2 * n, dtype=bool)
    mask[[2 * slack, 2 * slack + 1]] = False
    J = J[mask][:, mask]

    return J


def ac_pf(case, max_iter=100, tol=1e-5):
    # 初始化
    Y = build_YMatrix(case)
    V = build_initial_conditions(case)
    bus = case['bus']
    n = len(bus)

    for iter in range(max_iter):
        # 计算功率不平衡量
        S_calc = V['V'] * np.conj(Y['Y'] @ V['V'])
        delta_P = V['P'] - S_calc.real
        delta_Q = V['Q'] - S_calc.imag

        # 构建雅可比矩阵（复数）
        J = np.zeros((2 * n, 2 * n), dtype=complex)
        for i in range(n):
            if bus[i, 1] == 3:  # 跳过Slack节点
                continue
            for j in range(n):
                Yij = Y['Y'][i, j]
                Vj = V['V'][j]
                # ∂P/∂θ, ∂P/∂V
                J[2 * i, 2 * j] = -V['V'][i].imag * Yij * Vj
                J[2 * i, 2 * j + 1] = V['V'][i].real * Yij * Vj
                # ∂Q/∂θ, ∂Q/∂V
                J[2 * i + 1, 2 * j] = -V['V'][i].real * Yij * Vj
                J[2 * i + 1, 2 * j + 1] = V['V'][i].imag * Yij * Vj

        # 筛选有效方程
        mask = []
        delta = []
        for i in range(n):
            if bus[i, 1] == 3:
                continue
            delta.append(delta_P[i])
            mask.append(2 * i)
            if bus[i, 1] == 1:  # PQ节点
                delta.append(delta_Q[i])
                mask.append(2 * i + 1)
            else:  # PV节点：替换Q方程为V方程
                delta.append(np.abs(V['V'][i]) ** 2 - (V['V'][i].real ** 2 + V['V'][i].imag ** 2))
                mask.append(2 * i + 1)

        J = J[mask][:, mask].real  # 转换为实数矩阵
        delta = np.array(delta)

        # 收敛判断
        if np.max(np.abs(delta)) < tol:
            print(f"Converged in {iter} iterations")
            break
        print(np.max(np.abs(delta)))
        # 求解修正方程
        dx = np.linalg.solve(J, delta)

        # 更新电压
        idx = 0
        for i in range(n):
            if bus[i, 1] == 3:
                continue
            # 更新相角（θ）
            theta = np.angle(V['V'][i]) + dx[idx]
            if bus[i, 1] == 1:  # PQ节点更新幅值
                mag = np.abs(V['V'][i]) + dx[idx + 1]
                idx += 2
            else:  # PV节点幅值固定
                mag = np.abs(V['V'][i])
                idx += 1
            V['V'][i] = mag * np.exp(1j * theta)

    return V['V']


if __name__ == "__main__":
    try:
        case = parse_matpower_case('case5.m')
        print("解析成功！")
        print("\nBus数据示例:")
        print(len(case['bus']))  # 打印前两行bus数据
    except Exception as e:
        print(f"解析失败: {str(e)}")

    ac_pf(case)

