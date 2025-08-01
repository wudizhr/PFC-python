import numpy as np
import re
from scipy import sparse
from scipy.sparse.linalg import spsolve

from config import *
import inspect  # 添加这行导入语句
import math
from functools import partial


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


def makeYbus(baseMVA, bus, branch):
    """
    构建节点导纳矩阵和支路导纳矩阵

    参数:
        baseMVA: 基准功率
        bus: 节点数据矩阵
        branch: 支路数据矩阵

    返回:
        Ybus: 节点导纳矩阵
        Yf: 支路首端电流注入矩阵
        Yt: 支路末端电流注入矩阵
    """
    nb = bus.shape[0]  # 节点数量
    nl = branch.shape[0]  # 支路数量

    # 检查节点编号是否连续 (1-based)
    if not np.array_equal(bus[:, BUS_I], np.arange(1, nb + 1)):
        raise ValueError("节点必须连续编号，请使用ext2int()转换为内部编号")

    # 计算支路导纳矩阵元素
    stat = branch[:, BR_STATUS]  # 支路状态
    Ys = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])  # 串联导纳
    Bc = stat * branch[:, BR_B]  # 线路充电电纳

    # 处理变压器变比和移相器
    tap = np.ones(nl, dtype=complex)
    non_zero_tap = np.where(branch[:, TAP] != 0)[0]
    tap[non_zero_tap] = branch[non_zero_tap, TAP]
    tap = tap * np.exp(1j * np.pi / 180 * branch[:, SHIFT])  # 考虑移相器

    # 计算支路导纳矩阵元素
    Ytt = Ys + 1j * Bc / 2
    print("Ytt")
    print(Ytt)
    Yff = Ytt / (tap * np.conj(tap))
    print("Yff")
    print(Yff)
    Yft = -Ys / np.conj(tap)
    print("Yft")
    print(Yft)
    Ytf = -Ys / tap
    print("Ytf")
    print(Ytf)

    # 计算节点并联导纳
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
    print(Ysh)

    # 获取支路连接关系 (转换为0-based索引)
    f = branch[:, F_BUS].astype(int) - 1
    t = branch[:, T_BUS].astype(int) - 1

    # 构建Yf和Yt矩阵
    i = np.concatenate([np.arange(nl), np.arange(nl)])
    Yf = sparse.csr_matrix(
        (np.concatenate([Yff, Yft]),
         (i, np.concatenate([f, t]))),
        shape=(nl, nb)
    )
    Yt = sparse.csr_matrix(
        (np.concatenate([Ytf, Ytt]),
         (i, np.concatenate([f, t]))),
        shape=(nl, nb)  # 注意这里括号的闭合
    )
    print(np.concatenate([Yff, Yft, Ytf, Ytt]), np.concatenate([f, f, t, t]), np.concatenate([f, t, f, t]))
    # 构建Ybus矩阵
    Ybus = sparse.csr_matrix(
        (np.concatenate([Yff, Yft, Ytf, Ytt]),
         (np.concatenate([f, f, t, t]), np.concatenate([f, t, f, t]))),
        shape=(nb, nb)  # 注意这里括号的闭合
    )

    # 构建Ybus矩阵
    Ybus = sparse.csr_matrix(
        (np.concatenate([Yff, Yft, Ytf, Ytt]),
         (np.concatenate([f, f, t, t]), np.concatenate([f, t, f, t]))),
        shape=(nb, nb)  # 注意这里括号的闭合
    )
    print("Ybus")
    print(Ybus)
    Ybus = sparse.csr_matrix(
        (np.concatenate([Yff, Yft, Ytf, Ytt]),
         (np.concatenate([f, f, t, t]), np.concatenate([f, t, f, t]))),
        shape=(nb, nb)  # 注意这里括号的闭合
    ) + sparse.diags(Ysh, 0, shape=(nb, nb))
    print("Ybus")
    print(Ybus)

    return Ybus, Yf, Yt


def bustypes(bus, gen):
    """
    构建每种类型总线(REF, PV, PQ)的索引列表

    参数:
        bus: 总线数据矩阵，每行代表一个总线
        gen: 发电机数据矩阵，每行代表一个发电机

    返回:
        ref: 参考(松弛)总线索引列表
        pv: PV(电压控制)总线索引列表
        pq: PQ(负荷)总线索引列表

    注意:
        - 状态为"停运"的发电机被视为PQ总线(发电量为零)
        - 假设总线和发电机数据已转换为内部连续编号
    """
    nb = bus.shape[0]  # 总线数量
    ng = gen.shape[0]  # 发电机数量

    # 构建发电机连接矩阵
    # Cg[i,j] = 1 表示第j台发电机连接在总线i上且处于运行状态
    row = gen[:, GEN_BUS].astype(int) - 1  # 转换为0-based索引
    col = np.arange(ng)
    data = (gen[:, GEN_STATUS] > 0).astype(float)
    Cg = sparse.csr_matrix((data, (row, col)), shape=(nb, ng))

    # 计算每条总线上运行的发电机数量
    bus_gen_status = Cg.dot(np.ones(ng))  # 每行求和

    # 构建索引列表
    ref = np.where((bus[:, BUS_TYPE] == REF) & (bus_gen_status > 0))[0]  # 参考总线
    pv = np.where((bus[:, BUS_TYPE] == PV) & (bus_gen_status > 0))[0]  # PV总线
    pq = np.where((bus[:, BUS_TYPE] == PQ) | (bus_gen_status == 0))[0]  # PQ总线

    if len(ref) == 0 and len(pv) > 0:
        ref = np.array([pv[0]])  # 选择第一个PV总线
        pv = np.delete(pv, 0)  # 从PV列表中移除

    return ref, pv, pq


def makeSdzip(baseMVA, bus, mpopt):
    """
    构建ZIP负荷模型参数

    参数:
        baseMVA: 基准功率
        bus: 节点数据矩阵
        mpopt: 选项字典

    返回:
        包含ZIP负荷参数的字典 {'p', 'i', 'z'}
    """
    # 默认ZIP系数 (恒定功率负荷)
    p = (bus[:, PD] + 1j * bus[:, QD]) / baseMVA
    i = np.zeros_like(p)
    z = np.zeros_like(p)

    # 如果有ZIP负荷选项
    if mpopt is not None and 'exp' in mpopt and 'sys_wide_zip_loads' in mpopt['exp']:
        zip_coeff = mpopt['exp']['sys_wide_zip_loads']
        if 'pw' in zip_coeff and len(zip_coeff['pw']) == 3:
            pw = zip_coeff['pw']
            p = p * pw[0]
            i = p * pw[1]
            z = p * pw[2]

        if 'qw' in zip_coeff and len(zip_coeff['qw']) == 3:
            qw = zip_coeff['qw']
            q = (bus[:, QD] / baseMVA) * qw[0]
            i += 1j * q * qw[1]
            z += 1j * q * qw[2]

    return {'p': p, 'i': i, 'z': z}


def makeSbus(baseMVA, bus, gen, mpopt=None, Vm=None, Sg=None):
    """
    构建节点复功率注入向量

    参数:
        baseMVA: 基准功率
        bus: 节点数据矩阵
        gen: 发电机数据矩阵
        mpopt: 选项字典 (可选)
        Vm: 电压幅值向量 (可选)
        Sg: 发电机功率注入向量 (可选)

    返回:
        Sbus: 节点复功率注入向量
        dSbus_dVm: 功率注入对电压幅值的偏导数矩阵 (可选)
    """
    nb = bus.shape[0]  # 节点数量

    # 处理默认输入
    if Vm is None:
        Vm = np.ones(nb)

    # 获取负荷参数 (ZIP模型)
    Sd = makeSdzip(baseMVA, bus, mpopt)

    # 如果请求偏导数
    if len(inspect.signature(makeSbus).parameters) > 5 and 'dSbus_dVm' in locals():
        Sbus = None
        if Vm is None:
            dSbus_dVm = sparse.csr_matrix((nb, nb))
        else:
            dSbus_dVm = -sparse.diags(Sd['i'] + 2 * Vm * Sd['z'], 0)
        return Sbus, dSbus_dVm
    else:
        # 计算节点发电功率
        on = np.where(gen[:, GEN_STATUS] > 0)[0]  # 运行中的发电机
        gbus = gen[on, GEN_BUS].astype(int) - 1  # 转换为0-based索引
        ngon = len(on)

        # 构建发电机连接矩阵
        Cg = sparse.csr_matrix(
            (np.ones(ngon), (gbus, np.arange(ngon))),
            shape=(nb, ngon)
        )

        # 计算发电机功率注入
        if Sg is not None and len(Sg) > 0:
            Sbusg = Cg.dot(Sg[on])
        else:
            Sbusg = Cg.dot((gen[on, PG] + 1j * gen[on, QG]) / baseMVA)

        # 计算负荷功率 (考虑ZIP模型)
        if Vm is None:
            Vm = np.ones(nb)
        Sbusd = Sd['p'] + Sd['i'] * Vm + Sd['z'] * Vm ** 2

        # 计算净功率注入 (发电 - 负荷)
        Sbus = Sbusg - Sbusd

        return Sbus


def dImis_dV(Sbus, Ybus, V):
    n = len(V)
    Vm = np.abs(V)  # 电压幅值
    Vnorm = V / Vm  # 单位化电压
    Ibus = np.conj(Sbus / V)  # 节点注入电流

    diagV = sparse.diags(V, 0, format='csr')
    diagIbus = sparse.diags(Ibus, 0, format='csr')
    diagIbusVm = sparse.diags(Ibus / Vm, 0, format='csr')
    diagVnorm = sparse.diags(Vnorm, 0, format='csr')

    # 计算偏导数
    dImis_dVa = 1j * (Ybus @ diagV - diagIbus)  # ∂Imis/∂θ
    dImis_dVm = Ybus @ diagVnorm + diagIbusVm  # ∂Imis/∂V

    return dImis_dVa, dImis_dVm


def newtonpf_I_polar(Ybus, Sbus, V0, ref, pv, pq, mpopt=None):
    # Default options
    if mpopt is None:
        mpopt = {
            'pf': {
                'tol': 1e-8,
                'nr': {
                    'max_it': 10,
                    'lin_solver': 'spsolve'
                }
            },
            'verbose': 1
        }

    # Extract options
    tol = mpopt['pf']['tol']
    max_it = mpopt['pf']['nr']['max_it']
    lin_solver = mpopt['pf']['nr']['lin_solver']
    verbose = mpopt['verbose']

    # Initialize
    converged = False
    i = 0
    V = V0.copy()
    Va = np.angle(V)
    Vm = np.abs(V)
    n = len(V0)

    # Set up indexing for updating V
    npv = len(pv)
    npq = len(pq)
    j1 = 0  # j1:j2 - V angle of pv buses
    j2 = j1 + npv
    j3 = j2  # j3:j4 - V angle of pq buses
    j4 = j3 + npq
    j5 = j4  # j5:j6 - Q of pv buses
    j6 = j5 + npv
    j7 = j6  # j7:j8 - V mag of pq buses
    j8 = j7 + npq

    pvpq = np.concatenate([pv, pq])
    # Evaluate initial mismatch
    Sb = Sbus(Vm)
    Sb[pv] = np.real(Sb[pv]) + 1j * np.imag(V[pv] * (Ybus[pv, :] * V))
    mis = Ybus * V - np.conj(Sb / V)
    F = np.concatenate([np.real(mis[pvpq]), np.imag(mis[pvpq])])
    normF = max(abs(F))

    print('\n it   max Ir & Ii mismatch (p.u.)')
    print('\n----  ---------------------------')
    print('\n%3d        %10.3e', i, normF)

    if normF < tol:
        converged = 1
        print('\nConverged!\n')

    # lin_solver = '\''
    while (not converged) & (i < max_it):
        i = i + 1
        dImis_dQ = sparse.csr_matrix(
            (1j / np.conj(V[pv]),  # 非零元素值
             (pv, pv)),  # (行索引, 列索引)
            shape=(n, n)  # 矩阵形状
        )
        dImis_dVa, dImis_dVm = dImis_dV(Sb, Ybus, V)
        dImis_dVm[:, pv] = dImis_dQ[:, pv]

        # 提取子矩阵并分离实部/虚部
        def extract_submatrix(mat, rows, cols):
            """通用子矩阵提取函数，适配稀疏和稠密矩阵"""
            if sparse.issparse(mat):
                return mat[rows, :][:, cols].toarray()  # 转换为稠密以便后续操作
            else:
                return mat[np.ix_(rows, cols)]

        j11 = np.real(extract_submatrix(dImis_dVa, pvpq, pvpq))  # 实部对角度偏导
        j12 = np.real(extract_submatrix(dImis_dVm, pvpq, pvpq))  # 实部对幅值偏导
        j21 = np.imag(extract_submatrix(dImis_dVa, pvpq, pvpq))  # 虚部对角度偏导
        j22 = np.imag(extract_submatrix(dImis_dVm, pvpq, pvpq))  # 虚部对幅值偏导

        J = np.block([
            [j11, j12],
            [j21, j22]
        ])
        print("J", J.shape, "F", F.shape)
        dx = np.linalg.solve(J, -F)

        if npv > 0:
            Va[pv] += dx[j1:j2]  # 更新PV母线角度
            Q_new = np.imag(Sb[pv]) + dx[j5:j6]  # 计算新无功
            Sb[pv] = np.real(Sb[pv]) + 1j * Q_new  # 更新复功率

        if npq > 0:
            Va[pq] += dx[j3:j4]  # 更新PQ母线角度
            Vm[pq] += dx[j7:j8]  # 更新PQ母线幅值

        # 同步复电压并保护数值
        V = Vm * np.exp(1j * Va)
        Vm = np.abs(V)  # 确保幅值非负
        Va = np.angle(V)  # 标准化角度
        # evaluate F(x)
        mis = Ybus * V - np.conj(Sb / V)
        F = np.concatenate([np.real(mis[pvpq]), np.imag(mis[pvpq])])
        normF = max(abs(F))
        print(i, normF)
        if normF < tol:
            print('\nNewton''s method power flow (current balance, polar) converged in ', i, ' iterations.\n')
            converged = 1

    return V, converged, i


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

    ref, pv, pq = bustypes(bus, gen)
    print(ref, pv, pq)
    # generator info

    # 计算功率注入函数
    Sbus = partial(makeSbus, baseMVA, bus, gen, None)

    on = np.where(gen[:, GEN_STATUS] > 0)[0]  # which generators are on?
    gbus = gen[on, GEN_BUS]  # what buses are they at?

    # initial state
    V0 = bus[:, VM] * np.exp(1j * np.pi / 180 * bus[:, VA])
    print("V0", V0)
    print("Sbus", Sbus(V0))
    on = np.where(gen[:, GEN_STATUS] > 0)[0].astype(int)  # 运行中的发电机索引
    gbus = gen[on, GEN_BUS].astype(int) - 1  # 转换为0-based索引
    # 创建电压控制总线掩码
    vcb = np.ones(bus.shape[0], dtype=int)  # 明确指定为整数
    vcb[pq] = 0
    # 安全获取索引
    k = np.where(vcb[gbus] == 1)[0]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)


    # V, converged, i = newtonpf_I_polar(Ybus, Sbus, V0, ref, pv, pq)
    #
    # for i in range(len(V)):
    #     print(i + 1, V[i])

    # Test
    V, success, iterations = newtonpf_I_polar(Ybus, Sbus, V0, ref, pv, pq, None)
    # n = len(V0)
    # V = V0
    # Sb = Sbus(abs(V0))
    # print(Sb)
    # Sb[pv] = np.real(Sb[pv]) + 1j * np.imag(V[pv] * (Ybus[pv, :] * V))
    # print(Sb)
    # mis = Ybus * V - np.conj(Sb / V)
    # print(mis)
    # print((1j / np.conj(V[pv])))
    # dImis_dQ = sparse.csr_matrix(
    #     (1j / np.conj(V[pv]),  # 非零元素值
    #      (pv, pv)),  # (行索引, 列索引)
    #     shape=(n, n)  # 矩阵形状
    # )
    # print(dImis_dQ)
    # pvpq = np.concatenate([pv, pq])
    # F = np.concatenate([np.real(mis[pvpq]), np.imag(mis[pvpq])])
    # print(F)
    # normF = max(abs(F))
    # print(normF)
    #
    # dImis_dQ = sparse.csr_matrix(
    #     (1j / np.conj(V[pv]),  # 非零元素值
    #      (pv, pv)),  # (行索引, 列索引)
    #     shape=(n, n)  # 矩阵形状
    # )
    # dImis_dVa, dImis_dVm = dImis_dV(Sb, Ybus, V)
    # dImis_dVm[:, pv] = dImis_dQ[:, pv]
    #
    #
    # # 提取子矩阵并分离实部/虚部
    # def extract_submatrix(mat, rows, cols):
    #     """通用子矩阵提取函数，适配稀疏和稠密矩阵"""
    #     if sparse.issparse(mat):
    #         return mat[rows, :][:, cols].toarray()  # 转换为稠密以便后续操作
    #     else:
    #         return mat[np.ix_(rows, cols)]
    #
    #
    # j11 = np.real(extract_submatrix(dImis_dVa, pvpq, pvpq))  # 实部对角度偏导
    # j12 = np.real(extract_submatrix(dImis_dVm, pvpq, pvpq))  # 实部对幅值偏导
    # j21 = np.imag(extract_submatrix(dImis_dVa, pvpq, pvpq))  # 虚部对角度偏导
    # j22 = np.imag(extract_submatrix(dImis_dVm, pvpq, pvpq))  # 虚部对幅值偏导
    #
    # J = np.block([
    #     [j11, j12],
    #     [j21, j22]
    # ])
    #
    # x = np.linalg.solve(J, -F)
    #
    # print(x)
