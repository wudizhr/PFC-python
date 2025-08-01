# Defines constants for named column indices to ``branch`` matrix.
F_BUS = 1-1  # f, from bus number
T_BUS = 2-1  # t, to bus number
BR_R = 3-1  # r, resistance (p.u.)
BR_X = 4-1  # x, reactance (p.u.)
BR_B = 5-1  # b, total line charging susceptance (p.u.)
RATE_A = 6-1  # rateA, MVA rating A (long term rating)
RATE_B = 7-1  # rateB, MVA rating B (short term rating)
RATE_C = 8-1  # rateC, MVA rating C (emergency rating)
TAP = 9-1  # ratio, transformer off nominal turns ratio
SHIFT = 10-1  # angle, transformer phase shift angle (degrees)
BR_STATUS = 11-1  # initial branch status, 1 - in service, 0 - out of service
ANGMIN = 12-1  # minimum angle difference, angle(Vf) - angle(Vt) (degrees)
ANGMAX = 13-1  # maximum angle difference, angle(Vf) - angle(Vt) (degrees)
# included in power flow solution, not necessarily in input
PF = 14-1  # real power injected at "from" bus end (MW)       (not in PTI format)
QF = 15-1  # reactive power injected at "from" bus end (MVAr) (not in PTI format)
PT = 16-1  # real power injected at "to" bus end (MW)         (not in PTI format)
QT = 17-1  # reactive power injected at "to" bus end (MVAr)   (not in PTI format)
# included in opf solution, not necessarily in input
# assume objective function has units, u
MU_SF = 18-1  # Kuhn-Tucker multiplier on MVA limit at "from" bus (u/MVA)
MU_ST = 19-1  # Kuhn-Tucker multiplier on MVA limit at "to" bus (u/MVA)
MU_ANGMIN = 20-1  # Kuhn-Tucker multiplier lower angle difference limit (u/degree)
MU_ANGMAX = 21-1  # Kuhn-Tucker multiplier upper angle difference limit (u/degree)

# idx_bus - Defines constants for named column indices to ``bus`` matrix.
# define bus types
PQ = 1
PV = 2
REF = 3
NONE = 4

# define the indices
BUS_I = 1-1  # bus number (1 to 29997)
BUS_TYPE = 2-1  # bus type (1 - PQ bus, 2 - PV bus, 3 - reference bus, 4 - isolated bus)
PD = 3-1  # Pd, real power demand (MW)
QD = 4-1  # Qd, reactive power demand (MVAr)
GS = 5-1  # Gs, shunt conductance (MW at V = 1.0 p.u.)
BS = 6-1  # Bs, shunt susceptance (MVAr at V = 1.0 p.u.)
BUS_AREA = 7-1  # area number, 1-100
VM = 8-1  # Vm, voltage magnitude (p.u.)
VA = 9-1  # Va, voltage angle (degrees)
BASE_KV = 10-1  # baseKV, base voltage (kV)
ZONE = 11-1  # zone, loss zone (1-999)
VMAX = 12-1  # maxVm, maximum voltage magnitude (p.u.)      (not in PTI format)
VMIN = 13-1  # minVm, minimum voltage magnitude (p.u.)      (not in PTI format)

# included in opf solution, not necessarily in input
# assume objective function has units, u
LAM_P = 14-1  # Lagrange multiplier on real power mismatch (u/MW)
LAM_Q = 15-1  # Lagrange multiplier on reactive power mismatch (u/MVAr)
MU_VMAX = 16-1  # Kuhn-Tucker multiplier on upper voltage limit (u/p.u.)
MU_VMIN = 17-1  # Kuhn-Tucker multiplier on lower voltage limit (u/p.u.)

# idx_gen - Defines constants for named column indices to ``gen`` matrix.
# define the indices
GEN_BUS = 1-1  # bus number
PG = 2-1  # Pg, real power output (MW)
QG = 3-1  # Qg, reactive power output (MVAr)
QMAX = 4-1  # Qmax, maximum reactive power output at Pmin (MVAr)
QMIN = 5-1  # Qmin, minimum reactive power output at Pmin (MVAr)
VG = 6-1  # Vg, voltage magnitude setpoint (p.u.)
MBASE = 7-1  # mBase, total MVA base of this machine, defaults to baseMVA
GEN_STATUS = 8-1  # status, 1 - machine in service, 0 - machine out of service
PMAX = 9-1  # Pmax, maximum real power output (MW)
PMIN = 10-1  # Pmin, minimum real power output (MW)
PC1 = 11-1  # Pc1, lower real power output of PQ capability curve (MW)
PC2 = 12-1  # Pc2, upper real power output of PQ capability curve (MW)
QC1MIN = 13-1  # Qc1min, minimum reactive power output at Pc1 (MVAr)
QC1MAX = 14-1  # Qc1max, maximum reactive power output at Pc1 (MVAr)
QC2MIN = 15-1  # Qc2min, minimum reactive power output at Pc2 (MVAr)
QC2MAX = 16-1  # Qc2max, maximum reactive power output at Pc2 (MVAr)
RAMP_AGC = 17-1  # ramp rate for load following/AGC (MW/min)
RAMP_10 = 18-1  # ramp rate for 10 minute reserves (MW)
RAMP_30 = 19-1  # ramp rate for 30 minute reserves (MW)
RAMP_Q = 20-1  # ramp rate for reactive power (2 sec timescale) (MVAr/min)
APF = 21-1  # area participation factor

# included in opf solution, not necessarily in input
# assume objective function has units, u
MU_PMAX = 22-1  # Kuhn-Tucker multiplier on upper Pg limit (u/MW)
MU_PMIN = 23-1  # Kuhn-Tucker multiplier on lower Pg limit (u/MW)
MU_QMAX = 24-1  # Kuhn-Tucker multiplier on upper Qg limit (u/MVAr)
MU_QMIN = 25-1  # Kuhn-Tucker multiplier on lower Qg limit (u/MVAr)
