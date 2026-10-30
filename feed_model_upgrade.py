"""
feed_model_upgrade.py -- regulated N2-pressurized fuel feed model, u' = f(u).

Same physics and inputs as feed_model.py, reorganized:
  1. Parameters and initial state
  2. Regulator (closed / droop / wide open)
  3. Algebraic relations: info(u, p)
  4. Right-hand side f(u, p)
  5. Analytic Jacobian J(u, p)  (Sec. 2.2 of the report) + finite-difference check

State (Lecture notation, u):  u = [m_p, T_p, m_u, T_u, m_f]
  m_p, T_p : N2 mass [kg] and temperature [K] in the pressurant tank
  m_u, T_u : N2 mass [kg] and temperature [K] in the fuel-tank ullage
  m_f      : fuel mass remaining [kg]; burnout when m_f = 0
All internal units are SI, absolute pressure.
"""
import numpy as np
from scipy.optimize import brentq

PSI = 6894.757            # Pa per psi
PATM = 14.7*PSI           # atmosphere, Pa
STATE_NAMES = ["m_p", "T_p", "m_u", "T_u", "m_f"]


# ---------------------------------------------------------------------------
# 1. Parameters and initial state
# ---------------------------------------------------------------------------
def make_params(V_P_L=9.0, P_P0_PSIG=3900.0, P_SET_PSIG=360.0, P_U0_PSIG=360.0,
                T_P0_K=300.0, T_U0_K=300.0,
                V_F_L=17.1, V_U0_L=1.8, RHO_F=851.0,
                CDA_P=2.05e-5, CDA_TRUNK_F=8.917e-5, CDA_BR_F=2.700e-5, N_BR=2,
                P_C_PSIG=300.0,
                CV_REG=1.0, CV_REF=1.0, DROOP_SCALES_WITH_CV=False,
                DROOP_PIN_PSIG=(1000.0, 3000.0), DROOP_K_PSI_PER_SCFM=(3.7, 2.0),
                R=296.8, G=1.4):
    """Inputs in psig / L / K (as feed_model.py); returns dict p in SI absolute."""
    scale = CV_REF/CV_REG if DROOP_SCALES_WITH_CV else 1.0
    return dict(
        R=R, G=G, CV=R/(G - 1),                      # gas constant, gamma, c_v
        RC=(2/(G + 1))**(G/(G - 1)),                 # critical (choked) pressure ratio
        SCFM=2118.88/(101325/(R*288.7)),             # scfm per kg/s  -> K' = K*SCFM = K/rho_std
        V_P=V_P_L*1e-3, P_P0=P_P0_PSIG*PSI + PATM, T_P0=T_P0_K,
        P_SET=P_SET_PSIG*PSI + PATM,
        CDA_REG=CV_REG/38.0*6.4516e-4,               # wide-open regulator CdA = Cv/38 in^2
        PIN_TAB=np.array(DROOP_PIN_PSIG)*PSI + PATM, # droop table: inlet pressures
        K_TAB=np.array(DROOP_K_PSI_PER_SCFM)*PSI*scale,  #             droop slopes [Pa/scfm]
        CDA_P=CDA_P,
        RHO_F=RHO_F, V_F=V_F_L*1e-3, V_U0=V_U0_L*1e-3,
        P_U0=P_U0_PSIG*PSI + PATM, T_U0=T_U0_K,
        CDA_F=1/np.sqrt(1/CDA_TRUNK_F**2 + 1/(N_BR*CDA_BR_F)**2), N_BR=N_BR,  # Eq. (5)
        P_C=P_C_PSIG*PSI + PATM,                     # engine (chamber) pressure P_e
    )


def initial_state(p):
    """u_0 from the ideal-gas law and the initial fuel volume, Eq. (11)."""
    m_p0 = p["P_P0"]*p["V_P"]/(p["R"]*p["T_P0"])
    m_u0 = p["P_U0"]*p["V_U0"]/(p["R"]*p["T_U0"])
    m_f0 = (p["V_F"] - p["V_U0"])*p["RHO_F"]
    return np.array([m_p0, p["T_P0"], m_u0, p["T_U0"], m_f0])


# ---------------------------------------------------------------------------
# 2. Regulator
# ---------------------------------------------------------------------------
def droop_K(P_p, p):
    """K' = K(P_p)/rho_std [Pa s/kg] and its slope dK'/dP_p (Eq. 2, clamped outside table)."""
    x, k = p["PIN_TAB"], p["K_TAB"]*p["SCFM"]
    slope = (k[1] - k[0])/(x[1] - x[0]) if x[0] < P_p < x[1] else 0.0
    return np.interp(P_p, x, k), slope


def mdot_orifice(P0, T0, P, CdA, p):
    """Compressible orifice flow from (P0, T0) to P; choked below the critical ratio."""
    G, R = p["G"], p["R"]
    r = max(P/P0, p["RC"])
    return CdA*P0*np.sqrt(2*G/((G - 1)*R*T0)*(r**(2/G) - r**((G + 1)/G)))


def n2_flow(P_p, T_p, P_u, p):
    """Regulator + pressurant line. Returns (mdot_p, P_out, regime)."""
    R = p["R"]
    # closed: downstream already at/above what the regulator can deliver
    P_top = min(p["P_SET"], P_p)
    if P_u >= P_top:
        return 0.0, P_top, "closed"
    # droop: P_out = P_set - K' mdot (Eq. 1) and P_out - P_u = line loss (Eq. 3) -> Eq. (10)
    K, _ = droop_K(P_p, p)
    a = R*T_p/(2*P_u*p["CDA_P"]**2)
    mp = (-K + np.sqrt(K*K + 4*a*(p["P_SET"] - P_u)))/(2*a)
    P_out = p["P_SET"] - K*mp
    if P_out <= P_p and mdot_orifice(P_p, T_p, P_out, p["CDA_REG"], p) >= mp:
        return mp, P_out, "droop"
    # wide open: fixed orifice in series with the line; solve for P_out
    line = lambda Po: p["CDA_P"]*np.sqrt(2*P_u*max(Po - P_u, 0.0)/(R*T_p))
    P_out = brentq(lambda Po: mdot_orifice(P_p, T_p, Po, p["CDA_REG"], p) - line(Po), P_u, P_p)
    return line(P_out), P_out, "wide open"


# ---------------------------------------------------------------------------
# 3. Algebraic relations
# ---------------------------------------------------------------------------
def info(u, p):
    """All intermediate quantities at state u (Eqs. 8a, 8b and the regulator)."""
    m_p, T_p, m_u, T_u, m_f = u
    R = p["R"]
    P_p = m_p*R*T_p/p["V_P"]
    V_u = p["V_F"] - m_f/p["RHO_F"]
    P_u = m_u*R*T_u/V_u
    mp, P_out, regime = n2_flow(P_p, T_p, P_u, p)
    mf = p["CDA_F"]*np.sqrt(2*p["RHO_F"]*max(P_u - p["P_C"], 0.0))
    return dict(P_p=P_p, V_u=V_u, P_u=P_u, P_out=P_out, mdot_p=mp, mdot_f=mf, regime=regime)


def outputs(u, p):
    """(P_p, P_out, P_u, mdot_p, mdot_f) -- same tuple as feed_model.outputs."""
    s = info(u, p)
    return s["P_p"], s["P_out"], s["P_u"], s["mdot_p"], s["mdot_f"]


def mdot_f(u, p):
    """Fuel flow only (used for the error metric)."""
    return info(u, p)["mdot_f"]


# ---------------------------------------------------------------------------
# 4. Right-hand side, Eqs. (7a)-(7e)
# ---------------------------------------------------------------------------
def f(u, p):
    """u' = f(u). Autonomous, so no t argument."""
    m_p, T_p, m_u, T_u, m_f = u
    s = info(u, p)
    G, mp, mf = p["G"], s["mdot_p"], s["mdot_f"]
    return np.array([
        -mp,                                                            # (7a) tank mass
        -(G - 1)*T_p*mp/m_p,                                            # (7b) isentropic tank
        mp,                                                             # (7c) ullage mass
        (mp*(G*T_p - T_u) - s["P_u"]*(mf/p["RHO_F"])/p["CV"])/m_u,      # (7d) ullage energy
        -mf,                                                            # (7e) fuel mass
    ])


ydot = f   # alias so old code calling ydot(y, p) still works


# ---------------------------------------------------------------------------
# 5. Jacobian J = df/du
# ---------------------------------------------------------------------------
def jacobian(u, p):
    """Analytic Jacobian, Eqs. (13)-(17) of the report (chain rule through P_p, P_u, mdot_f, mdot_p)."""
    m_p, T_p, m_u, T_u, m_f = u
    s = info(u, p)
    G, R, RHO, CV = p["G"], p["R"], p["RHO_F"], p["CV"]
    P_p, P_u, V_u, mp, mf = s["P_p"], s["P_u"], s["V_u"], s["mdot_p"], s["mdot_f"]
    e = np.eye(5)

    # gradients of the pressures, Eqs. (dPp), (dPu)
    dPp = np.array([P_p/m_p, P_p/T_p, 0.0, 0.0, 0.0])
    dPu = np.array([0.0, 0.0, P_u/m_u, P_u/T_u, P_u/(RHO*V_u)])

    # fuel flow, Eq. (dmf): d mdot_f / d P_u = mdot_f / (2 (P_u - P_e))
    dmf_dPu = mf/(2*(P_u - p["P_C"])) if P_u > p["P_C"] else 0.0
    dmf = dmf_dPu*dPu

    # N2 flow, by regime
    if s["regime"] == "closed":
        dmp = np.zeros(5)                                  # mdot_p = 0 identically
    elif s["regime"] == "droop":
        # implicit differentiation of Eq. (9): F(mdot_p; P_u, T_p, P_p) = 0, Eq. (dmp)
        K, dK = droop_K(P_p, p)
        dpl = mp**2*R*T_p/(2*P_u*p["CDA_P"]**2)            # line loss
        D = K + 2*dpl/mp                                   # -dF/d mdot_p
        dmp = -((1 - dpl/P_u)*dPu + (dpl/T_p)*e[1] + mp*dK*dPp)/D
    else:
        # wide open: not reached on the nominal burn; fall back to finite differences
        return jacobian_fd(u, p)

    # rows of J, Eqs. (J1)-(J5)
    f4 = (mp*(G*T_p - T_u) - P_u*mf/(RHO*CV))/m_u
    J = np.empty((5, 5))
    J[0] = -dmp
    J[1] = -(G - 1)*(T_p/m_p*dmp + mp/m_p*e[1] - T_p*mp/m_p**2*e[0])
    J[2] = dmp
    J[3] = ((G*T_p - T_u)*dmp + mp*(G*e[1] - e[3])
            - (mf + P_u*dmf_dPu)/(RHO*CV)*dPu - f4*e[2])/m_u
    J[4] = -dmf
    return J


def jacobian_fd(u, p, rel=1e-6):
    """Central-difference Jacobian -- only to check jacobian()."""
    J = np.empty((5, 5))
    for j in range(5):
        h = rel*abs(u[j])
        up, um = u.copy(), u.copy()
        up[j] += h
        um[j] -= h
        J[:, j] = (f(up, p) - f(um, p))/(2*h)
    return J


if __name__ == "__main__":
    # quick self-check at the initial state and slightly into the burn
    p = make_params()
    u = initial_state(p)
    print("u0 =", u, "| regime:", info(u, p)["regime"])
    u1 = u + 0.01*f(u, p)
    rel = np.max(np.abs(jacobian(u1, p) - jacobian_fd(u1, p)))/np.max(np.abs(jacobian_fd(u1, p)))
    print(f"analytic vs FD Jacobian, max rel. diff = {rel:.1e} | regime: {info(u1, p)['regime']}")
