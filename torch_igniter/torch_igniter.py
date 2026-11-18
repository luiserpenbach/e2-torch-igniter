#!/usr/bin/env python3
"""
torch_igniter.py -- sizing and off-design model for a single-doublet N2O / ethanol torch igniter.

What it does
  * Orifice flow: ethanol as incompressible liquid (SPI); N2O with SPI, HEM (isentropic,
    homogeneous equilibrium, choked) or the Dyer blend of the two.
  * Operating point: solves chamber pressure for a fixed throat from the injector flows and
    c*(O/F) x eta. Works for any feed pressures, N2O temperature, Cd and efficiency.
  * Sizing: throat for a target pc, chamber volume / L* from the head cone, cylinder and
    convergent lengths.
  * Doublet: momentum ratio J (SPI..HEM velocity bound), spray resultant angle, balanced
    jet angles, free-jet l/d.
  * Test evaluation: from measured pc, injector pressures, N2O temperature and flows,
    back-calculates O/F, c*, c* efficiency and the implied N2O Cd for each flow model.

c*(O/F) and Tc(O/F) come from rocketcea if it is installed, otherwise from the table in
CEA_TABLE (equilibrium, gaseous N2O / liquid ethanol, from the V1 design report). Replace the
table with an h2cea run when available.

Use it as a library (see app.py) or run it:  python torch_igniter.py
Needs numpy, scipy and CoolProp.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache

import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import brentq

N2O, ETOH = "NitrousOxide", "Ethanol"
MODELS = ("spi", "dyer", "hem")

# O/F, c* [m/s], Tc [K] -- equilibrium, eta = 1 (V1 report; O/F 1.0 and 4.0 extrapolated, EST)
CEA_TABLE = np.array([
    [1.0, 1080.0, 1000.0],
    [1.6, 1245.0, 1466.0],
    [2.0, 1355.0, 1894.0],
    [2.2, 1400.0, 2083.0],
    [3.1, 1590.0, 3160.0],
    [4.0, 1610.0, 3200.0],
])


# =========================================================================== inputs
@dataclass(frozen=True)
class Chamber:
    D_c: float = 12e-3        # m, chamber ID
    L_head: float = 5.6e-3    # m, head cone length (45 deg); 0 = flat top
    head_angle: float = 45.0  # deg from axis
    L_cyl: float = 37e-3      # m
    L_conv: float = 3e-3      # m, convergent length (angle follows from throat)


@dataclass(frozen=True)
class Igniter:
    d_o: float = 0.40e-3      # m, N2O orifice
    d_f: float = 0.30e-3      # m, ethanol orifice
    cd_o: float = 0.80        # EST until cold flow
    cd_f: float = 0.80
    alpha_o: float = 30.0     # deg from chamber axis
    alpha_f: float = 60.0
    l_free: float = 10e-3     # m, orifice exit to impingement
    d_t: float | None = None  # m, throat; None = size for pc_target at the nominal point
    chamber: Chamber = field(default_factory=Chamber)


@dataclass(frozen=True)
class Point:
    """Operating conditions at the injector inlets."""
    p_ox: float = 95e5        # Pa
    p_fu: float = 95e5        # Pa
    T_ox: float = 293.15      # K
    T_fu: float = 293.15      # K
    eta: float = 0.80         # c* efficiency
    model: str = "dyer"       # N2O flow model


PC_TARGET = 27.5e5            # Pa, 1.1 x 25 bar (H1B-ENG-IGN-003)


# =========================================================================== thermochemistry
try:
    from rocketcea.cea_obj_w_units import CEA_Obj  # type: ignore
    _CEA = CEA_Obj(oxName="N2O", fuelName="Ethanol", cstar_units="m/s", temperature_units="degK",
                   pressure_units="bar")
except Exception:  # rocketcea not installed or failed to build
    _CEA = None


def cea(of: float, pc: float = 27.5e5):
    """Ideal c* [m/s] and Tc [K] at O/F (rocketcea if available, else CEA_TABLE)."""
    if _CEA is not None:
        return float(_CEA.get_Cstar(Pc=pc / 1e5, MR=of)), float(_CEA.get_Tcomb(Pc=pc / 1e5, MR=of))
    t = CEA_TABLE
    return float(np.interp(of, t[:, 0], t[:, 1])), float(np.interp(of, t[:, 0], t[:, 2]))


# =========================================================================== fluids
@lru_cache(maxsize=None)
def n2o_state(p1: float, T1: float):
    """rho, h, s and vapour pressure of N2O at the injector inlet."""
    return (CP.PropsSI("D", "P", p1, "T", T1, N2O), CP.PropsSI("H", "P", p1, "T", T1, N2O),
            CP.PropsSI("S", "P", p1, "T", T1, N2O), CP.PropsSI("P", "T", T1, "Q", 0, N2O))


@lru_cache(maxsize=None)
def _hd_ps(p: float, s: float):
    try:
        return CP.PropsSI("H", "P", p, "S", s, N2O), CP.PropsSI("D", "P", p, "S", s, N2O)
    except ValueError:
        return None


@lru_cache(maxsize=None)
def _n2o_flux(p1: float, T1: float, p2: float, model: str) -> float:
    rho1, h1, s1, psat = n2o_state(p1, T1)
    g_spi = math.sqrt(2 * rho1 * max(p1 - p2, 0.0))
    if model == "spi":
        return g_spi
    g_hem = 0.0
    for p in np.linspace(p2, 0.995 * p1, 60):       # choked HEM: max over throat pressure
        st = _hd_ps(round(p, -3), s1)
        if st is not None:
            g_hem = max(g_hem, st[1] * math.sqrt(max(2 * (h1 - st[0]), 0.0)))
    if model == "hem":
        return g_hem
    if psat <= p2:
        return g_spi
    k = math.sqrt((p1 - p2) / (psat - p2))
    return k / (1 + k) * g_spi + 1 / (1 + k) * g_hem


def n2o_flux(p1, T1, p2, model):
    """Ideal N2O mass flux [kg/m2 s]; multiply by Cd*A. Inputs rounded for caching."""
    return _n2o_flux(round(p1, -3), round(T1, 2), round(p2, -3), model)


def n2o_velocity(p1, T1, p2):
    """N2O jet velocity bounds: liquid (SPI) and isentropic equilibrium expansion to p2 (HEM):
    v_HEM = sqrt(2 (h(p1,T1) - h(p2, s1)))."""
    rho1, h1, s1, _ = n2o_state(round(p1, -3), round(T1, 2))
    v_spi = math.sqrt(2 * max(p1 - p2, 0.0) / rho1)
    for dp in (0.0, 1e3, 5e3):
        st = _hd_ps(round(p2, -3) + dp, s1)
        if st is not None:
            return v_spi, math.sqrt(max(2 * (h1 - st[0]), 0.0))
    return v_spi, v_spi


@lru_cache(maxsize=None)
def _etoh_rho(p, T):
    return CP.PropsSI("D", "P", p, "T", T, ETOH)


def etoh_rho(p, T):
    return _etoh_rho(round(p, -3), round(T, 2))


def area(d):
    return math.pi * d * d / 4


def flows(ig: Igniter, pt: Point, pc: float, model: str | None = None):
    """N2O and ethanol mass flow [kg/s] at chamber pressure pc."""
    m = pt.model if model is None else model
    mo = ig.cd_o * area(ig.d_o) * n2o_flux(pt.p_ox, pt.T_ox, pc, m)
    mf = ig.cd_f * area(ig.d_f) * math.sqrt(2 * etoh_rho(pt.p_fu, pt.T_fu) * max(pt.p_fu - pc, 0.0))
    return mo, mf


# =========================================================================== geometry
def geometry(ig: Igniter, d_t: float) -> dict:
    ch = ig.chamber
    R, r_t = ch.D_c / 2, d_t / 2
    r_top = max(R - ch.L_head * math.tan(math.radians(ch.head_angle)), 0.0)
    fr = lambda h, a, b: math.pi / 3 * h * (a * a + a * b + b * b)
    V = fr(ch.L_head, R, r_top) + math.pi * R * R * ch.L_cyl + fr(ch.L_conv, R, r_t)
    A_t = area(d_t)
    return dict(R=R, r_t=r_t, r_top=r_top, V_c=V, A_t=A_t, L_star=V / A_t, CR=(R / r_t) ** 2,
                L_total=ch.L_head + ch.L_cyl + ch.L_conv,
                conv_angle=math.degrees(math.atan((R - r_t) / ch.L_conv)) if ch.L_conv > 0 else 90.0)


# =========================================================================== doublet
def resultant_angle(J, a_o, a_f):
    """Spray resultant from axis [deg], + = towards the fuel orifice side wall."""
    a_o, a_f = np.radians(a_o), np.radians(a_f)
    return np.degrees(np.arctan2(J * np.sin(a_o) - np.sin(a_f), J * np.cos(a_o) + np.cos(a_f)))


def balanced_angles(J, included):
    th = np.radians(included)
    a_o = np.arctan2(np.sin(th), J + np.cos(th))
    return np.degrees(a_o), included - np.degrees(a_o)


def momentum_ratio(pt: Point, pc: float, mo: float, mf: float):
    """J = (m v)_ox / (m v)_fu with the N2O velocity at its SPI and HEM bound."""
    v_f = math.sqrt(2 * max(pt.p_fu - pc, 0.0) / etoh_rho(pt.p_fu, pt.T_fu))
    v_spi, v_hem = n2o_velocity(pt.p_ox, pt.T_ox, pc)
    return mo * v_spi / (mf * v_f), mo * v_hem / (mf * v_f), v_spi, v_hem, v_f


# =========================================================================== operating point
def size_throat(ig: Igniter, pt: Point, pc: float = PC_TARGET) -> float:
    mo, mf = flows(ig, pt, pc)
    cstar = cea(mo / mf, pc)[0] * pt.eta
    return math.sqrt(4 * (mo + mf) * cstar / pc / math.pi)


def operate(ig: Igniter, pt: Point, d_t: float | None = None) -> dict:
    """Chamber pressure and flows for a fixed throat at the given inlet conditions."""
    d_t = ig.d_t if d_t is None else d_t
    if d_t is None:
        d_t = size_throat(ig, pt)
    A_t = area(d_t)

    def f(pc):
        mo, mf = flows(ig, pt, pc)
        return (mo + mf) * cea(mo / mf, pc)[0] * pt.eta / A_t - pc

    p_hi = min(pt.p_ox, pt.p_fu) - 2e3
    pc = brentq(f, 1.5e5, p_hi, xtol=500.0) if f(p_hi) < 0 else p_hi
    mo, mf = flows(ig, pt, pc)
    of = mo / mf
    cstar, Tc = cea(of, pc)
    J_spi, J_hem, v_spi, v_hem, v_f = momentum_ratio(pt, pc, mo, mf)
    J_mid = math.sqrt(J_spi * J_hem)
    psat = n2o_state(round(pt.p_ox, -3), round(pt.T_ox, 2))[3]
    return dict(pc=pc, mo=mo, mf=mf, mdot=mo + mf, of=of, cstar_ideal=cstar, cstar=cstar * pt.eta, Tc=Tc,
                d_t=d_t, A_t=A_t, J=(J_spi, J_mid, J_hem), beta=float(resultant_angle(J_mid, ig.alpha_o, ig.alpha_f)),
                beta_band=tuple(float(resultant_angle(j, ig.alpha_o, ig.alpha_f)) for j in (J_spi, J_hem)),
                v_o=(v_spi, v_hem), v_f=v_f, dp_ox=pt.p_ox - pc, dp_fu=pt.p_fu - pc, psat=psat,
                ld_o=ig.l_free / ig.d_o, ld_f=ig.l_free / ig.d_f)


def sweep(ig: Igniter, pt: Point, d_t: float, name: str, values) -> list[dict]:
    """Off-design sweep of one Point field (p_ox, p_fu, p_feed (both), T_ox, eta, model)."""
    out = []
    for v in values:
        if name == "p_feed":
            q = replace(pt, p_ox=v, p_fu=v)
        else:
            q = replace(pt, **{name: v})
        r = operate(ig, q, d_t)
        r["x"] = v
        out.append(r)
    return out


# =========================================================================== test evaluation
def evaluate_test(ig: Igniter, d_t: float, pc: float, p_ox: float, p_fu: float, T_ox: float,
                  mdot_o: float | None = None, mdot_f: float | None = None, T_fu: float = 293.15,
                  eta_nominal: float = 0.80, model: str = "dyer") -> dict:
    """Compare a measured run with the model.
    Measured: pc, injector inlet pressures, N2O temperature, optional flows.
    Missing flows are estimated from the orifice equations at the measured pc."""
    pt = Point(p_ox=p_ox, p_fu=p_fu, T_ox=T_ox, T_fu=T_fu, eta=eta_nominal, model=model)
    pred = operate(ig, pt, d_t)
    mo_model, mf_model = flows(ig, pt, pc)
    mo = mdot_o if mdot_o else mo_model
    mf = mdot_f if mdot_f else mf_model
    of = mo / mf
    cstar_meas = pc * area(d_t) / (mo + mf)
    cstar_ideal = cea(of, pc)[0]
    cd_o_impl = {m: (mdot_o / (area(ig.d_o) * n2o_flux(p_ox, T_ox, pc, m)) if mdot_o else None) for m in MODELS}
    cd_f_impl = (mdot_f / (area(ig.d_f) * math.sqrt(2 * etoh_rho(p_fu, T_fu) * (p_fu - pc)))) if mdot_f else None
    J_spi, J_hem, *_ = momentum_ratio(pt, pc, mo, mf)
    return dict(pred=pred, pc=pc, mo=mo, mf=mf, of=of, mo_est=mdot_o is None, mf_est=mdot_f is None,
                cstar=cstar_meas, cstar_ideal=cstar_ideal, eta=cstar_meas / cstar_ideal,
                cd_o_implied=cd_o_impl, cd_f_implied=cd_f_impl, J=(J_spi, J_hem),
                beta=float(resultant_angle(math.sqrt(J_spi * J_hem), ig.alpha_o, ig.alpha_f)))


# =========================================================================== CLI
def main():
    ig, pt = Igniter(), Point()
    d_t = size_throat(ig, pt)
    r = operate(ig, pt, d_t)
    g = geometry(ig, d_t)
    print(f"Design point ({pt.model}, eta {pt.eta}, feed {pt.p_ox/1e5:.0f}/{pt.p_fu/1e5:.0f} bar, N2O {pt.T_ox-273.15:.0f} C)")
    print(f"  pc {r['pc']/1e5:.1f} bar  mdot {r['mdot']*1e3:.1f} g/s (ox {r['mo']*1e3:.2f}, fu {r['mf']*1e3:.2f})  O/F {r['of']:.2f}")
    print(f"  throat {d_t*1e3:.2f} mm  CR {g['CR']:.1f}  L* {g['L_star']:.2f} m  V_c {g['V_c']*1e6:.2f} cm3  "
          f"length {g['L_total']*1e3:.1f} mm  conv {g['conv_angle']:.0f} deg")
    print(f"  c* {r['cstar']:.0f} m/s (ideal {r['cstar_ideal']:.0f})  Tc {r['Tc']:.0f} K")
    print(f"  J {r['J'][0]:.2f}..{r['J'][2]:.2f}  beta {r['beta']:+.1f} deg  l/d {r['ld_o']:.0f}/{r['ld_f']:.0f}")
    print("\nN2O temperature sweep (fixed throat):")
    for s in sweep(ig, pt, d_t, "T_ox", [278.15, 288.15, 293.15, 298.15, 303.15]):
        print(f"  {s['x']-273.15:4.0f} C  pc {s['pc']/1e5:5.1f} bar  O/F {s['of']:.2f}  beta {s['beta']:+.1f} deg")


if __name__ == "__main__":
    main()
