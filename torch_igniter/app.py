"""
Torch igniter explorer -- Streamlit front end for torch_igniter.py.

    streamlit run app.py

Tabs
  Design      nominal point, throat, chamber volume / L*, doublet geometry
  Off-design  sweep one input (feed pressure, N2O temperature, c* efficiency, ...)
  Test data   enter or upload measured runs and compare them with the model
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import torch_igniter as ti

st.set_page_config(page_title="Torch igniter", layout="wide")

INK, MUTED, GRID = "#111111", "#8a8f98", "#ececec"
COL = {"spi": "#2b6cb0", "dyer": "#111111", "hem": "#dd6b20", "ox": "#2b6cb0", "fu": "#dd6b20"}


def style(fig, h=420):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=50, r=20, t=40, b=40),
                      font=dict(size=12, color=INK), legend=dict(orientation="h", y=-0.18),
                      paper_bgcolor="white", plot_bgcolor="white")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#d0d0d0")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#d0d0d0")
    return fig


# =========================================================================== sidebar
sb = st.sidebar
sb.markdown("### Igniter")
c1, c2 = sb.columns(2)
d_o = c1.number_input("N2O orifice [mm]", 0.1, 2.0, 0.40, 0.05, format="%.2f")
d_f = c2.number_input("EtOH orifice [mm]", 0.1, 2.0, 0.30, 0.05, format="%.2f")
cd_o = c1.number_input("Cd N2O", 0.3, 1.2, 0.80, 0.02)
cd_f = c2.number_input("Cd EtOH", 0.3, 1.2, 0.80, 0.02)
a_o = c1.number_input("N2O jet angle [deg]", 0.0, 90.0, 30.0, 1.0)
a_f = c2.number_input("EtOH jet angle [deg]", 0.0, 90.0, 60.0, 1.0)
l_free = c1.number_input("Free jet length [mm]", 1.0, 30.0, 10.0, 0.5)
z_imp = c2.number_input("Impingement z [mm]", 0.0, 40.0, 13.2, 0.5, help="From the top of the chamber, for the sketch")

with sb.expander("Chamber", expanded=False):
    D_c = st.number_input("Chamber ID [mm]", 4.0, 40.0, 12.0, 0.5)
    L_head = st.number_input("Head cone length [mm] (45 deg)", 0.0, 20.0, 5.6, 0.1)
    L_cyl = st.number_input("Cylinder length [mm]", 0.0, 200.0, 37.0, 0.5)
    L_conv = st.number_input("Convergent length [mm]", 0.5, 30.0, 3.0, 0.1)

sb.markdown("### Nominal point")
c1, c2 = sb.columns(2)
p_ox = c1.number_input("N2O feed [bar]", 20.0, 150.0, 95.0, 1.0)
p_fu = c2.number_input("EtOH feed [bar]", 20.0, 150.0, 95.0, 1.0)
T_ox = c1.number_input("N2O temp [C]", -20.0, 35.0, 20.0, 1.0)
eta = c2.number_input("c* efficiency", 0.4, 1.0, 0.80, 0.01)
model = sb.selectbox("N2O flow model", ti.MODELS, index=1,
                     help="spi: liquid, no flashing. hem: equilibrium flashing, choked. dyer: blend weighted by the bubble-growth ratio.")
pc_target = sb.number_input("Target pc for throat sizing [bar]", 5.0, 60.0, 27.5, 0.5)
throat_mode = sb.radio("Throat", ["Size for target pc", "Fixed"], horizontal=True)
d_t_fixed = sb.number_input("Fixed throat [mm]", 0.5, 10.0, 2.76, 0.01, format="%.2f",
                            disabled=throat_mode != "Fixed")

ig = ti.Igniter(d_o=d_o * 1e-3, d_f=d_f * 1e-3, cd_o=cd_o, cd_f=cd_f, alpha_o=a_o, alpha_f=a_f,
                l_free=l_free * 1e-3,
                chamber=ti.Chamber(D_c=D_c * 1e-3, L_head=L_head * 1e-3, L_cyl=L_cyl * 1e-3, L_conv=L_conv * 1e-3))
pt = ti.Point(p_ox=p_ox * 1e5, p_fu=p_fu * 1e5, T_ox=T_ox + 273.15, eta=eta, model=model)


@st.cache_data(show_spinner=False)
def c_size(ig, pt, pc):
    return ti.size_throat(ig, pt, pc)


@st.cache_data(show_spinner=False)
def c_operate(ig, pt, d_t):
    return ti.operate(ig, pt, d_t)


@st.cache_data(show_spinner="Sweeping ...")
def c_sweep(ig, pt, d_t, name, values):
    return ti.sweep(ig, pt, d_t, name, list(values))


d_t = d_t_fixed * 1e-3 if throat_mode == "Fixed" else c_size(ig, pt, pc_target * 1e5)
r = c_operate(ig, pt, d_t)
g = ti.geometry(ig, d_t)

st.markdown("## Torch igniter")
st.caption(f"Single unlike doublet, N2O / ethanol. c*(O/F) from {'rocketcea' if ti._CEA else 'the CEA table in torch_igniter.py'}.")
tab_d, tab_o, tab_t = st.tabs(["Design", "Off-design", "Test data"])

# =========================================================================== design
with tab_d:
    m = st.columns(6)
    m[0].metric("Chamber pressure", f"{r['pc'] / 1e5:.1f} bar")
    m[1].metric("Total flow", f"{r['mdot'] * 1e3:.1f} g/s")
    m[2].metric("O/F", f"{r['of']:.2f}")
    m[3].metric("Throat", f"{d_t * 1e3:.2f} mm")
    m[4].metric("L*", f"{g['L_star']:.2f} m")
    m[5].metric("Spray resultant", f"{r['beta']:+.1f} deg", help="+ = towards the fuel orifice side")

    left, right = st.columns([1.3, 1])
    with left:
        R = g["R"] * 1e3
        z = [0, 0, L_head, L_head + L_cyl, L_head + L_cyl + L_conv, L_head + L_cyl + L_conv + 3]
        rr = [0, g["r_top"] * 1e3, R, R, g["r_t"] * 1e3, g["r_t"] * 1e3]
        fig = go.Figure()
        for s in (1, -1):
            fig.add_scatter(x=z, y=[s * v for v in rr], mode="lines", line=dict(color="#555", width=2),
                            showlegend=False, hoverinfo="skip")
        for lab, ang, side, col in (("N2O", a_o, -1, COL["ox"]), ("EtOH", a_f, 1, COL["fu"])):
            x0 = side * l_free * math.sin(math.radians(ang))
            z0 = z_imp - l_free * math.cos(math.radians(ang))
            fig.add_scatter(x=[z0, z_imp], y=[x0, 0], mode="lines+markers", name=f"{lab} jet {ang:.0f} deg",
                            line=dict(color=col, width=3), marker=dict(size=[7, 0]))
        b0, b1 = r["beta_band"]
        for b, dash in ((r["beta"], "solid"), (b0, "dot"), (b1, "dot")):
            L = (L_head + L_cyl + L_conv) - z_imp
            fig.add_scatter(x=[z_imp, z_imp + L], y=[0, L * math.tan(math.radians(b))], mode="lines",
                            line=dict(color=INK, width=1.5 if dash == "solid" else 1, dash=dash),
                            showlegend=dash == "solid", name="spray resultant (SPI..HEM band dotted)")
        fig.update_yaxes(scaleanchor="x", scaleratio=1, title="x [mm] (+ fuel side)")
        fig.update_xaxes(title="z from top [mm]")
        st.plotly_chart(style(fig, 380), width="stretch")

        st.markdown("**Balanced angles for the current J**")
        cc = st.columns(3)
        for col_, th in zip(cc, (60, 90, 120)):
            ao_, af_ = ti.balanced_angles(r["J"][1], th)
            col_.markdown(f"{th} deg included: **{float(ao_):.1f} / {float(af_):.1f} deg**")

    with right:
        rows = [
            ("N2O / EtOH flow", f"{r['mo'] * 1e3:.2f} / {r['mf'] * 1e3:.2f} g/s"),
            ("Injector dp N2O / EtOH", f"{r['dp_ox'] / 1e5:.1f} / {r['dp_fu'] / 1e5:.1f} bar"),
            ("dp / pc", f"{r['dp_ox'] / r['pc']:.2f}"),
            ("N2O vapour pressure / pc", f"{r['psat'] / 1e5:.1f} bar / {r['psat'] / r['pc']:.2f}"),
            ("c* ideal / used", f"{r['cstar_ideal']:.0f} / {r['cstar']:.0f} m/s"),
            ("Tc (ideal)", f"{r['Tc']:.0f} K"),
            ("Jet velocity N2O SPI / HEM", f"{r['v_o'][0]:.0f} / {r['v_o'][1]:.0f} m/s"),
            ("Jet velocity EtOH", f"{r['v_f']:.0f} m/s"),
            ("Momentum ratio J (SPI / mid / HEM)", " / ".join(f"{j:.2f}" for j in r["J"])),
            ("Resultant over J band", f"{r['beta_band'][0]:+.1f} .. {r['beta_band'][1]:+.1f} deg"),
            ("Free jet l/d N2O / EtOH", f"{r['ld_o']:.0f} / {r['ld_f']:.0f}"),
            ("Contraction ratio", f"{g['CR']:.1f}"),
            ("Chamber volume", f"{g['V_c'] * 1e6:.2f} cm3"),
            ("Total length / convergent angle", f"{g['L_total'] * 1e3:.1f} mm / {g['conv_angle']:.0f} deg"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Quantity", "Value"]), hide_index=True, width="stretch")
        flags = []
        if r["psat"] > r["pc"]:
            flags.append("N2O flashes after the orifice (vapour pressure above pc).")
        if r["pc"] < 27.4e5:
            flags.append("pc below 27.5 bar (1.1 x main chamber, H1B-ENG-IGN-003).")
        if r["of"] > 3:
            flags.append("O/F above 3.")
        if abs(r["beta"]) > 5:
            flags.append("Spray resultant more than 5 deg off axis: re-balance the jet angles.")
        if max(r["ld_o"], r["ld_f"]) > 7:
            flags.append("Free jet l/d above 7: impingement sensitive to alignment.")
        for f in flags:
            st.warning(f, icon=None)

# =========================================================================== off-design
with tab_o:
    st.markdown(f"Fixed throat **{d_t * 1e3:.2f} mm**, all other inputs from the sidebar.")
    params = {
        "Feed pressure, both [bar]": ("p_feed", 1e5, (60.0, 110.0)),
        "N2O feed pressure [bar]": ("p_ox", 1e5, (60.0, 110.0)),
        "EtOH feed pressure [bar]": ("p_fu", 1e5, (60.0, 110.0)),
        "N2O temperature [C]": ("T_ox", None, (0.0, 32.0)),
        "c* efficiency [-]": ("eta", 1.0, (0.55, 1.0)),
    }
    c1, c2, c3 = st.columns([2, 2, 1])
    pname = c1.selectbox("Sweep", list(params))
    key, scale, (lo, hi) = params[pname]
    rng = c2.slider("Range", lo, hi, (lo, hi))
    n = c3.number_input("Points", 5, 40, 15)
    all_models = st.checkbox("Overlay all three N2O flow models", value=True)
    xs = np.linspace(*rng, int(n))
    xv = xs + 273.15 if key == "T_ox" else xs * scale

    models = ti.MODELS if all_models else (model,)
    res = {mm: c_sweep(ig, replace(pt, model=mm), d_t, key, tuple(xv)) for mm in models}

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Chamber pressure [bar]", "O/F [-]",
                                                        "Total flow [g/s]", "Spray resultant [deg]"),
                        horizontal_spacing=0.09, vertical_spacing=0.16)
    for mm, rs in res.items():
        kw = dict(mode="lines+markers", line=dict(color=COL[mm], width=2.5 if mm == model else 1.5),
                  marker=dict(size=4), legendgroup=mm)
        fig.add_scatter(x=xs, y=[q["pc"] / 1e5 for q in rs], name=mm.upper(), row=1, col=1, **kw)
        fig.add_scatter(x=xs, y=[q["of"] for q in rs], showlegend=False, row=1, col=2, **kw)
        fig.add_scatter(x=xs, y=[q["mdot"] * 1e3 for q in rs], showlegend=False, row=2, col=1, **kw)
        fig.add_scatter(x=xs, y=[q["beta"] for q in rs], showlegend=False, row=2, col=2, **kw)
    fig.add_hline(y=27.5, line=dict(color="#c53030", dash="dash", width=1), row=1, col=1)
    fig.add_hline(y=3.0, line=dict(color="#c53030", dash="dash", width=1), row=1, col=2)
    fig.add_hline(y=0.0, line=dict(color=MUTED, width=1), row=2, col=2)
    fig.update_xaxes(title=pname, row=2)
    st.plotly_chart(style(fig, 620), width="stretch")

    tab = pd.DataFrame([dict(x=x, pc_bar=q["pc"] / 1e5, of=q["of"], mdot_gs=q["mdot"] * 1e3,
                             mo_gs=q["mo"] * 1e3, mf_gs=q["mf"] * 1e3, J_mid=q["J"][1], beta_deg=q["beta"])
                        for x, q in zip(xs, res[model])]).rename(columns={"x": pname})
    st.dataframe(tab.round(3), hide_index=True, width="stretch")

# =========================================================================== test data
with tab_t:
    st.markdown(
        "Enter measured runs below or upload a CSV with the same columns. Flows are optional: a blank "
        "N2O or EtOH flow is estimated from the orifice equation at the measured pc. The throat is the "
        f"one in the sidebar (**{d_t * 1e3:.2f} mm**); set it to Fixed and enter the measured throat for real tests.")
    cols = ["run", "pc_bar", "p_ox_bar", "p_fu_bar", "T_ox_C", "mdot_ox_gs", "mdot_fu_gs"]
    up = st.file_uploader("CSV", type="csv")
    if up is not None:
        df0 = pd.read_csv(up)
    else:
        df0 = pd.DataFrame([["example-1", 24.0, 92.0, 93.0, 18.0, 10.4, None],
                            ["example-2", 26.5, 95.0, 95.0, 21.0, 10.1, 5.8]], columns=cols)
    df = st.data_editor(df0, num_rows="dynamic", width="stretch", key="tests")

    out = []
    for _, row in df.iterrows():
        try:
            e = ti.evaluate_test(ig, d_t, pc=float(row.pc_bar) * 1e5, p_ox=float(row.p_ox_bar) * 1e5,
                                 p_fu=float(row.p_fu_bar) * 1e5, T_ox=float(row.T_ox_C) + 273.15,
                                 mdot_o=float(row.mdot_ox_gs) * 1e-3 if pd.notna(row.mdot_ox_gs) else None,
                                 mdot_f=float(row.mdot_fu_gs) * 1e-3 if pd.notna(row.mdot_fu_gs) else None,
                                 eta_nominal=eta, model=model)
        except Exception as ex:  # incomplete row
            continue
        p = e["pred"]
        out.append(dict(
            run=row.run, pc_meas=e["pc"] / 1e5, pc_model=p["pc"] / 1e5,
            pc_err_pct=(e["pc"] / p["pc"] - 1) * 100,
            of_meas=e["of"], of_model=p["of"],
            mdot_meas=(e["mo"] + e["mf"]) * 1e3, mdot_model=p["mdot"] * 1e3,
            flows_estimated=("ox " if e["mo_est"] else "") + ("fu" if e["mf_est"] else ""),
            cstar_meas=e["cstar"], cstar_ideal=e["cstar_ideal"], eta_cstar=e["eta"],
            cd_ox_spi=e["cd_o_implied"]["spi"], cd_ox_dyer=e["cd_o_implied"]["dyer"],
            cd_ox_hem=e["cd_o_implied"]["hem"], cd_fu=e["cd_f_implied"],
            J=math.sqrt(e["J"][0] * e["J"][1]), beta_deg=e["beta"]))
    if not out:
        st.info("No complete rows yet.")
    else:
        res_df = pd.DataFrame(out)
        st.dataframe(res_df.round(3), hide_index=True, width="stretch")
        st.caption("eta_cstar = measured c* / ideal c* at the measured O/F. cd_ox_* = N2O Cd that "
                   "reproduces the measured N2O flow with each model (blank without a flow measurement).")
        c1, c2 = st.columns(2)
        with c1:
            f1 = go.Figure()
            lim = [0, max(res_df.pc_meas.max(), res_df.pc_model.max()) * 1.15]
            f1.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=MUTED, dash="dash"), name="1:1")
            f1.add_scatter(x=res_df.pc_model, y=res_df.pc_meas, mode="markers+text", text=res_df.run,
                           textposition="top center", marker=dict(color=INK, size=9), name="runs")
            f1.update_xaxes(title="Model pc [bar]")
            f1.update_yaxes(title="Measured pc [bar]")
            f1.update_layout(title="Chamber pressure")
            st.plotly_chart(style(f1, 380), width="stretch")
        with c2:
            f2 = go.Figure()
            f2.add_bar(x=res_df.run, y=res_df.eta_cstar, marker_color=INK, name="c* efficiency")
            f2.add_hline(y=eta, line=dict(color="#c53030", dash="dash"), annotation_text="nominal")
            f2.update_yaxes(title="c* efficiency [-]", range=[0, 1.1])
            f2.update_layout(title="c* efficiency")
            st.plotly_chart(style(f2, 380), width="stretch")
        if res_df[["cd_ox_spi", "cd_ox_dyer", "cd_ox_hem"]].notna().any().any():
            f3 = go.Figure()
            for mm in ti.MODELS:
                f3.add_bar(x=res_df.run, y=res_df[f"cd_ox_{mm}"], name=mm.upper(), marker_color=COL[mm])
            f3.add_hline(y=cd_o, line=dict(color="#c53030", dash="dash"), annotation_text="Cd used")
            f3.update_layout(title="Implied N2O Cd per flow model", barmode="group")
            st.plotly_chart(style(f3, 360), width="stretch")
        st.download_button("Download results CSV", res_df.to_csv(index=False), "igniter_test_comparison.csv")
