# Torch igniter sizing

`torch_igniter.py` – model library and CLI (N2O SPI/HEM/Dyer, ethanol SPI, throat sizing,
pc solve for a fixed throat, doublet momentum ratio and spray angle, test back-calculation).
`app.py` – Streamlit explorer (Design / Off-design / Test data).

    pip install -r requirements.txt
    python torch_igniter.py        # design point + N2O temperature sweep
    streamlit run app.py

c*(O/F) comes from rocketcea if installed, otherwise from `CEA_TABLE` in torch_igniter.py
(V1 report values, replace with h2cea). Test CSV columns:
`run, pc_bar, p_ox_bar, p_fu_bar, T_ox_C, mdot_ox_gs, mdot_fu_gs` (flows may be blank).
