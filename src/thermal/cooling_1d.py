import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

# Optional CoolProp integration with NIST 50-bar supercritical CH4 table fallback
try:
    from CoolProp.CoolProp import PropsSI
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False

# NIST reference properties for supercritical CH4 at 50 bar (120-600 K)
_T_TAB   = np.array([120.0, 140.0, 160.0, 180.0, 195.0, 210.0, 230.0, 260.0, 300.0, 350.0, 400.0, 500.0, 600.0])
_RHO_TAB = np.array([418.0, 390.0, 355.0, 305.0, 205.0, 118.0,  78.0,  56.0,  42.0,  33.0,  27.5,  21.0,  17.0])
_CP_TAB  = np.array([3450., 3680., 4150., 5150., 9200., 5200., 3500., 2850., 2600., 2580., 2680., 2950., 3280.])
_MU_TAB  = np.array([1.08e-4, 7.8e-5, 5.5e-5, 3.8e-5, 2.1e-5, 1.48e-5, 1.30e-5, 1.25e-5, 1.28e-5, 1.38e-5, 1.50e-5, 1.75e-5, 1.98e-5])
_K_TAB   = np.array([0.178, 0.152, 0.125, 0.098, 0.068, 0.049, 0.043, 0.041, 0.042, 0.047, 0.053, 0.068, 0.084])


def get_methane_properties(temp_k, press_pa):
    # Return supercritical CH4 (rho, cp, mu, k) via CoolProp or NIST 50-bar lookup
    t_clamped = float(np.clip(temp_k, 115.0, 650.0))
    p_clamped = float(np.clip(press_pa, 20e5, 100e5))
    if HAS_COOLPROP:
        try:
            rho = PropsSI('D', 'T', t_clamped, 'P', p_clamped, 'Methane')
            cp  = PropsSI('Cpmass', 'T', t_clamped, 'P', p_clamped, 'Methane')
            mu  = PropsSI('V', 'T', t_clamped, 'P', p_clamped, 'Methane')
            k_f = PropsSI('L', 'T', t_clamped, 'P', p_clamped, 'Methane')
            return rho, cp, mu, k_f
        except Exception:
            pass

    rho_50 = float(np.interp(t_clamped, _T_TAB, _RHO_TAB))
    p_corr = 1.0 + (p_clamped / 50e5 - 1.0) * float(np.clip((t_clamped - 180.0) / 60.0, 0.0, 1.0))
    rho = rho_50 * p_corr
    cp  = float(np.interp(t_clamped, _T_TAB, _CP_TAB))
    mu  = float(np.interp(t_clamped, _T_TAB, _MU_TAB))
    k_f = float(np.interp(t_clamped, _T_TAB, _K_TAB))
    return rho, cp, mu, k_f


def run_cooling_analysis(workdir, results_dir, m_dot_total=8.68):
    # Compute 1D counter-flow supercritical CH4 thermal and hydraulic cooling profiles
    prop_src = "CoolProp" if HAS_COOLPROP else "NIST 50-bar Supercritical Table"
    print(f"--- [THERMAL MODULE] Computing 1D regenerative cooling & hydraulics ({prop_src}) ---")

    thermal_file = os.path.join(workdir, "wall_thermal_loads.csv")
    if not os.path.exists(thermal_file):
        print("[THERMAL MODULE] Missing wall_thermal_loads.csv. Skipping.")
        return

    df = pd.read_csv(thermal_file)
    df.columns = df.columns.str.strip().str.replace('"', '')
    df = df.sort_values(by="x").reset_index(drop=True)

    x = df['x'].values
    r = df['y'].values
    q_gas = df['Heat_Flux_CFD'].values
    n_pts = len(x)

    # Propellant flow and channel geometric parameters
    m_dot_fuel = m_dot_total / (1.0 + config.O_F_RATIO)
    n_ch = getattr(config, 'COOLANT_CHANNELS', 80)
    h_throat = getattr(config, 'CHANNEL_HEIGHT_THROAT', 0.0014)
    h_conv = getattr(config, 'CHANNEL_HEIGHT_CONV', 0.0022)
    h_bell = getattr(config, 'CHANNEL_HEIGHT_BELL', 0.0028)
    rib_frac = getattr(config, 'RIB_PITCH_FRACTION', 0.42)
    w_max_conv = getattr(config, 'MAX_CONV_CHANNEL_WIDTH', 0.0036)
    eps_rough = getattr(config, 'CHANNEL_ROUGHNESS', 5.0e-6)
    t_hot = getattr(config, 'LINER_THICKNESS', 0.0010)
    k_cu = getattr(config, 'WALL_K', 320.0)
    p_cool_in = getattr(config, 'COOLANT_INLET_PRESSURE', 55.0e5)
    t_cool_in = getattr(config, 'COOLANT_INLET_TEMP', 120.0)

    # Local circumferential pitch and variable channel width/height
    r_t = float(np.min(r))
    r_c = float(r[0])
    r_e = float(r[-1])
    pitch = 2.0 * np.pi * (r + t_hot) / n_ch
    t_rib = np.maximum(0.0010, rib_frac * pitch)
    w_ch = np.clip(pitch - t_rib, 0.0008, w_max_conv)
    w_ch = np.where(x < 0.0, np.minimum(w_ch, w_max_conv), w_ch)
    t_rib = pitch - w_ch

    h_ch = np.zeros(n_pts)
    for i in range(n_pts):
        if x[i] <= 0.0:
            frac = np.clip((r[i] - r_t) / max(r_c - r_t, 1e-6), 0.0, 1.0)
            h_ch[i] = h_throat + (h_conv - h_throat) * frac
        else:
            frac = np.clip((r[i] - r_t) / max(r_e - r_t, 1e-6), 0.0, 1.0)
            h_ch[i] = h_throat + (h_bell - h_throat) * np.sqrt(frac)

    a_ch = w_ch * h_ch
    d_h = (2.0 * w_ch * h_ch) / (w_ch + h_ch)
    g_ch = (m_dot_fuel / n_ch) / a_ch

    t_cool = np.zeros(n_pts)
    p_cool = np.zeros(n_pts)
    h_c_eff = np.zeros(n_pts)
    v_cool = np.zeros(n_pts)

    t_cool[-1] = t_cool_in
    p_cool[-1] = p_cool_in

    # Counter-flow marching from exit (n_pts - 1) to chamber inlet (0)
    for i in range(n_pts - 1, -1, -1):
        rho, cp, mu, k_f = get_methane_properties(t_cool[i], p_cool[i])
        pr = (mu * cp) / k_f
        re = (g_ch[i] * d_h[i]) / mu
        v_cool[i] = g_ch[i] / rho

        # Haaland Darcy friction factor and Gnielinski Nusselt correlation
        f_darcy = (-1.8 * np.log10((eps_rough / (3.7 * d_h[i]))**1.11 + 6.9 / max(re, 3000.0)))**(-2)
        nu_g = ((f_darcy / 8.0) * (max(re, 3000.0) - 1000.0) * pr) / (
            1.0 + 12.7 * np.sqrt(f_darcy / 8.0) * (pr**(2.0 / 3.0) - 1.0)
        )
        h_c = (nu_g * k_f) / d_h[i]

        # Copper channel rib fin efficiency
        m_fin = np.sqrt((2.0 * h_c) / (k_cu * t_rib[i]))
        eta_f = np.tanh(m_fin * h_ch[i]) / (m_fin * h_ch[i])
        h_c_eff[i] = h_c * (w_ch[i] + 2.0 * eta_f * h_ch[i]) / pitch[i]

        if i > 0:
            ds = np.hypot(x[i] - x[i - 1], r[i] - r[i - 1])
            r_mid = 0.5 * (r[i] + r[i - 1])
            q_mid = 0.5 * (q_gas[i] + q_gas[i - 1])
            dQ = q_mid * (2.0 * np.pi * r_mid * ds)
            t_cool[i - 1] = t_cool[i] + dQ / (m_dot_fuel * cp)

            # Darcy-Weisbach friction + flow acceleration pressure drop
            rho_next, _, _, _ = get_methane_properties(t_cool[i - 1], p_cool[i])
            v_next = g_ch[i - 1] / rho_next
            dp_fric = f_darcy * (ds / d_h[i]) * 0.5 * (g_ch[i]**2 / rho)
            dp_acc = 0.5 * (rho + rho_next) * 0.5 * (v_next**2 - v_cool[i]**2)
            p_cool[i - 1] = max(p_cool[i] - dp_fric - dp_acc, 1.0e5)

    t_w_cold = t_cool + (q_gas / h_c_eff)
    t_w_hot = t_w_cold + (q_gas * t_hot) / k_cu
    dp_total_bar = (p_cool[-1] - p_cool[0]) / 1.0e5
    idx_th = int(np.argmin(r))

    df['T_cool'] = t_cool
    df['P_cool_bar'] = p_cool / 1.0e5
    df['V_cool'] = v_cool
    df['w_ch_mm'] = w_ch * 1000.0
    df['h_ch_mm'] = h_ch * 1000.0
    df['h_c_eff'] = h_c_eff
    df['T_w_cold'] = t_w_cold
    df['T_w_hot'] = t_w_hot
    df.to_csv(thermal_file, index=False)

    print(
        f"[COOLING 1D] CH4 Flow: {m_dot_fuel:.2f} kg/s ({n_ch} channels, "
        f"w_th={w_ch[idx_th]*1000:.2f} mm -> w_exit={w_ch[-1]*1000:.2f} mm)"
    )
    print(
        f"[COOLING 1D] Hydraulics -> P: {p_cool[-1]/1e5:.1f} -> {p_cool[0]/1e5:.1f} bar "
        f"(dP = {dp_total_bar:.2f} bar) | Max V: {np.max(v_cool):.1f} m/s | "
        f"h_c_eff(throat): {h_c_eff[idx_th]:.0f} W/(m^2*K)"
    )
    print(
        f"[COOLING 1D] Thermal -> T_cool: {t_cool[-1]:.1f} -> {t_cool[0]:.1f} K | "
        f"T_w_hot(throat): {t_w_hot[idx_th]:.1f} K | Max T_w_hot: {np.max(t_w_hot):.1f} K"
    )

    # Dual-panel plot: Thermal profiles (top) and Hydraulic pressure/velocity (bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)

    ax1.plot(x, t_w_hot, 'r-', linewidth=2.0, label='Inner Hot Wall T_w,hot [K]')
    ax1.plot(x, t_w_cold, 'm--', linewidth=1.6, label='Coolant-Side Wall T_w,cold [K]')
    ax1.plot(x, t_cool, 'b-.', linewidth=1.8, label='Supercritical CH4 Coolant T_cool [K]')
    ax1.axvline(x=0.0, color='gray', linestyle=':', label='Nozzle Throat')
    ax1.set_ylabel('Temperature [K]')
    ax1.set_title('1D Counter-Flow Supercritical CH4 Regenerative Cooling (Variable Channels)')
    ax1.grid(True, ls='--')
    ax1.legend(loc='upper right')

    ax2_r = ax2.twinx()
    l1, = ax2.plot(x, p_cool / 1.0e5, 'b-', linewidth=2.0, label='Coolant Static Pressure [bar]')
    l2 = ax2.axhline(y=45.99, color='k', linestyle=':', linewidth=1.2, label='CH4 Critical Pressure (46.0 bar)')
    l3, = ax2_r.plot(x, v_cool, 'g--', linewidth=1.8, label='Coolant Velocity [m/s]')
    ax2.axvline(x=0.0, color='gray', linestyle=':')
    ax2.set_xlabel('Axial Coordinate x [m]')
    ax2.set_ylabel('Coolant Pressure [bar]', color='b')
    ax2_r.set_ylabel('Coolant Velocity [m/s]', color='g')
    ax2.set_title(f'Cooling Channel Hydraulics (dP = {dp_total_bar:.2f} bar, Roughness = {eps_rough*1e6:.0f} um)')
    ax2.grid(True, ls='--')
    ax2.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()], loc='center right')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_cooling_profile.png"), dpi=300)
    plt.close()