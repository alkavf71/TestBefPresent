# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# KONFIGURASI GLOBAL - MULTI-DOMAIN EXPERT SYSTEM
# ============================================================================
# --- Mechanical Vibration Limits (ISO 10816-3/7) ---
ISO_LIMITS_VELOCITY = {
    "Zone A (Good)": 2.8,
    "Zone B (Acceptable)": 4.5,
    "Zone C (Unacceptable)": 7.1,
    "Zone D (Danger)": 11.0
}

ACCEL_BASELINE = {
    "Band1 (0.5-1.5kHz)": 0.3,
    "Band2 (1.5-5kHz)": 0.2,
    "Band3 (5-16kHz)": 0.15
}

# --- Bearing Temperature Thresholds (IEC 60034-1, API 610, SKF) ---
BEARING_TEMP_LIMITS = {
    "normal_max": 70,
    "elevated_min": 70,
    "elevated_max": 80,
    "warning_min": 80,
    "warning_max": 90,
    "critical_min": 90,
    "delta_threshold": 15,
    "ambient_reference": 30
}

# --- Hydraulic Fluid Properties (BBM Specific - Pertamina) ---
FLUID_PROPERTIES = {
    "Pertalite (RON 90)": {
        "sg": 0.73,
        "vapor_pressure_kpa_38C": 52,
        "viscosity_cst_40C": 0.6,
        "flash_point_C": -43,
        "risk_level": "High"
    },
    "Pertamax (RON 92)": {
        "sg": 0.74,
        "vapor_pressure_kpa_38C": 42,
        "viscosity_cst_40C": 0.6,
        "flash_point_C": -43,
        "risk_level": "High"
    },
    "Diesel / Solar": {
        "sg": 0.84,
        "vapor_pressure_kpa_38C": 0.5,
        "viscosity_cst_40C": 3.0,
        "flash_point_C": 52,
        "risk_level": "Moderate"
    }
}

# --- Electrical Thresholds (IEC 60034-1 & Practical Limits) ---
ELECTRICAL_LIMITS = {
    "voltage_unbalance_warning": 1.0,
    "voltage_unbalance_critical": 2.0,
    "current_unbalance_warning": 5.0,
    "current_unbalance_critical": 8.0,
    "voltage_tolerance_low": 90,
    "voltage_tolerance_high": 110,
    "current_load_warning": 90,
    "current_load_critical": 100,
    "service_factor": 1.0
}

# ============================================================================
# FUNGSI REKOMENDASI - MULTI-DOMAIN
# ============================================================================
def get_mechanical_recommendation(diagnosis: str, location: str, severity: str = "Medium") -> str:
    rec_map = {
        "UNBALANCE": (
            f"🔧 **{location} - Unbalance**\n"
            f"• Lakukan single/dual plane balancing pada rotor\n"
            f"• Periksa: material buildup pada impeller, korosi blade, keyway wear\n"
            f"• Target residual unbalance: < 4W/N (g·mm) per ISO 1940-1\n"
            f"• Severity: {severity} → {'Segera jadwalkan balancing' if severity != 'Low' else 'Monitor trend'}"
        ),
        "MISALIGNMENT": (
            f"🔧 **{location} - Misalignment**\n"
            f"• Lakukan laser alignment pump-motor coupling\n"
            f"• Toleransi target: < 0.05 mm offset, < 0.05 mm/m angular\n"
            f"• Periksa: pipe strain, soft foot, coupling wear\n"
            f"• Severity: {severity} → {'Stop & align segera' if severity == 'High' else 'Jadwalkan alignment'}"
        ),
        "LOOSENESS": (
            f"🔧 **{location} - Mechanical Looseness**\n"
            f"• Torque check semua baut: foundation, bearing housing, baseplate\n"
            f"• Periksa: crack pada struktur, worn dowel pins, grout deterioration\n"
            f"• Gunakan torque wrench sesuai spec manufacturer\n"
            f"• Severity: {severity} → {'Amankan sebelum operasi' if severity == 'High' else 'Jadwalkan tightening'}"
        ),
        "BEARING_EARLY": (
            f"🔧 **{location} - Early Bearing Fault / Lubrication**\n"
            f"• Cek lubrication: jenis grease, interval, quantity\n"
            f"• Ambil oil sample jika applicable (particle count, viscosity)\n"
            f"• Monitor trend Band 3 mingguan\n"
            f"• Severity: {severity} → {'Ganti grease & monitor ketat' if severity != 'Low' else 'Lanjutkan monitoring'}"
        ),
        "BEARING_DEVELOPED": (
            f"🔧 **{location} - Developed Bearing Fault**\n"
            f"• Jadwalkan bearing replacement dalam 1-3 bulan\n"
            f"• Siapkan spare bearing (pastikan clearance & fit sesuai spec)\n"
            f"• Monitor weekly: jika Band 1 naik drastis → percepat jadwal\n"
            f"• Severity: {severity} → {'Plan shutdown segera' if severity == 'High' else 'Siapkan work order'}"
        ),
        "BEARING_SEVERE": (
            f"🔴 **{location} - Severe Bearing Damage**\n"
            f"• RISK OF CATASTROPHIC FAILURE - Pertimbangkan immediate shutdown\n"
            f"• Jika continue operasi: monitor hourly, siapkan emergency replacement\n"
            f"• Investigasi root cause: lubrication, installation, loading?\n"
            f"• Severity: HIGH → Action required dalam 24 jam"
        ),
        "Tidak Terdiagnosa": (
            "⚠️ **Pola Tidak Konsisten**\n"
            "• Data tidak match dengan rule mekanikal standar\n"
            "• Kemungkinan: multi-fault interaction, measurement error, atau fault non-rutin\n"
            "• Rekomendasi: Analisis manual oleh Vibration Analyst Level II+ dengan full spectrum review"
        )
    }
    return rec_map.get(diagnosis, rec_map["Tidak Terdiagnosa"])

def get_hydraulic_recommendation(diagnosis: str, fluid_type: str, severity: str = "Medium") -> str:
    rec_map = {
        "CAVITATION": (
            f"💧 **{fluid_type} - Cavitation Risk**\n"
            f"• Tingkatkan suction pressure atau turunkan fluid temperature\n"
            f"• Cek: strainer clogged, valve posisi, NPSH margin\n"
            f"• Target NPSH margin: > 0.5 m untuk {fluid_type}\n"
            f"• Severity: {severity} → {'Evaluasi immediate shutdown jika NPSH margin <0.3m' if severity == 'High' else 'Monitor intensif'}"
        ),
        "IMPELLER_WEAR": (
            f"💧 **{fluid_type} - Impeller Wear / Internal Clearance**\n"
            f"• Jadwalkan inspection impeller & wear ring\n"
            f"• Ukur internal clearance vs spec OEM\n"
            f"• Pertimbangkan: fluid viscosity effect pada slip loss\n"
            f"• Severity: {severity} → {'Siapkan spare impeller' if severity != 'Low' else 'Monitor trend efisiensi'}"
        ),
        "SYSTEM_RESISTANCE_HIGH": (
            f"💧 **{fluid_type} - System Resistance Higher Than Design**\n"
            f"• Cek valve discharge position, clogged line, atau filter pressure drop\n"
            f"• Verifikasi P&ID vs as-built condition\n"
            f"• Evaluasi: apakah operating point masih dalam acceptable range?\n"
            f"• Severity: {severity} → {'Adjust valve / clean line segera' if severity == 'High' else 'Jadwalkan system review'}"
        ),
        "EFFICIENCY_DROP": (
            f"💧 **{fluid_type} - Efficiency Degradation**\n"
            f"• Investigasi: mechanical loss vs hydraulic loss vs fluid property mismatch\n"
            f"• Severity: {severity} → {'Plan overhaul dalam 1-3 bulan' if severity != 'Low' else 'Monitor monthly'}"
        ),
        "NORMAL_OPERATION": (
            f"✅ **{fluid_type} - Normal Operation**\n"
            f"• Semua parameter dalam batas acceptable (±5% dari design)\n"
            f"• Rekam data ini sebagai baseline untuk trend monitoring\n"
            f"• Severity: Low → Continue routine monitoring"
        ),
        "Tidak Terdiagnosa": (
            "⚠️ **Pola Tidak Konsisten**\n"
            "• Data hydraulic tidak match dengan rule standar\n"
            "• Rekomendasi: Verifikasi data lapangan + cross-check dengan electrical/mechanical data"
        )
    }
    return rec_map.get(diagnosis, rec_map["Tidak Terdiagnosa"])

def get_electrical_recommendation(diagnosis: str, severity: str = "Medium") -> str:
    rec_map = {
        "UNDER_VOLTAGE": (
            f"⚡ **Under Voltage Condition**\n"
            f"• Cek supply voltage di MCC: possible transformer tap / cable voltage drop\n"
            f"• Verify: motor rated voltage vs actual operating voltage\n"
            f"• Severity: {severity} → {'Coordinate dengan electrical team segera' if severity == 'High' else 'Monitor voltage trend'}"
        ),
        "OVER_VOLTAGE": (
            f"⚡ **Over Voltage Condition**\n"
            f"• Cek supply voltage di MCC: possible transformer tap issue\n"
            f"• Verify: motor rated voltage vs actual operating voltage\n"
            f"• Severity: {severity} → {'Coordinate dengan electrical team segera' if severity == 'High' else 'Monitor voltage trend'}"
        ),
        "VOLTAGE_UNBALANCE": (
            f"⚡ **Voltage Unbalance Detected**\n"
            f"• Cek 3-phase supply balance di source: possible single-phase loading\n"
            f"• Inspect: loose connection, corroded terminal, faulty breaker\n"
            f"• Severity: {severity} → {'Balance supply sebelum mechanical damage' if severity != 'Low' else 'Monitor monthly'}"
        ),
        "CURRENT_UNBALANCE": (
            f"⚡ **Current Unbalance Detected**\n"
            f"• Investigasi: winding fault, rotor bar issue, atau supply problem\n"
            f"• Cek insulation resistance & winding resistance balance\n"
            f"• Severity: {severity} → {'Schedule electrical inspection' if severity != 'Low' else 'Continue monitoring'}"
        ),
        "OVER_LOAD": (
            f"⚡ **Over Load Condition**\n"
            f"• Motor operating above FLA rating\n"
            f"• Verify: process load, mechanical binding, or electrical issue\n"
            f"• Severity: {severity} → {'Reduce load immediately' if severity == 'High' else 'Monitor trend closely'}"
        ),
        "UNDER_LOAD": (
            f"⚡ **Under Load Condition**\n"
            f"• Motor operating below 50% FLA\n"
            f"• Verify: process demand, pump sizing, or system resistance\n"
            f"• Severity: Low → Review operating point vs BEP"
        ),
        "NORMAL_ELECTRICAL": (
            f"✅ **Normal Electrical Condition**\n"
            f"• Voltage balance <2%, current balance <5%, within rated limits\n"
            f"• Severity: Low → Continue routine electrical monitoring"
        ),
        "Tidak Terdiagnosa": (
            "⚠️ **Pola Tidak Konsisten**\n"
            "• Data electrical tidak match dengan rule standar\n"
            "• Rekomendasi: Verifikasi dengan power quality analyzer + cross-check domain lain"
        )
    }
    return rec_map.get(diagnosis, rec_map["Tidak Terdiagnosa"])

# ============================================================================
# FUNGSI TEMPERATURE ANALYSIS
# ============================================================================
def get_temperature_status(temp_celsius):
    if temp_celsius is None or temp_celsius == 0:
        return "N/A", "⚪", 0
    if temp_celsius < BEARING_TEMP_LIMITS["normal_max"]:
        return "Normal", "🟢", 0
    elif temp_celsius < BEARING_TEMP_LIMITS["elevated_max"]:
        return "Elevated", "🟡", 0
    elif temp_celsius < BEARING_TEMP_LIMITS["warning_max"]:
        return "Warning", "🟠", 1
    else:
        return "Critical", "🔴", 2

def calculate_temperature_confidence_adjustment(temp_dict, diagnosis_consistent):
    adjustment = 0
    notes = []
    for location, temp in temp_dict.items():
        if temp is None or temp == 0:
            continue
        status, color, sev_level = get_temperature_status(temp)
        if status == "Critical":
            if diagnosis_consistent:
                adjustment += 20
                notes.append(f"⚠️ {location}: {temp}°C (Critical) - Strong thermal confirmation")
            else:
                adjustment -= 10
                notes.append(f"⚠️ {location}: {temp}°C (Critical) - Review required")
        elif status == "Warning":
            if diagnosis_consistent:
                adjustment += 15
                notes.append(f"⚠️ {location}: {temp}°C (Warning) - Thermal confirmation")
            else:
                adjustment -= 5
                notes.append(f"⚠️ {location}: {temp}°C (Warning) - Monitor closely")
        elif status == "Elevated":
            if diagnosis_consistent:
                adjustment += 10
                notes.append(f"📈 {location}: {temp}°C (Elevated) - Early thermal indication")
            else:
                notes.append(f"📈 {location}: {temp}°C (Elevated) - Monitor trend")
    
    if temp_dict.get("Pump_DE") and temp_dict.get("Pump_NDE"):
        delta_pump = abs(temp_dict["Pump_DE"] - temp_dict["Pump_NDE"])
        if delta_pump > BEARING_TEMP_LIMITS["delta_threshold"]:
            adjustment += 5
            notes.append(f"🔍 Pump DE-NDE ΔT: {delta_pump}°C (>15°C) - Localized fault indicated")
    
    if temp_dict.get("Motor_DE") and temp_dict.get("Motor_NDE"):
        delta_motor = abs(temp_dict["Motor_DE"] - temp_dict["Motor_NDE"])
        if delta_motor > BEARING_TEMP_LIMITS["delta_threshold"]:
            adjustment += 5
            notes.append(f"🔍 Motor DE-NDE ΔT: {delta_motor}°C (>15°C) - Localized fault indicated")
            
    if temp_dict.get("Motor_DE") and temp_dict.get("Pump_DE"):
        if temp_dict["Motor_DE"] > temp_dict["Pump_DE"] + 10:
            notes.append("⚡ Motor DE > Pump DE - Possible electrical origin")
            
    return min(20, max(-10, adjustment)), notes

# ============================================================================
# FUNGSI PERHITUNGAN - HYDRAULIC DOMAIN
# ============================================================================
def calculate_hydraulic_parameters(suction_pressure, discharge_pressure, flow_rate,
                                   motor_power, sg, fluid_temp_c=40):
    delta_p = discharge_pressure - suction_pressure
    head = delta_p * 10.2 / sg if sg > 0 else 0
    hydraulic_power = (flow_rate * head * sg * 9.81) / 3600 if flow_rate > 0 and head > 0 else 0
    efficiency = (hydraulic_power / motor_power * 100) if motor_power > 0 else 0
    return {
        "delta_p_bar": delta_p,
        "head_m": head,
        "hydraulic_power_kw": hydraulic_power,
        "efficiency_percent": efficiency
    }

def classify_hydraulic_performance(head_aktual, head_design, efficiency_aktual,
                                   efficiency_bep, flow_aktual, flow_design):
    dev_head = ((head_aktual - head_design) / head_design * 100) if head_design > 0 else 0
    dev_eff = ((efficiency_aktual - efficiency_bep) / efficiency_bep * 100) if efficiency_bep > 0 else 0
    dev_flow = ((flow_aktual - flow_design) / flow_design * 100) if flow_design > 0 else 0
    
    if dev_head < -5 and dev_eff < -5:
        return "UNDER_PERFORMANCE", {"head_dev": dev_head, "eff_dev": dev_eff}
    elif dev_head > 5 and dev_flow < -5:
        return "OVER_RESISTANCE", {"head_dev": dev_head, "flow_dev": dev_flow}
    elif dev_eff < -10 and abs(dev_head) <= 5:
        return "EFFICIENCY_DROP", {"eff_dev": dev_eff}
    elif abs(dev_head) <= 5 and abs(dev_eff) <= 5 and abs(dev_flow) <= 5:
        return "NORMAL", {"head_dev": dev_head, "eff_dev": dev_eff, "flow_dev": dev_flow}
    else:
        return "MIXED_DEVIATION", {"head_dev": dev_head, "eff_dev": dev_eff, "flow_dev": dev_flow}

# ============================================================================
# FUNGSI PERHITUNGAN - ELECTRICAL DOMAIN
# ============================================================================
def calculate_electrical_parameters(v_l1l2, v_l2l3, v_l3l1, i_l1, i_l2, i_l3,
                                    rated_voltage, fla):
    v_avg = (v_l1l2 + v_l2l3 + v_l3l1) / 3
    i_avg = (i_l1 + i_l2 + i_l3) / 3
    v_deviations = [abs(v - v_avg) for v in [v_l1l2, v_l2l3, v_l3l1]]
    voltage_unbalance = (max(v_deviations) / v_avg * 100) if v_avg > 0 else 0
    i_deviations = [abs(i - i_avg) for i in [i_l1, i_l2, i_l3]]
    current_unbalance = (max(i_deviations) / i_avg * 100) if i_avg > 0 else 0
    load_estimate = (i_avg / fla * 100) if fla > 0 else 0
    voltage_within_tolerance = (ELECTRICAL_LIMITS["voltage_tolerance_low"] <= 
                                (v_avg / rated_voltage * 100) <= 
                                ELECTRICAL_LIMITS["voltage_tolerance_high"])
    return {
        "v_avg": v_avg,
        "i_avg": i_avg,
        "voltage_unbalance_percent": voltage_unbalance,
        "current_unbalance_percent": current_unbalance,
        "load_estimate_percent": load_estimate,
        "voltage_within_tolerance": voltage_within_tolerance
    }

def diagnose_electrical_condition(electrical_calc, motor_specs):
    result = {
        "diagnosis": "NORMAL_ELECTRICAL",
        "confidence": 95,
        "severity": "Low",
        "fault_type": "normal",
        "domain": "electrical",
        "details": {}
    }
    voltage_unbalance = electrical_calc.get("voltage_unbalance_percent", 0)
    current_unbalance = electrical_calc.get("current_unbalance_percent", 0)
    load_estimate = electrical_calc.get("load_estimate_percent", 0)
    voltage_within_tolerance = electrical_calc.get("voltage_within_tolerance", True)
    v_avg = electrical_calc.get("v_avg", 0)
    rated_voltage = motor_specs.get("rated_voltage", 400)

    if not voltage_within_tolerance:
        if v_avg < rated_voltage * 0.9:
            result["diagnosis"] = "UNDER_VOLTAGE"
            result["confidence"] = 70
            result["severity"] = "High" if load_estimate > 80 else "Medium"
            result["fault_type"] = "voltage"
        elif v_avg > rated_voltage * 1.1:
            result["diagnosis"] = "OVER_VOLTAGE"
            result["confidence"] = 70
            result["severity"] = "Medium"
            result["fault_type"] = "voltage"
        result["details"] = {
            "voltage_unbalance": voltage_unbalance,
            "current_unbalance": current_unbalance,
            "load_estimate": load_estimate
        }
        return result

    if voltage_unbalance > ELECTRICAL_LIMITS["voltage_unbalance_warning"]:
        result["diagnosis"] = "VOLTAGE_UNBALANCE"
        calculated_conf = 60 + int((voltage_unbalance - ELECTRICAL_LIMITS["voltage_unbalance_warning"]) * 15)
        result["confidence"] = min(95, calculated_conf)
        result["severity"] = "High" if voltage_unbalance > ELECTRICAL_LIMITS["voltage_unbalance_critical"] else "Medium"
        result["fault_type"] = "voltage"
    elif current_unbalance > ELECTRICAL_LIMITS["current_unbalance_warning"]:
        result["diagnosis"] = "CURRENT_UNBALANCE"
        calculated_conf = 60 + int((current_unbalance - ELECTRICAL_LIMITS["current_unbalance_warning"]) * 5)
        result["confidence"] = min(95, calculated_conf)
        result["severity"] = "High" if current_unbalance > ELECTRICAL_LIMITS["current_unbalance_critical"] else "Medium"
        result["fault_type"] = "current"
    else:
        if load_estimate > ELECTRICAL_LIMITS["current_load_critical"]:
            result["diagnosis"] = "OVER_LOAD"
            result["confidence"] = min(95, 55 + int(load_estimate - 100))
            result["severity"] = "Medium"
            result["fault_type"] = "load"
        elif load_estimate < 50:
            result["diagnosis"] = "UNDER_LOAD"
            result["confidence"] = 50
            result["severity"] = "Low"
            result["fault_type"] = "load"
    
    result["details"] = {
        "voltage_unbalance": voltage_unbalance,
        "current_unbalance": current_unbalance,
        "load_estimate": load_estimate
    }
    return result

# ============================================================================
# FUNGSI DIAGNOSA - MECHANICAL DOMAIN (MULTI-POINT SUPPORT)
# ============================================================================
def diagnose_mechanical_system(vel_data, bands_data, fft_data_dict, rpm_hz, temp_data):
    result = {
        "diagnosis": "Normal",
        "confidence": 99,
        "severity": "Low",
        "fault_type": "normal",
        "domain": "mechanical",
        "champion_points": [],
        "temperature_notes": [],
        "point_diagnoses": {}
    }
    limit_warning = ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]
    limit_danger = ISO_LIMITS_VELOCITY["Zone C (Unacceptable)"]
    worst_bearing_severity = "Low"
    bearing_diag = "Normal"
    base3 = ACCEL_BASELINE["Band3 (5-16kHz)"]
    base2 = ACCEL_BASELINE["Band2 (1.5-5kHz)"]
    base1 = ACCEL_BASELINE["Band1 (0.5-1.5kHz)"]
    problematic_points = []

    for point, bands in bands_data.items():
        b3 = bands.get("Band3", 0)
        b2 = bands.get("Band2", 0)
        b1 = bands.get("Band1", 0)
        vel = vel_data.get(point, 0)
        point_diagnosis = {
            "velocity": vel,
            "bands": bands,
            "fault_type": "normal",
            "severity": "Low"
        }

        # High Frequency Analysis (Bearing)
        if b1 > 2.5 * base1 and b2 > 1.5 * base2:
            point_diagnosis["fault_type"] = "BEARING_SEVERE"
            point_diagnosis["severity"] = "High"
            worst_bearing_severity = "High"
            bearing_diag = "BEARING_SEVERE"
            problematic_points.append(point)
        elif b2 > 2.0 * base2 and b3 > 1.5 * base3:
            point_diagnosis["fault_type"] = "BEARING_DEVELOPED"
            point_diagnosis["severity"] = "High" if b2 > 3*base2 else "Medium"
            if point_diagnosis["severity"] == "High":
                worst_bearing_severity = "High"
            bearing_diag = "BEARING_DEVELOPED"
            problematic_points.append(point)
        elif b3 > 2.0 * base3:
            if worst_bearing_severity == "Low":
                worst_bearing_severity = "Medium"
            point_diagnosis["fault_type"] = "BEARING_EARLY"
            point_diagnosis["severity"] = "Medium"
            bearing_diag = "BEARING_EARLY"
            problematic_points.append(point)

        # Low Frequency Analysis (Velocity > Limit)
        if vel > limit_warning:
            low_freq_severity = "High" if vel > limit_danger else "Medium"
            parts = point.split()
            if len(parts) >= 3:
                machine = parts[0]
                end = parts[1]
                direction = parts[2]
            else:
                machine, end, direction = "Pump", "DE", "Horizontal"
            
            fft_champ_data = fft_data_dict.get(point, [(rpm_hz, 0.1), (2*rpm_hz, 0.05)])
            amp_1x = next((p[1] for p in fft_champ_data if abs(p[0]-rpm_hz) < 0.05*rpm_hz), 0)
            amp_2x = next((p[1] for p in fft_champ_data if abs(p[0]-2*rpm_hz) < 0.05*rpm_hz), 0)
            
            low_freq_diag = None
            if direction == "Axial" and end == "DE":
                opp_machine = "Pump" if machine == "Motor" else "Motor"
                opp_point = f"{opp_machine} DE Axial"
                opp_vel = vel_data.get(opp_point, 0)
                if amp_2x > 0.5 * amp_1x or opp_vel > limit_warning:
                    low_freq_diag = "MISALIGNMENT"
            elif direction == "Horizontal":
                opp_end = "NDE" if end == "DE" else "DE"
                opp_point = f"{machine} {opp_end} Horizontal"
                opp_vel = vel_data.get(opp_point, 0)
                total_fft = sum(p[1] for p in fft_champ_data) if fft_champ_data else 1
                if amp_1x > 0.7 * total_fft or opp_vel > limit_warning:
                    low_freq_diag = "UNBALANCE"
            elif direction == "Vertical":
                high_verts = sum(1 for p, v in vel_data.items() if "Vertical" in p and v > limit_warning)
                if high_verts >= 2 or (amp_2x > 0.1 and amp_1x > 0.1):
                    low_freq_diag = "LOOSENESS"
            
            if low_freq_diag:
                point_diagnosis["fault_type"] = low_freq_diag
                point_diagnosis["severity"] = low_freq_severity
                problematic_points.append(point)
        
        result["point_diagnoses"][point] = point_diagnosis

    result["champion_points"] = problematic_points if problematic_points else ["Tidak Ada (Normal)"]
    
    if any(p["severity"] == "High" for p in result["point_diagnoses"].values()):
        result["severity"] = "High"
    elif any(p["severity"] == "Medium" for p in result["point_diagnoses"].values()):
        result["severity"] = "Medium"

    high_freq_faults = ["BEARING_SEVERE", "BEARING_DEVELOPED", "BEARING_EARLY"]
    low_freq_faults = ["UNBALANCE", "MISALIGNMENT", "LOOSENESS"]

    for fault in high_freq_faults:
        if any(p["fault_type"] == fault for p in result["point_diagnoses"].values()):
            result["diagnosis"] = fault
            result["fault_type"] = "high_freq"
            result["confidence"] = 85
            break
    else:
        for fault in low_freq_faults:
            if any(p["fault_type"] == fault for p in result["point_diagnoses"].values()):
                result["diagnosis"] = fault
                result["fault_type"] = "low_freq"
                result["confidence"] = 75
                break
    
    return result

# ============================================================================
# FUNGSI DIAGNOSA - HYDRAULIC DOMAIN (REVISI - TANPA OBSERVASI)
# ============================================================================
def diagnose_hydraulic_single_point(hydraulic_calc, design_params, fluid_props, context):
    result = {
        "diagnosis": "NORMAL_OPERATION",
        "confidence": 95,
        "severity": "Low",
        "fault_type": "normal",
        "domain": "hydraulic",
        "details": {}
    }
    head_aktual = hydraulic_calc.get("head_m", 0)
    eff_aktual = hydraulic_calc.get("efficiency_percent", 0)
    head_design = design_params.get("rated_head_m", 0)
    eff_bep = design_params.get("bep_efficiency", 0)
    flow_design = design_params.get("rated_flow_m3h", 0)
    flow_aktual = context.get("flow_aktual", 0)

    pattern, deviations = classify_hydraulic_performance(
        head_aktual, head_design, eff_aktual, eff_bep, flow_aktual, flow_design
    )
    result["details"]["deviations"] = deviations

    suction_pressure_bar = context.get("suction_pressure_bar", 0)
    vapor_pressure_kpa = fluid_props.get("vapor_pressure_kpa_38C", 0)
    sg = fluid_props.get("sg", 0.84)
    p_suction_abs_kpa = (suction_pressure_bar + 1.013) * 100
    npsha_estimated = (p_suction_abs_kpa - vapor_pressure_kpa) / (sg * 9.81) if sg > 0 else 0
    npshr = design_params.get("npsh_required_m", 0)
    npsh_margin = npsha_estimated - npshr
    result["details"]["npsh_margin_m"] = npsh_margin

    if npsh_margin < 0.5:
        result["diagnosis"] = "CAVITATION"
        result["confidence"] = min(90, 70 + int((0.5 - npsh_margin) * 20) if npsh_margin < 0.5 else 70)
        result["severity"] = "High" if npsh_margin < 0.3 else "Medium"
        result["fault_type"] = "cavitation"
        return result

    if pattern == "UNDER_PERFORMANCE":
        result["diagnosis"] = "IMPELLER_WEAR"
        result["confidence"] = min(85, 60 + int(abs(deviations.get("head_dev", 0)) * 2))
        result["severity"] = "High" if deviations.get("head_dev", 0) < -15 else "Medium"
        result["fault_type"] = "wear"
        return result
    
    if pattern == "OVER_RESISTANCE":
        result["diagnosis"] = "SYSTEM_RESISTANCE_HIGH"
        result["confidence"] = min(90, 70 + int(abs(deviations.get("head_dev", 0))))
        result["severity"] = "High" if deviations.get("flow_dev", 0) < -30 else "Medium"
        result["fault_type"] = "system"
        return result
    
    if pattern == "EFFICIENCY_DROP":
        result["diagnosis"] = "EFFICIENCY_DROP"
        result["confidence"] = min(80, 65 + int(abs(deviations.get("eff_dev", 0))))
        result["severity"] = "High" if deviations.get("eff_dev", 0) < -20 else "Medium"
        result["fault_type"] = "efficiency"
        return result
    
    if pattern == "NORMAL":
        result["diagnosis"] = "NORMAL_OPERATION"
        result["confidence"] = 95
        result["severity"] = "Low"
        result["fault_type"] = "normal"
        return result

    result["diagnosis"] = "Tidak Terdiagnosa"
    result["confidence"] = 40
    result["severity"] = "Medium"
    result["fault_type"] = "unknown"
    return result

# ============================================================================
# 🔥 FAULT PROPAGATION MAP GENERATOR (LOGIC ONLY)
# ============================================================================
def generate_fault_propagation_map(mech_result, hyd_result, elec_result, temp_data=None):
    """
    Generate fault propagation map data structure. 
    UI Rendering dilakukan di main() untuk kontrol layout yang lebih baik.
    """
    propagation_data = []
    
    # === PATTERN 1: Electrical Origin ===
    if elec_result.get("fault_type") == "voltage":
        if mech_result.get("diagnosis") in ["MISALIGNMENT", "LOOSENESS"]:
            propagation_data.append({
                "root_cause": "⚡ Electrical Supply Issue",
                "fault_chain": ["Voltage Unbalance", "Torque Pulsation", "Mechanical Stress", "Misalignment/Looseness"],
                "repair_actions": [
                    "Balance 3-phase supply di MCC",
                    "Check connection & terminal",
                    "Verify transformer tap setting",
                    "Laser alignment setelah electrical fix"
                ],
                "priority": "HIGH",
                "timeline": "1-3 hari"
            })
    
    # === PATTERN 2: Hydraulic Origin (Cavitation) ===
    if hyd_result.get("fault_type") == "cavitation":
        if mech_result.get("fault_type") == "wear" or mech_result.get("diagnosis") in ["BEARING_EARLY", "BEARING_DEVELOPED"]:
            propagation_data.append({
                "root_cause": "💧 Cavitation Damage",
                "fault_chain": ["Low NPSH Margin", "Bubble Collapse", "Impeller Erosion", "Unbalance", "Bearing Wear"],
                "repair_actions": [
                    "Increase suction pressure",
                    "Clean strainer/filter",
                    "Check valve position",
                    "Replace damaged impeller",
                    "Replace bearing if worn"
                ],
                "priority": "CRITICAL",
                "timeline": "Immediate - 1 minggu"
            })
    
    # === PATTERN 3: Mechanical Origin ===
    if mech_result.get("fault_type") in ["low_freq", "high_freq"]:
        if hyd_result.get("fault_type") == "efficiency":
            propagation_data.append({
                "root_cause": "🔧 Mechanical Fault",
                "fault_chain": ["Unbalance/Misalignment/Bearing", "Increased Friction", "Efficiency Drop", "Motor Overload"],
                "repair_actions": [
                    "Rotor balancing / Laser alignment",
                    "Bearing replacement",
                    "Check internal clearance",
                    "Verify lubrication"
                ],
                "priority": "HIGH",
                "timeline": "1-2 minggu"
            })
    
    # === PATTERN 4: Bearing Temperature ===
    if temp_data:
        high_temps = [k for k, v in temp_data.items() if v and v > 80]
        if high_temps:
            propagation_data.append({
                "root_cause": "🌡️ Bearing Overheating",
                "fault_chain": ["Poor Lubrication", "Increased Friction", "Temperature Rise", "Bearing Damage"],
                "repair_actions": [
                    "Check lubrication type & quantity",
                    "Take oil sample analysis",
                    "Verify bearing clearance",
                    "Plan bearing replacement"
                ],
                "priority": "HIGH" if any(temp_data.get(k, 0) > 90 for k in high_temps) else "MEDIUM",
                "timeline": "1-7 hari"
            })
    
    # === PATTERN 5: Delta Temperature ===
    if temp_data:
        if temp_data.get("Pump_DE") and temp_data.get("Pump_NDE"):
            if abs(temp_data["Pump_DE"] - temp_data["Pump_NDE"]) > 15:
                propagation_data.append({
                    "root_cause": "🔍 Localized Bearing Fault",
                    "fault_chain": ["Uneven Load", "Localized Heating", "Bearing Damage"],
                    "repair_actions": [
                        "Check bearing housing alignment",
                        "Verify mounting procedure",
                        "Inspect bearing raceway",
                        "Replace bearing if damaged"
                    ],
                    "priority": "MEDIUM",
                    "timeline": "1-4 minggu"
                })
    
    # === DEFAULT: No Clear Propagation ===
    if not propagation_data:
        mech_diag = mech_result.get("diagnosis", "Normal")
        hyd_diag = hyd_result.get("diagnosis", "Normal")
        elec_diag = elec_result.get("diagnosis", "Normal")
        if mech_diag != "Normal" or hyd_diag != "NORMAL_OPERATION" or elec_diag != "NORMAL_ELECTRICAL":
            propagation_data.append({
                "root_cause": "❓ Individual Domain Fault",
                "fault_chain": ["Single domain fault detected", "No strong cross-domain correlation"],
                "repair_actions": [
                    "Address individual domain fault per recommendation",
                    "Continue monitoring",
                    "Collect more data for trend analysis"
                ],
                "priority": "MEDIUM",
                "timeline": "Routine maintenance"
            })
    
    return propagation_data

# ============================================================================
# CROSS-DOMAIN INTEGRATION LOGIC
# ============================================================================
def aggregate_cross_domain_diagnosis(mech_result, hyd_result, elec_result,
                                     shared_context, temp_data=None):
    system_result = {
        "diagnosis": "Tidak Ada Korelasi Antar Domain Terdeteksi",
        "confidence": 0,
        "severity": "Low",
        "location": "N/A",
        "domain_breakdown": {},
        "correlation_notes": [],
        "temperature_notes": [],
        "affected_points": []
    }
    system_result["domain_breakdown"] = {
        "mechanical": mech_result,
        "hydraulic": hyd_result,
        "electrical": elec_result
    }
    
    mech_fault = mech_result.get("fault_type")
    hyd_fault = hyd_result.get("fault_type")
    elec_fault = elec_result.get("fault_type")
    mech_sev = mech_result.get("severity", "Low")
    hyd_sev = hyd_result.get("severity", "Low")
    elec_sev = elec_result.get("severity", "Low")
    
    system_result["affected_points"] = mech_result.get("champion_points", [])
    
    correlation_bonus = 0
    correlated_faults = []
    
    if (elec_fault == "voltage" and 
        mech_result.get("diagnosis") in ["MISALIGNMENT", "LOOSENESS"] and 
        hyd_result.get("details", {}).get("deviations", {}).get("head_dev", 0) < -5):
        correlation_bonus += 15
        correlated_faults.append("Voltage unbalance → torque pulsation → hydraulic instability")
        system_result["diagnosis"] = "Electrical-Mechanical-Hydraulic Coupled Fault"
    
    if (hyd_fault == "cavitation" and mech_fault == "wear" and 
        elec_result.get("details", {}).get("current_unbalance", 0) > 5):
        correlation_bonus += 20
        correlated_faults.append("Cavitation → impeller erosion → unbalance → current fluctuation")
        system_result["diagnosis"] = "Cascading Failure: Cavitation Origin"
    
    if (elec_result.get("diagnosis") == "OVER_LOAD" and hyd_fault == "efficiency"):
        correlation_bonus += 10
        correlated_faults.append("High electrical input + low hydraulic output → internal mechanical/hydraulic loss")
        system_result["diagnosis"] = "Internal Loss Investigation Required"
    
    if temp_data:
        temp_adjustment, temp_notes = calculate_temperature_confidence_adjustment(
            temp_data, 
            diagnosis_consistent=(mech_fault is not None and mech_fault != "normal")
        )
        correlation_bonus += temp_adjustment
        system_result["temperature_notes"] = temp_notes
        
        if temp_data.get("Pump_DE") and temp_data.get("Pump_NDE"):
            if abs(temp_data["Pump_DE"] - temp_data["Pump_NDE"]) > BEARING_TEMP_LIMITS["delta_threshold"]:
                correlated_faults.append(f"Pump DE-NDE ΔT >15°C → Localized fault on DE bearing")
        
        if temp_data.get("Motor_DE") and temp_data.get("Pump_DE"):
            if temp_data["Motor_DE"] > temp_data["Pump_DE"] + 10:
                correlated_faults.append("Motor DE > Pump DE → Possible electrical origin")
    
    severities = [mech_sev, hyd_sev, elec_sev]
    if "High" in severities:
        system_result["severity"] = "High"
    elif "Medium" in severities:
        system_result["severity"] = "Medium"
    else:
        system_result["severity"] = "Low"
        
    if temp_data:
        for temp in temp_data.values():
            if temp and temp > BEARING_TEMP_LIMITS["critical_min"]:
                system_result["severity"] = "High"
                correlated_faults.append("⚠️ Critical bearing temperature detected")
                break
    
    # SIMPLE AVERAGE CONFIDENCE (KEMBALI KE ORIGINAL - TANPA FUZZY)
    confidences = [r.get("confidence", 0) for r in [mech_result, hyd_result, elec_result] 
                   if r.get("confidence", 0) > 0]
    base_confidence = np.mean(confidences) if confidences else 0
    system_result["confidence"] = min(95, int(base_confidence + correlation_bonus))
    system_result["correlation_notes"] = correlated_faults if correlated_faults else ["Tidak ada korelasi kuat antar domain terdeteksi"]
    
    return system_result

# ============================================================================
# REPORT GENERATION - CSV
# ============================================================================
def generate_unified_csv_report(machine_id, rpm, timestamp, mech_data, hyd_data,
                                elec_data, integrated_result, temp_data=None):
    lines = []
    lines.append(f"MULTI-DOMAIN PUMP DIAGNOSTIC REPORT - {machine_id.upper()}")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"RPM: {rpm} | 1x RPM: {rpm/60:.2f} Hz")
    lines.append(f"Standards: ISO 10816-3/7 (Mech) | API 610 (Hyd) | IEC 60034 (Elec)")
    lines.append("")
    
    if temp_data:
        lines.append("=== BEARING TEMPERATURE ===")
        lines.append(f"Pump_DE: {temp_data.get('Pump_DE', 'N/A')}°C | Pump_NDE: {temp_data.get('Pump_NDE', 'N/A')}°C")
        lines.append(f"Motor_DE: {temp_data.get('Motor_DE', 'N/A')}°C | Motor_NDE: {temp_data.get('Motor_NDE', 'N/A')}°C")
        if temp_data.get('Pump_DE') and temp_data.get('Pump_NDE'):
            lines.append(f"Pump ΔT (DE-NDE): {abs(temp_data['Pump_DE'] - temp_data['Pump_NDE']):.1f}°C")
        if temp_data.get('Motor_DE') and temp_data.get('Motor_NDE'):
            lines.append(f"Motor ΔT (DE-NDE): {abs(temp_data['Motor_DE'] - temp_data['Motor_NDE']):.1f}°C")
        lines.append("")
    
    lines.append("=== MECHANICAL VIBRATION ===")
    if mech_data.get("points"):
        lines.append("POINT,Overall_Vel(mm/s),Band1(g),Band2(g),Band3(g),Status,Diagnosis")
        for point, data in mech_data["points"].items():
            vel = data.get('velocity', 0)
            bands = data.get('bands', {})
            b1 = bands.get('Band1', 0)
            b2 = bands.get('Band2', 0)
            b3 = bands.get('Band3', 0)
            point_diag = mech_data.get("point_diagnoses", {}).get(point, {})
            point_fault = point_diag.get("fault_type", "normal")
            if vel > 7.1:
                status = "Zone_D"
            elif vel > 4.5:
                status = "Zone_C"
            elif vel > 2.8:
                status = "Zone_B"
            else:
                status = "Zone_A"
            lines.append(f"{point},{vel:.2f},{b1:.3f},{b2:.3f},{b3:.3f},{status},{point_fault}")
        lines.append(f"System Diagnosis: {mech_data.get('system_diagnosis', 'N/A')}")
        champion_points = mech_data.get('champion_points', [])
        if isinstance(champion_points, list):
            lines.append(f"Champion Points: {', '.join(champion_points)}")
        else:
            lines.append(f"Champion Point: {champion_points}")
        lines.append("")
    
    lines.append("=== HYDRAULIC PERFORMANCE ===")
    if hyd_data.get("measurements"):
        m = hyd_data["measurements"]
        lines.append(f"Fluid: {hyd_data.get('fluid_type', 'N/A')} | SG: {hyd_data.get('sg', 'N/A')}")
        lines.append(f"Suction: {m.get('suction_pressure', 0):.2f} bar | Discharge: {m.get('discharge_pressure', 0):.2f} bar")
        lines.append(f"Flow: {m.get('flow_rate', 0):.1f} m³/h | Power: {m.get('motor_power', 0):.1f} kW")
        lines.append(f"Calculated Head: {hyd_data.get('head_m', 0):.1f} m | Efficiency: {hyd_data.get('efficiency_percent', 0):.1f}%")
        lines.append(f"NPSH Margin: {hyd_data.get('npsh_margin_m', 0):.2f} m")
        lines.append(f"Diagnosis: {hyd_data.get('diagnosis', 'N/A')} | Confidence: {hyd_data.get('confidence', 0)}% | Severity: {hyd_data.get('severity', 'N/A')}")
        lines.append("")
    
    lines.append("=== ELECTRICAL CONDITION ===")
    if elec_data.get("measurements"):
        e = elec_data["measurements"]
        lines.append(f"Voltage L1-L2: {e.get('v_l1l2', 0):.1f}V | L2-L3: {e.get('v_l2l3', 0):.1f}V | L3-L1: {e.get('v_l3l1', 0):.1f}V")
        lines.append(f"Current L1: {e.get('i_l1', 0):.1f}A | L2: {e.get('i_l2', 0):.1f}A | L3: {e.get('i_l3', 0):.1f}A")
        lines.append(f"Voltage Unbalance: {elec_data.get('voltage_unbalance', 0):.2f}% | Current Unbalance: {elec_data.get('current_unbalance', 0):.2f}%")
        lines.append(f"Load Estimate: {elec_data.get('load_estimate', 0):.1f}%")
        lines.append(f"Diagnosis: {elec_data.get('diagnosis', 'N/A')} | Confidence: {elec_data.get('confidence', 0)}% | Severity: {elec_data.get('severity', 'N/A')}")
        lines.append("")
    
    lines.append("=== INTEGRATED DIAGNOSIS ===")
    lines.append(f"Overall Diagnosis: {integrated_result.get('diagnosis', 'N/A')}")
    lines.append(f"Overall Confidence: {integrated_result.get('confidence', 0)}%")
    lines.append(f"Overall Severity: {integrated_result.get('severity', 'N/A')}")
    lines.append(f"Affected Points: {', '.join(integrated_result.get('affected_points', []))}")
    lines.append(f"Correlation Notes: {'; '.join(integrated_result.get('correlation_notes', []))}")
    if integrated_result.get("temperature_notes"):
        lines.append(f"Temperature Notes: {'; '.join(integrated_result['temperature_notes'])}")
    lines.append("")
    
    # FAULT PROPAGATION MAP (NEW - ADDED TO CSV)
    lines.append("=== FAULT PROPAGATION MAP FOR REPAIR ===")
    mech_result = integrated_result.get("domain_breakdown", {}).get("mechanical", {})
    hyd_result = integrated_result.get("domain_breakdown", {}).get("hydraulic", {})
    elec_result = integrated_result.get("domain_breakdown", {}).get("electrical", {})
    propagation_map = generate_fault_propagation_map(mech_result, hyd_result, elec_result, temp_data)
    for idx, prop in enumerate(propagation_map, 1):
        lines.append(f"Scenario {idx}: {prop['root_cause']}")
        lines.append(f"Priority: {prop['priority']} | Timeline: {prop['timeline']}")
        lines.append(f"Fault Chain: {' -> '.join(prop['fault_chain'])}")
        lines.append("Repair Actions:")
        for action in prop["repair_actions"]:
            lines.append(f"  - {action}")
        lines.append("")
    
    return "\n".join(lines)

# ============================================================================
# STREAMLIT UI - MAIN APPLICATION
# ============================================================================
def main():
    st.set_page_config(
        page_title="Pump Diagnostic Expert System",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if "shared_context" not in st.session_state:
        st.session_state.shared_context = {
            "machine_id": "P-101",
            "rpm": 2950,
            "service_criticality": "Essential (Utility)",
            "fluid_type": "Diesel / Solar",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    st.markdown("""
    <div style="background-color:#1E3A5F; padding:15px; border-radius:8px; margin-bottom:20px; text-align:center;">
        <h2 style="color:white; margin:0">🔧💧⚡ Pump Diagnostic Expert System</h2>
        <p style="color:#E0E0E0; margin:5px 0 0 0">
            Integrated Mechanical • Hydraulic • Electrical Analysis | Pertamina Patra Niaga
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("📍 Shared Context")
        machine_id = st.text_input("Machine ID / Tag", value=st.session_state.shared_context["machine_id"])
        rpm = st.number_input("Operating RPM", min_value=600, max_value=3600, 
                              value=st.session_state.shared_context["rpm"], step=10)
        service_type = st.selectbox("Service Criticality", 
                                    ["Critical (Process)", "Essential (Utility)", "Standby"],
                                    index=["Critical (Process)", "Essential (Utility)", "Standby"].index(
                                        st.session_state.shared_context["service_criticality"]))
        fluid_type = st.selectbox("Fluid Type (BBM)", 
                                  list(FLUID_PROPERTIES.keys()),
                                  index=list(FLUID_PROPERTIES.keys()).index(
                                      st.session_state.shared_context["fluid_type"]))
        
        st.session_state.shared_context.update({
            "machine_id": machine_id,
            "rpm": rpm,
            "service_criticality": service_type,
            "fluid_type": fluid_type
        })
        
        fluid_props = FLUID_PROPERTIES[fluid_type]
        st.info(f"""
        **Fluid Properties ({fluid_type}):**
        - SG: {fluid_props['sg']}
        - Vapor Pressure @38°C: {fluid_props['vapor_pressure_kpa_38C']} kPa
        - Risk Level: {fluid_props['risk_level']}
        """)
        
        st.divider()
        st.subheader("🧭 Navigasi")
        st.markdown("""
        - 🔧 **Mechanical**: Vibration analysis
        - 💧 **Hydraulic**: Performance troubleshooting
        - ⚡ **Electrical**: 3-phase condition monitoring
        - 🔗 **Integrated**: Cross-domain correlation
        """)
        
        st.divider()
        st.caption("📊 Status Analisis:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            mech_done = "✅" if "mech_result" in st.session_state else "⏳"
            st.write(f"{mech_done} Mechanical")
        with col_s2:
            hyd_done = "✅" if "hyd_result" in st.session_state else "⏳"
            st.write(f"{hyd_done} Hydraulic")
        col_s3, col_s4 = st.columns(2)
        with col_s3:
            elec_done = "✅" if "elec_result" in st.session_state else "⏳"
            st.write(f"{elec_done} Electrical")
        with col_s4:
            int_done = "✅" if "integrated_result" in st.session_state else "⏳"
            st.write(f"{int_done} Integrated")

    tab_mech, tab_hyd, tab_elec, tab_integrated = st.tabs([
        "🔧 Mechanical", "💧 Hydraulic", "⚡ Electrical", "🔗 Integrated Summary"
    ])

    # TAB 1: MECHANICAL
    with tab_mech:
        st.header("🔧 Mechanical Vibration Analysis")
        st.caption("ISO 10816-3/7 | Centrifugal Pump + Electric Motor")
        
        st.subheader("🌡️ Bearing Temperature (4 Points)")
        temp_cols = st.columns(4)
        temp_data = {}
        with temp_cols[0]:
            pump_de_temp = st.number_input("Pump DE (°C)", min_value=0, max_value=150, 
                                           value=65, step=1, key="temp_pump_de")
            temp_data["Pump_DE"] = pump_de_temp
            if pump_de_temp > BEARING_TEMP_LIMITS["warning_min"]:
                st.error(f"🔴 {pump_de_temp}°C - Warning")
            elif pump_de_temp > BEARING_TEMP_LIMITS["elevated_min"]:
                st.warning(f"🟡 {pump_de_temp}°C - Elevated")
            else:
                st.success(f"🟢 {pump_de_temp}°C - Normal")
        with temp_cols[1]:
            pump_nde_temp = st.number_input("Pump NDE (°C)", min_value=0, max_value=150, 
                                            value=63, step=1, key="temp_pump_nde")
            temp_data["Pump_NDE"] = pump_nde_temp
            if pump_nde_temp > BEARING_TEMP_LIMITS["warning_min"]:
                st.error(f"🔴 {pump_nde_temp}°C - Warning")
            elif pump_nde_temp > BEARING_TEMP_LIMITS["elevated_min"]:
                st.warning(f"🟡 {pump_nde_temp}°C - Elevated")
            else:
                st.success(f"🟢 {pump_nde_temp}°C - Normal")
        with temp_cols[2]:
            motor_de_temp = st.number_input("Motor DE (°C)", min_value=0, max_value=150, 
                                            value=68, step=1, key="temp_motor_de")
            temp_data["Motor_DE"] = motor_de_temp
            if motor_de_temp > BEARING_TEMP_LIMITS["warning_min"]:
                st.error(f"🔴 {motor_de_temp}°C - Warning")
            elif motor_de_temp > BEARING_TEMP_LIMITS["elevated_min"]:
                st.warning(f"🟡 {motor_de_temp}°C - Elevated")
            else:
                st.success(f"🟢 {motor_de_temp}°C - Normal")
        with temp_cols[3]:
            motor_nde_temp = st.number_input("Motor NDE (°C)", min_value=0, max_value=150, 
                                             value=66, step=1, key="temp_motor_nde")
            temp_data["Motor_NDE"] = motor_nde_temp
            if motor_nde_temp > BEARING_TEMP_LIMITS["warning_min"]:
                st.error(f"🔴 {motor_nde_temp}°C - Warning")
            elif motor_nde_temp > BEARING_TEMP_LIMITS["elevated_min"]:
                st.warning(f"🟡 {motor_nde_temp}°C - Elevated")
            else:
                st.success(f"🟢 {motor_nde_temp}°C - Normal")
        
        st.divider()
        st.subheader("📊 Input Data 12 Titik Pengukuran")
        points = [f"{machine} {end} {direction}" 
                  for machine in ["Pump", "Motor"] 
                  for end in ["DE", "NDE"] 
                  for direction in ["Horizontal", "Vertical", "Axial"]]
        
        input_data = {}
        bands_inputs = {}
        cols = st.columns(3)
        for idx, point in enumerate(points):
            with cols[idx % 3]:
                with st.expander(f"📍 {point}", expanded=False):
                    overall = st.number_input("Overall Vel (mm/s)", min_value=0.0, max_value=30.0, 
                                              value=1.0, step=0.1, key=f"mech_vel_{point}")
                    input_data[point] = overall
                    st.caption("Freq Bands (g) - Bearing")
                    b1 = st.number_input("Band 1", min_value=0.0, value=0.2, step=0.05, key=f"m_b1_{point}")
                    b2 = st.number_input("Band 2", min_value=0.0, value=0.15, step=0.05, key=f"m_b2_{point}")
                    b3 = st.number_input("Band 3", min_value=0.0, value=0.1, step=0.05, key=f"m_b3_{point}")
                    bands_inputs[point] = {"Band1": b1, "Band2": b2, "Band3": b3}
                    if overall > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]:
                        st.error(f"⚠️ {overall} mm/s (High)")
        
        problematic_points = [p for p, v in input_data.items() 
                              if v > ISO_LIMITS_VELOCITY["Zone B (Acceptable)"]]
        if problematic_points:
            st.markdown(f"""
            <div style="background-color:#ffeeba; padding:15px; border-radius:8px; border-left:5px solid #ffc107; margin-top:20px;">
                <h4 style="margin:0; color:#856404;">🎯 Multi-Point Detection: {len(problematic_points)} Titik Bermasalah</h4>
                <p style="margin:5px 0 0 0; color:#856404;">
                    Titik dengan vibrasi tinggi: <b>{', '.join(problematic_points)}</b><br>
                    Silakan masukkan data Spektrum FFT untuk <b>semua titik yang ditandai</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            fft_data_dict = {}
            for point in problematic_points:
                with st.expander(f"📈 Input FFT Spectrum untuk: {point}", expanded=True):
                    rpm_hz = rpm / 60
                    point_fft_peaks = []
                    for i in range(1, 4):
                        c1, c2 = st.columns(2)
                        with c1:
                            default_freq = rpm_hz * i
                            freq = st.number_input(f"Peak {i} Freq (Hz)", min_value=0.1, 
                                                   value=default_freq, key=f"fft_f_{point}_{i}")
                        with c2:
                            amp = st.number_input(f"Peak {i} Amp (mm/s)", min_value=0.01, 
                                                  value=1.0, step=0.1, key=f"fft_a_{point}_{i}")
                        point_fft_peaks.append((freq, amp))
                    fft_data_dict[point] = point_fft_peaks
        else:
            rpm_hz = rpm / 60
            fft_data_dict = {p: [(rpm_hz, 0.1), (2*rpm_hz, 0.05)] for p in points}
            st.success("✅ Semua titik vibrasi dalam batas normal.")
        
        if st.button("🔍 Jalankan Mechanical Analysis", type="primary", key="run_mech"):
            with st.spinner("Menganalisis pola getaran..."):
                mech_system = diagnose_mechanical_system(
                    input_data, bands_inputs, fft_data_dict, rpm/60, temp_data
                )
                st.session_state.mech_result = mech_system
                st.session_state.mech_data = {
                    "points": {p: {"velocity": input_data[p], "bands": bands_inputs[p]} for p in points},
                    "point_diagnoses": mech_system["point_diagnoses"],
                    "system_diagnosis": mech_system["diagnosis"],
                    "champion_points": mech_system["champion_points"]
                }
                st.session_state.temp_data = temp_data
                st.success(f"✅ Analisis Selesai: {mech_system['diagnosis']}")
        
        if "mech_result" in st.session_state:
            result = st.session_state.mech_result
            champion_points = result.get("champion_points", [])
            if isinstance(champion_points, list):
                points_display = ", ".join(champion_points)
            else:
                points_display = champion_points
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Diagnosis Utama", result["diagnosis"])
            with col_b:
                st.metric("Titik Sumber", points_display)
            with col_c:
                st.metric("Severity", {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(result["severity"],"⚪"))
            
            if result["diagnosis"] != "Normal":
                st.info(get_mechanical_recommendation(result["diagnosis"], points_display, result["severity"]))
            
            st.subheader("📋 Diagnosis Per Titik")
            point_df_data = []
            for point, diag in result.get("point_diagnoses", {}).items():
                point_df_data.append({
                    "Titik": point,
                    "Velocity (mm/s)": diag.get("velocity", 0),
                    "Fault Type": diag.get("fault_type", "normal"),
                    "Severity": diag.get("severity", "Low")
                })
            point_df = pd.DataFrame(point_df_data)
            st.dataframe(point_df, use_container_width=True)

    # TAB 2: HYDRAULIC
    with tab_hyd:
        st.header("💧 Hydraulic Troubleshooting")
        st.caption("Single-Point Steady-State Measurement")
        
        def estimate_bep_efficiency(Q, H, P_motor, SG, motor_eff=0.90):
            P_hyd_design = (Q * H * SG * 9.81) / 3600
            P_shaft_est = P_motor * motor_eff
            if P_shaft_est > 0 and P_hyd_design > 0:
                eff = (P_hyd_design / P_shaft_est) * 100
                return min(90, max(50, eff))
            return 75
        
        def estimate_npshr_conservative(Q_m3h):
            if Q_m3h < 50:
                return 3.0
            elif Q_m3h < 200:
                return 4.0
            else:
                return 5.5
        
        st.subheader("📊 Data Primer Hidrolik")
        col1, col2, col3 = st.columns(3)
        with col1:
            suction_pressure = st.number_input("Suction Pressure [bar]", min_value=-1.0, 
                                               value=0.44, step=0.01, key="suction_p")
            discharge_pressure = st.number_input("Discharge Pressure [bar]", min_value=0.0, 
                                                 value=3.73, step=0.01, key="discharge_p")
            delta_p = discharge_pressure - suction_pressure
            st.metric("ΔP", f"{delta_p:.2f} bar")
        with col2:
            flow_rate = st.number_input("Flow Rate [m³/h]", min_value=0.0, value=100.0, 
                                        step=1.0, key="flow_rate")
            motor_power = st.number_input("Motor Power [kW]", min_value=0.0, 
                                          value=15.0, step=0.5, key="motor_power")
        with col3:
            fluid_props = FLUID_PROPERTIES[fluid_type]
            sg = st.number_input("Specific Gravity", min_value=0.5, max_value=1.5, 
                                 value=fluid_props["sg"], step=0.01, key="sg_input")
        
        with st.expander("📋 Data Nameplate", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                rated_flow = st.number_input("Rated Flow Q [m³/h]", min_value=0.0, 
                                             value=100.0, step=1.0, key="rated_flow")
                rated_head = st.number_input("Rated Head H [m]", min_value=0.0, 
                                             value=59.73, step=0.1, key="rated_head")
            with col2:
                bep_efficiency = st.number_input("BEP Efficiency [%] (Optional)", 
                                                 min_value=0, max_value=100, value=0, step=1, 
                                                 key="bep_eff")
                npsh_required = st.number_input("NPSH Required [m] (Optional)", 
                                                min_value=0.0, value=0.0, step=0.1, 
                                                key="npshr")
        
        estimation_notes = []
        if bep_efficiency <= 0:
            bep_efficiency = estimate_bep_efficiency(rated_flow, rated_head, motor_power, sg)
            estimation_notes.append(f"BEP diestimasi: {bep_efficiency:.1f}%")
        if npsh_required <= 0:
            npsh_required = estimate_npshr_conservative(rated_flow)
            estimation_notes.append(f"NPSHr diestimasi: {npsh_required:.1f}m")
        if estimation_notes:
            st.info("🔧 **Auto-Estimation:** " + " | ".join(estimation_notes))
        
        analyze_hyd_disabled = suction_pressure >= discharge_pressure
        if st.button("💧 Generate Diagnosis", type="primary", key="run_hyd", 
                     disabled=analyze_hyd_disabled):
            with st.spinner("Menganalisis performa hidrolik..."):
                hyd_calc = calculate_hydraulic_parameters(
                    suction_pressure, discharge_pressure, flow_rate, 
                    motor_power, sg
                )
                design_params = {
                    "rated_flow_m3h": rated_flow,
                    "rated_head_m": rated_head,
                    "bep_efficiency": bep_efficiency,
                    "npsh_required_m": npsh_required
                }
                context = {
                    "flow_aktual": flow_rate,
                    "suction_pressure_bar": suction_pressure
                }
                hyd_result = diagnose_hydraulic_single_point(
                    hyd_calc, design_params, fluid_props, context
                )
                st.session_state.hyd_result = hyd_result
                st.session_state.hyd_data = {
                    "measurements": {
                        "suction_pressure": suction_pressure,
                        "discharge_pressure": discharge_pressure,
                        "flow_rate": flow_rate,
                        "motor_power": motor_power
                    },
                    "fluid_type": fluid_type,
                    "sg": sg,
                    "head_m": hyd_calc["head_m"],
                    "efficiency_percent": hyd_calc["efficiency_percent"],
                    "npsh_margin_m": hyd_result["details"].get("npsh_margin_m", 0),
                    "diagnosis": hyd_result["diagnosis"],
                    "confidence": hyd_result["confidence"],
                    "severity": hyd_result["severity"],
                    "estimation_note": " | ".join(estimation_notes) if estimation_notes else "Data OEM lengkap"
                }
                st.success(f"✅ {hyd_result['diagnosis']} ({hyd_result['confidence']}%)")
        
        if "hyd_result" in st.session_state:
            result = st.session_state.hyd_result
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Diagnosis", result["diagnosis"])
            with col_b:
                st.metric("Severity", {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(result["severity"],"⚪"))
            with col_c:
                st.metric("Domain", "Hydraulic")
            if result["diagnosis"] != "NORMAL_OPERATION":
                st.info(get_hydraulic_recommendation(result["diagnosis"], fluid_type, result["severity"]))

    # TAB 3: ELECTRICAL
    with tab_elec:
        st.header("⚡ Electrical Condition Analysis")
        st.caption("3-Phase Voltage/Current | Unbalance Detection")
        
        with st.expander("⚙️ Motor Nameplate", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                rated_voltage = st.number_input("Rated Voltage (V)", min_value=200, max_value=690, 
                                                value=400, step=10, key="rated_v")
            with col2:
                fla = st.number_input("Full Load Amps - FLA (A)", min_value=10, max_value=500, 
                                      value=85, step=5, key="rated_i")
        
        st.subheader("📊 Pengukuran 3-Phase")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Voltage (Line-to-Line)")
            v_l1l2 = st.number_input("L1-L2 (V)", min_value=0.0, value=400.0, step=1.0, key="v_l1l2")
            v_l2l3 = st.number_input("L2-L3 (V)", min_value=0.0, value=402.0, step=1.0, key="v_l2l3")
            v_l3l1 = st.number_input("L3-L1 (V)", min_value=0.0, value=398.0, step=1.0, key="v_l3l1")
        with col2:
            st.caption("Current (Per Phase)")
            i_l1 = st.number_input("L1 (A)", min_value=0.0, value=82.0, step=0.5, key="i_l1")
            i_l2 = st.number_input("L2 (A)", min_value=0.0, value=84.0, step=0.5, key="i_l2")
            i_l3 = st.number_input("L3 (A)", min_value=0.0, value=83.0, step=0.5, key="i_l3")
        
        if st.button("⚡ Generate Electrical Diagnosis", type="primary", key="run_elec"):
            with st.spinner("Menganalisis kondisi electrical..."):
                elec_calc = calculate_electrical_parameters(
                    v_l1l2, v_l2l3, v_l3l1, i_l1, i_l2, i_l3, 
                    rated_voltage, fla
                )
                motor_specs = {
                    "rated_voltage": rated_voltage,
                    "fla": fla
                }
                elec_result = diagnose_electrical_condition(elec_calc, motor_specs)
                st.session_state.elec_result = elec_result
                st.session_state.elec_data = {
                    "measurements": {
                        "v_l1l2": v_l1l2, "v_l2l3": v_l2l3, "v_l3l1": v_l3l1,
                        "i_l1": i_l1, "i_l2": i_l2, "i_l3": i_l3
                    },
                    "voltage_unbalance": elec_calc["voltage_unbalance_percent"],
                    "current_unbalance": elec_calc["current_unbalance_percent"],
                    "load_estimate": elec_calc["load_estimate_percent"],
                    "diagnosis": elec_result["diagnosis"],
                    "confidence": elec_result["confidence"],
                    "severity": elec_result["severity"]
                }
                st.success(f"✅ {elec_result['diagnosis']} ({elec_result['confidence']}%)")
        
        if "elec_result" in st.session_state:
            result = st.session_state.elec_result
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Diagnosis", result["diagnosis"])
            with col_b:
                st.metric("Severity", {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(result["severity"],"⚪"))
            with col_c:
                st.metric("Domain", "Electrical")
            if result["diagnosis"] != "NORMAL_ELECTRICAL":
                st.info(get_electrical_recommendation(result["diagnosis"], result["severity"]))

    # TAB 4: INTEGRATED
    with tab_integrated:
        st.header("🔗 Integrated Diagnostic Summary")
        st.caption("Cross-Domain Correlation | Temperature Analysis | Multi-Point Support")
        
        analyses_complete = all([
            "mech_result" in st.session_state,
            "hyd_result" in st.session_state,
            "elec_result" in st.session_state
        ])
        
        if not analyses_complete:
            st.info("""
            💡 **Langkah Selanjutnya:**
            1. Jalankan analisis di tab **🔧 Mechanical**
            2. Jalankan analisis di tab **💧 Hydraulic**
            3. Jalankan analisis di tab **⚡ Electrical**
            4. Kembali ke tab ini untuk integrated diagnosis
            """)
            col1, col2, col3 = st.columns(3)
            with col1:
                status_mech = "✅" if "mech_result" in st.session_state else "⏳"
                st.metric("Mechanical", status_mech)
            with col2:
                status_hyd = "✅" if "hyd_result" in st.session_state else "⏳"
                st.metric("Hydraulic", status_hyd)
            with col3:
                status_elec = "✅" if "elec_result" in st.session_state else "⏳"
                st.metric("Electrical", status_elec)
        else:
            with st.spinner("Mengintegrasikan hasil tiga domain..."):
                temp_data = st.session_state.get("temp_data", None)
                integrated_result = aggregate_cross_domain_diagnosis(
                    st.session_state.mech_result,
                    st.session_state.hyd_result,
                    st.session_state.elec_result,
                    st.session_state.shared_context,
                    temp_data
                )
                st.session_state.integrated_result = integrated_result
            
            st.subheader("📊 Overall Assessment")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; border-left:5px solid #1E3A5F; text-align:center;">
                    <h4 style="margin:0 0 10px 0; color:#1E3A5F">🔗 Integrated Diagnosis</h4>
                    <p style="margin:0; font-size:1.1em; font-weight:600; color:#2c3e50;">
                    {integrated_result["diagnosis"]}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                severity_config = {
                    "Low": ("🟢", "#27ae60"),
                    "Medium": ("🟠", "#f39c12"),
                    "High": ("🔴", "#c0392b")
                }
                sev_icon, sev_color = severity_config.get(integrated_result["severity"], ("⚪", "#95a5a6"))
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; border-left:5px solid {sev_color}; text-align:center;">
                    <h4 style="margin:0 0 10px 0; color:#1E3A5F">⚠️ Overall Severity</h4>
                    <p style="margin:0; font-size:1.5em; font-weight:700; color:{sev_color};">
                    {sev_icon} {integrated_result["severity"]}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("Confidence", f"{integrated_result['confidence']}%")
            with col4:
                correlation_text = "Detected" if integrated_result['correlation_notes'] and integrated_result['correlation_notes'][0] != "Tidak ada korelasi kuat antar domain terdeteksi" else "None"
                st.metric("Cross-Domain Correlation", correlation_text)
            with col5:
                temp_status = "Available" if temp_data else "N/A"
                st.metric("Temperature Data", temp_status)
            
            affected_points = integrated_result.get("affected_points", [])
            if affected_points and affected_points != ["Tidak Ada (Normal)"]:
                st.warning(f"📍 **Titik Terpengaruh:** {', '.join(affected_points)}")
            
            # ========================================================================
            # 🔥 FAULT PROPAGATION MAP DISPLAY (FIXED - NATIVE COMPONENTS)
            # ========================================================================
            st.divider()
            st.subheader("🗺️ Fault Propagation Map untuk Perbaikan")
            st.caption("Rantai fault dari root cause ke effect + action perbaikan yang diperlukan")
            
            propagation_map = generate_fault_propagation_map(
                st.session_state.mech_result,
                st.session_state.hyd_result,
                st.session_state.elec_result,
                temp_data
            )
            
            if propagation_map:
                for idx, prop in enumerate(propagation_map, 1):
                    # Tentukan warna box berdasarkan priority
                    priority = prop["priority"]
                    if priority == "CRITICAL":
                        box_color = "error"
                        priority_icon = "🔴"
                    elif priority == "HIGH":
                        box_color = "warning"
                        priority_icon = "🟠"
                    elif priority == "MEDIUM":
                        box_color = "info"
                        priority_icon = "🟡"
                    else:
                        box_color = "success"
                        priority_icon = "🟢"
                    
                    # Gunakan Container untuk membuat efek Kartu
                    with st.container():
                        # Header Kartu
                        col_h1, col_h2 = st.columns([3, 1])
                        with col_h1:
                            st.markdown(f"**{priority_icon} Scenario {idx}: {prop['root_cause']}**")
                        with col_h2:
                            st.markdown(f"`Priority: {priority}` | `Timeline: {prop['timeline']}`")
                        
                        st.markdown("---") # Garis pemisah
                        
                        # Fault Chain (Menggunakan Columns agar sejajar)
                        st.markdown("**🔗 Fault Chain:**")
                        n_nodes = len(prop["fault_chain"])
                        chain_cols = st.columns(n_nodes)
                        for i, fault in enumerate(prop["fault_chain"]):
                            with chain_cols[i]:
                                # Tampilkan node fault dengan box sederhana
                                st.markdown(
                                    f"""
                                    <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; 
                                    text-align:center; border:1px solid #ddd; height:100%;">
                                    <b>{fault}</b>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                                # Tambah panah jika bukan node terakhir
                                if i < n_nodes - 1:
                                    st.markdown("<div style='text-align:center; margin-top:-5px; margin-bottom:5px;'>⬇️</div>", unsafe_allow_html=True)
                        
                        st.markdown("---") # Garis pemisah
                        
                        # Repair Actions
                        st.markdown("**🔧 Repair Actions:**")
                        for action in prop["repair_actions"]:
                            clean_action = action.replace("✅ ", "").strip()
                            st.markdown(f"✅ {clean_action}")
                    
                    st.markdown("<br>", unsafe_allow_html=True) # Jarak antar kartu
            else:
                st.info("ℹ️ Tidak ada fault propagation map yang dihasilkan. Semua domain dalam kondisi normal.")

            st.divider()
            # ========================================================================
            st.subheader("📥 Export Report")
            if st.button("📊 Generate Unified CSV Report", type="primary"):
                csv_report = generate_unified_csv_report(
                    machine_id,
                    rpm,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    st.session_state.get("mech_data", {}),
                    st.session_state.get("hyd_data", {}),
                    st.session_state.get("elec_data", {}),
                    integrated_result,
                    temp_data
                )
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_report,
                    file_name=f"PUMP_DIAG_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("✅ Report generated successfully!")
            
            st.divider()
            st.caption("""
            **Standar Acuan**: ISO 10816-3/7 | ISO 13373-1 | API 610 | IEC 60034 | API 670
            **Algoritma**: Hybrid rule-based dengan cross-domain correlation + confidence scoring
            ⚠️ Decision Support System - Verifikasi oleh personnel kompeten untuk keputusan kritis
            🏭 Pertamina Patra Niaga - Asset Integrity Management
            """)

if __name__ == "__main__":
    main()
