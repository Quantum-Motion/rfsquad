DEVICE_REQUIRED = ["C_dd", "C_dg"]
DEVICE_ALLOWED = [
    "C_dd",
    "C_dg",
    "dot_names",
    "gate_names",
    "plunger_gates",
    "barrier_gates",
    "gate_voltages",
    "threshold_voltages",
    "virtual_gate_matrix",
]

SOLVER_ALLOWED = [
    "max_charge_carriers",
    "reduce_fock_space",
    "use_virtual_gates",
    "max_fock_depth",
]

LABEL_MAP = {
    "cim": "Energy (eV)",  # defaults
    "energy": "Energy (eV)",
    "occupancy": "Occupancy",
    "rf": r"$\Delta C$ (1/V)",
}

CSD_REQUIRED = [
    "sweep",
    "raw_data",
    "solver",
    "virtual_gates",
    "fock_states",
]

CSD_ALLOWED = [
    "sweep",
    "raw_data",
    "solver",
    "virtual_gates",
    "fock_states",
    "temperature",
    "sensor_gate",
    "data",
]

NOISE_ALLOWED = [
    "amplitude",
    "mean",
    "std",
    "time",
    "sweep_gate_order",
    "random_seed",
    "alpha",
    "fmin",
]

NOISE_REQUIRED = ["amplitude"]
