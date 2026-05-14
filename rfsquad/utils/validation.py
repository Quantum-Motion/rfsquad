import jax.numpy as jnp


def validate_capacitance_matrix_shapes(C_dd, C_dg, num_dots, num_gates) -> None:
    if C_dd.shape != (num_dots, num_dots):
        raise ValueError(
            "Dot-dot capacitance matrix must be of a num_dots x num_dots shape."
        )

    if C_dg.shape[0] != num_dots:
        raise ValueError(f"Dot-gate capacitance matrix must have {num_dots} rows.")

    if num_gates < num_dots:
        raise ValueError(
            f"Dot-gate capacitance matrix must have {num_dots} or more columns (at least one gate per dot)."
        )


def validate_capacitance_matrix_elements(C_dd) -> None:
    if not jnp.all(C_dd[jnp.diag_indices_from(C_dd)] > 0):
        raise ValueError(
            "Dot-dot capacitance matrix diagonal elements must all be positive."
        )

    mask = jnp.where(~jnp.eye(C_dd.shape[0], dtype=bool))
    if not jnp.all(C_dd[mask] < 0):
        raise ValueError(
            "Dot-dot capacitance matrix off-diagonal elements must all be negative."
        )

    if not jnp.all(jnp.equal(C_dd, jnp.transpose(C_dd))):
        raise ValueError("Dot-dot capacitance matrix must be symmetric.")


def validate_gate_definitions(gate_names, num_gates) -> None:
    if len(gate_names) != len(set(gate_names)):
        raise ValueError("Gate names must be unique.")

    if len(gate_names) != num_gates:
        raise ValueError(
            f"Number of named gates ({len(gate_names)}) must match the number of defined gates ({num_gates})."
        )


def validate_threshold_gate_definitions(gate_names, gate_voltages, threshold_voltages):
    kgv = set(gate_voltages.keys())
    ktv = set(threshold_voltages.keys())
    kgn = set(gate_names)

    if not (kgv.issubset(kgn) and ktv.issubset(kgn)):
        raise KeyError(
            "Gate voltage keys and threshold voltage keys must be a subset of gate names."
        )


def validate_barrier_plunger_gates(gate_names, plunger_gates, barrier_gates):
    if barrier_gates is not None:
        if set(barrier_gates).intersection(set(plunger_gates)):
            raise ValueError(
                "Plunger gates and barrier gates must be mutually exclusive."
            )

        if not set(gate_names) == set(plunger_gates).union(set(barrier_gates)):
            raise ValueError("Each gate must be defined as either plunger or barrier.")


def validate_virtual_gate_matrix(virtual_gate_matrix, num_gates):
    if virtual_gate_matrix is not None:
        if not isinstance(virtual_gate_matrix, jnp.ndarray):
            raise TypeError("Virtual gate matrix must be supplied as a jnp.ndarray.")

        if virtual_gate_matrix.shape != (num_gates, num_gates):
            raise ValueError(
                f"Virtual gate matrix is not the correct shape - it should be a {num_gates}x{num_gates} square array."
            )
