"""
Bit-accurate golden model and test vector generator for INT8 PE MAC.
Project: Configurable INT8 Approximate PE (RTL to GDSII) - Day 3
"""

import numpy as np

# Bit limits
INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def quantize_to_int8(tensor: np.ndarray, scale: float) -> np.ndarray:
    """
    Symmetric per-tensor quantization: q = clamp(round(x / scale), -128, 127)
    """
    q = np.round(tensor / scale)
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)


def dequantize(q_tensor: np.ndarray, scale: float) -> np.ndarray:
    """Reconstruct float representation from quantized INT8 tensor."""
    return q_tensor.astype(np.float32) * scale


def pe_mac_exact(
    weight: int, act: int, acc_in: int, saturate: bool = False
) -> int:
    """
    Bit-accurate exact INT8 MAC operation.
    product = weight * act (signed 16-bit)
    acc_out = acc_in + product (signed 32-bit wrap or saturate)
    """
    prod = int(np.int16(np.int8(weight)) * np.int16(np.int8(act)))
    raw_acc = acc_in + prod

    if saturate:
        if raw_acc > INT32_MAX:
            return INT32_MAX
        elif raw_acc < INT32_MIN:
            return INT32_MIN
        return raw_acc
    else:
        # Standard 32-bit two's complement wrap-around
        return int(np.int32(raw_acc))


def to_hex2(val: int) -> str:
    """Format 8-bit signed integer as 2-digit hex."""
    return f"{int(np.uint8(np.int8(val))):02X}"


def to_hex8(val: int) -> str:
    """Format 32-bit signed integer as 8-digit hex."""
    return f"{int(np.uint32(np.int32(val))):08X}"

"""
Bit-accurate golden model and test vector generator for INT8 PE MAC.
Project: Configurable INT8 Approximate PE (RTL to GDSII) - Day 3
"""

import numpy as np

# Bit limits
INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def quantize_to_int8(tensor: np.ndarray, scale: float) -> np.ndarray:
    """
    Symmetric per-tensor quantization: q = clamp(round(x / scale), -128, 127)
    """
    q = np.round(tensor / scale)
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)


def dequantize(q_tensor: np.ndarray, scale: float) -> np.ndarray:
    """Reconstruct float representation from quantized INT8 tensor."""
    return q_tensor.astype(np.float32) * scale


def pe_mac_exact(
    weight: int, act: int, acc_in: int, saturate: bool = False
) -> int:
    """
    Bit-accurate exact INT8 MAC operation.
    product = weight * act (signed 16-bit)
    acc_out = acc_in + product (signed 32-bit wrap or saturate)
    """
    prod = int(np.int16(np.int8(weight)) * np.int16(np.int8(act)))
    raw_acc = acc_in + prod

    if saturate:
        if raw_acc > INT32_MAX:
            return INT32_MAX
        elif raw_acc < INT32_MIN:
            return INT32_MIN
        return raw_acc
    else:
        # Standard 32-bit two's complement wrap-around
        return int(np.int32(raw_acc))


def to_hex2(val: int) -> str:
    """Format 8-bit signed integer as 2-digit hex."""
    return f"{int(np.uint8(np.int8(val))):02X}"


def to_hex8(val: int) -> str:
    """Format 32-bit signed integer as 8-digit hex."""
    return f"{int(np.uint32(np.int32(val))):08X}"

def generate_corner_cases() -> list[tuple[int, int, int, int]]:
    """
    Generates directed corner cases for INT8 MAC verification:
      - Zeros (0 * 0, 0 * max, max * 0)
      - Extremes (+127 * +127, -128 * -128, -128 * +127)
      - Mixed and alternating signs - Accumulator near-overflow boundary tests
    Returns list of tuples: (weight, act, acc_in, expected_acc_out)
    """
    vectors = []

    corner_values = [0, 127, -128]

    # 1. Full cartesian product of 8-bit edge values with 0 accumulator
    for w in corner_values:
        for a in corner_values:
            acc_in = 0
            exp = pe_mac_exact(w, a, acc_in)
            vectors.append((w, a, acc_in, exp))

    # 2. Asymmetric extreme: -128 * -128 = +16384
    for acc in [0, 1000, -1000]:
        vectors.append((-128, -128, acc, pe_mac_exact(-128, -128, acc)))

    # 3. Alternating sign stress sequences (positive and negative cancellation)
    running_acc = 0
    pattern = [
        (127, 127),
        (-128, 127),
        (127, -128),
        (-128, -128),
        (-1, 1),
        (1, -1),
    ]
    for w, a in pattern:
        exp = pe_mac_exact(w, a, running_acc)
        vectors.append((w, a, running_acc, exp))
        running_acc = exp

    # 4. 32-bit accumulator boundary & wrap-around checks
    acc_edges = [
        INT32_MAX,
        INT32_MAX - 100,
        INT32_MIN,
        INT32_MIN + 100,
        0x7FFF0000,
        -0x7FFF0000,
    ]
    for acc in acc_edges:
        vectors.append((127, 127, acc, pe_mac_exact(127, 127, acc)))
        vectors.append((-128, 127, acc, pe_mac_exact(-128, 127, acc)))
        vectors.append((-128, -128, acc, pe_mac_exact(-128, -128, acc)))

    return vectors


def export_vectors_hex(
    vectors: list[tuple[int, int, int, int]], filename: str = "vectors.hex"
):
    """
    Export stimulus in Verilog $readmemh compatible format.
    Line format: <weight_hex:2> <act_hex:2> <acc_in_hex:8> <expected_acc_out_hex:8>
    """
    with open(filename, "w") as f:
        for w, a, acc_in, exp in vectors:
            f.write(
                f"{to_hex2(w)} {to_hex2(a)} {to_hex8(acc_in)} {to_hex8(exp)}\n"
            )
    print(f"Exported {len(vectors)} test vectors to {filename}")


if __name__ == "__main__":
    vecs = generate_corner_cases()
    export_vectors_hex(vecs, "vectors.hex")