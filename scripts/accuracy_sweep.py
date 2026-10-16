import csv
import numpy as np
import matplotlib.pyplot as plt

def truncate_bits(values: np.ndarray, bits_to_truncate: int) -> np.ndarray:
    """
    Simulates hardware truncation by clearing the lowest N bits.
    Works for fixed-point integer representations.
    """
    if bits_to_truncate == 0:
        return values
    mask = ~((1 << bits_to_truncate) - 1)
    return values & mask

def run_inference_simulation(inputs: np.ndarray, weights: np.ndarray, bits_to_truncate: int) -> np.ndarray:
    """
    Simulates a quantized/fixed-point layer: (Inputs * Weights) with truncation.
    """
    approx_weights = truncate_bits(weights, bits_to_truncate)
    approx_inputs = truncate_bits(inputs, bits_to_truncate)
    
    # Simple dot product / linear accumulation
    raw_outputs = np.dot(approx_inputs, approx_weights.T)
    # Output class prediction via argmax
    return np.argmax(raw_outputs, axis=1)

def evaluate_accuracy(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.mean(predictions == ground_truth) * 100.0)

def main():
    # 1. Generate / Mock fixed-point INT8 test data (Replace with actual dataset/weights)
    np.random.seed(42)
    num_samples = 10000
    features = 64
    num_classes = 10

    # Simulating 8-bit unsigned/fixed-point activations and weights [0, 255]
    test_inputs = np.random.randint(0, 256, size=(num_samples, features), dtype=np.int32)
    test_weights = np.random.randint(0, 256, size=(num_classes, features), dtype=np.int32)
    
    # Ground truth labels
    ground_truth = np.random.randint(0, num_classes, size=(num_samples,))
    
    # Set weights such that baseline achieves ~98.4% by design for demo
    # (In production, replace with your actual model inference call)
    baseline_preds = run_inference_simulation(test_inputs, test_weights, bits_to_truncate=0)
    # Align a portion of ground_truth to mimic a realistic ~98.4% baseline
    match_mask = np.random.rand(num_samples) < 0.984
    ground_truth = np.where(match_mask, baseline_preds, ground_truth)

    # 2. Accuracy Sweep over bit truncation levels (0 to 6 bits)
    truncation_levels = [0, 1, 2, 3, 4, 5, 6]
    sweep_results = []

    baseline_accuracy = evaluate_accuracy(baseline_preds, ground_truth)
    print(f"Baseline Exact Accuracy: {baseline_accuracy:.2f}%\n")

    for bits in truncation_levels:
        preds = run_inference_simulation(test_inputs, test_weights, bits_to_truncate=bits)
        acc = evaluate_accuracy(preds, ground_truth)
        acc_loss = baseline_accuracy - acc
        
        sweep_results.append({
            "truncated_bits": bits,
            "accuracy_pct": acc,
            "accuracy_loss_pct": max(0.0, acc_loss)
        })
        print(f"Truncated Bits: {bits:2d} | Accuracy: {acc:6.2f}% | Loss: {acc_loss:5.2f}%")

    # 3. Export to CSV
    csv_file = "accuracy_sweep_results.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["truncated_bits", "accuracy_pct", "accuracy_loss_pct"])
        writer.writeheader()
        writer.writerows(sweep_results)
    print(f"\n[+] Results exported to {csv_file}")

    # 4. Generate Matplotlib Tradeoff Curve
    bits_x = [row["truncated_bits"] for row in sweep_results]
    loss_y = [row["accuracy_loss_pct"] for row in sweep_results]
    acc_y = [row["accuracy_pct"] for row in sweep_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:red"
    ax1.set_xlabel("Approximation Level (Bits Truncated)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Accuracy Loss (%)", color=color, fontsize=11, fontweight="bold")
    line1 = ax1.plot(bits_x, loss_y, marker="o", linewidth=2, color=color, label="Accuracy Loss (%)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Dual axis to view absolute accuracy side-by-side
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Model Accuracy (%)", color=color, fontsize=11, fontweight="bold")
    line2 = ax2.plot(bits_x, acc_y, marker="s", linestyle="--", linewidth=2, color=color, label="Accuracy (%)")
    ax2.tick_params(axis="y", labelcolor=color)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left")

    plt.title("Approximation Tradeoff: Accuracy Loss vs. Truncation Degree", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("accuracy_tradeoff_curve.png", dpi=300)
    print("[+] Plot saved to accuracy_tradeoff_curve.png")
    plt.show()

if __name__ == "__main__":
    main()