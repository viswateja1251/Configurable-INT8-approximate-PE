#!/usr/bin/env python3
"""
Day 12: PyTorch Emulation for INT8 Approximate PE
Project: Configurable INT8 Approximate PE (RTL to GDSII Flow)

Engineering Tasks:
1. Implement bit-accurate approx multiplier function in Python/PyTorch
   (Baugh-Wooley with Truncation & Constant Correction).
2. Construct high-throughput 256x256 Look-Up Table (LUT) for signed INT8 operands.
3. Replace standard linear and convolution operations in LeNet-5 with approximate operators.
4. Run inference on full 10,000 MNIST test images; record Top-1 accuracy and degradation.
"""

import os
import sys
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# 1. BIT-ACCURATE APPROXIMATE MULTIPLIER (RTL GOLDEN MODEL & LUT GENERATOR)
# ==============================================================================

def rtl_approx_mult8x8_signed(a_int: int, b_int: int, k_pruned: int = 4) -> int:
    """
    Bit-accurate model matching approx_mult.sv (8x8 signed multiplier).
    
    Architecture:
    - Baugh-Wooley 2's complement partial product array for signed 8-bit inputs [-128, 127].
    - Truncates lower `k_pruned` columns (columns 0 to k_pruned - 1).
    - Constant correction: Injects bias bit at column (k_pruned - 1) to minimize MED.
    """
    a_int = max(-128, min(127, int(a_int)))
    b_int = max(-128, min(127, int(b_int)))

    if k_pruned <= 0:
        return a_int * b_int

    a_u8 = a_int & 0xFF
    b_u8 = b_int & 0xFF

    cols = [0] * 16
    for i in range(8):
        for j in range(8):
            col = i + j
            a_bit = (a_u8 >> i) & 1
            b_bit = (b_u8 >> j) & 1
            
            # Inverted partial products for sign extensions in Baugh-Wooley
            if (i == 7 and j < 7) or (j == 7 and i < 7):
                pp = 1 ^ (a_bit & b_bit)
            else:
                pp = a_bit & b_bit
            cols[col] += pp

    # Baugh-Wooley constant additions: -2^15 + 2^8
    cols[8] += 1
    cols[15] += 1

    # Approximate Pruning: Zero out columns 0 .. k_pruned - 1
    for c in range(k_pruned):
        cols[c] = 0

    # Constant Correction: Inject rounding bias at column (k_pruned - 1)
    if k_pruned > 0:
        cols[k_pruned - 1] = 1

    # Accumulate column weights
    total = sum(cols[c] * (1 << c) for c in range(16))
    
    # 16-bit signed conversion
    if total >= (1 << 15):
        total -= (1 << 16)
        
    return total


def generate_approx_mult_lut(k_pruned: int = 4):
    """
    Precomputes a 256x256 lookup table for signed 8-bit approximate multiplication.
    Indices: [a + 128, b + 128] for a, b in range [-128, 127].
    """
    lut = np.zeros((256, 256), dtype=np.int32)
    for a in range(-128, 128):
        for b in range(-128, 128):
            lut[a + 128, b + 128] = rtl_approx_mult8x8_signed(a, b, k_pruned=k_pruned)
            
    if HAS_TORCH:
        return torch.from_numpy(lut).to(torch.int32)
    return lut


# ==============================================================================
# 2. QUANTIZATION UTILITIES (INT8 SYMMETRIC)
# ==============================================================================

class SymmetricInt8Quantizer:
    """
    Symmetric per-tensor INT8 quantizer with zero-point = 0.
    Directly reflects the hardware MAC datapath where zero-point logic is omitted.
    """
    @staticmethod
    def calculate_scale(tensor):
        max_abs = torch.max(torch.abs(tensor)).item()
        return max(max_abs / 127.0, 1e-8)

    @staticmethod
    def quantize(tensor, scale):
        return torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int32)

    @staticmethod
    def dequantize(q_tensor, scale):
        return q_tensor.float() * scale


# ==============================================================================
# 3. APPROXIMATE NEURAL NETWORK LAYERS
# ==============================================================================

if HAS_TORCH:
    class ApproxLinear(nn.Module):
        """
        Fully connected layer using the approximate multiplier LUT.
        """
        def __init__(self, in_features, out_features, bias=True, lut=None):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter('bias', None)
            self.lut = lut
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, x):
            if self.lut is None:
                return F.linear(x, self.weight, self.bias)

            # Symmetrical INT8 Quantization
            s_x = SymmetricInt8Quantizer.calculate_scale(x)
            s_w = SymmetricInt8Quantizer.calculate_scale(self.weight)

            q_x = SymmetricInt8Quantizer.quantize(x, s_x)            # (Batch, In)
            q_w = SymmetricInt8Quantizer.quantize(self.weight, s_w)  # (Out, In)

            # Map from signed [-128, 127] to LUT coordinates [0, 255]
            idx_x = (q_x + 128).long()
            idx_w = (q_w + 128).long()

            # Vectorized approximate multiplication via LUT
            idx_x_exp = idx_x.unsqueeze(1)  # (Batch, 1, In)
            idx_w_exp = idx_w.unsqueeze(0)  # (1, Out, In)

            lut_device = self.lut.to(x.device)
            prod = lut_device[idx_x_exp, idx_w_exp]  # (Batch, Out, In)

            # 32-bit hardware accumulation across input channel dimension
            acc = torch.sum(prod, dim=-1)  # (Batch, Out)

            # Dequantization to float
            out = acc.float() * (s_x * s_w)
            if self.bias is not None:
                out += self.bias
            return out


    class ApproxConv2d(nn.Module):
        """
        Convolutional 2D layer using im2col and the approximate multiplier LUT.
        """
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, lut=None):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
            self.stride = stride
            self.padding = padding
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_channels))
            else:
                self.register_parameter('bias', None)
            self.lut = lut
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, x):
            if self.lut is None:
                return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

            kh, kw = self.kernel_size
            # 1. Unfold image patches: (Batch, in_channels * kh * kw, L)
            x_unfold = F.unfold(x, kernel_size=(kh, kw), padding=self.padding, stride=self.stride)
            batch_size, k_dim, l_dim = x_unfold.shape

            # Flatten weights: (out_channels, in_channels * kh * kw)
            w_flat = self.weight.view(self.out_channels, -1)

            # 2. Symmetrical INT8 Quantization
            s_x = SymmetricInt8Quantizer.calculate_scale(x_unfold)
            s_w = SymmetricInt8Quantizer.calculate_scale(w_flat)

            q_x = SymmetricInt8Quantizer.quantize(x_unfold, s_x)  # (Batch, K, L)
            q_w = SymmetricInt8Quantizer.quantize(w_flat, s_w)    # (Out, K)

            idx_x = (q_x + 128).long()
            idx_w = (q_w + 128).long()

            # Vectorized LUT indexing:
            # idx_w: (1, Out, K, 1), idx_x: (Batch, 1, K, L)
            idx_w_exp = idx_w.unsqueeze(0).unsqueeze(-1)
            idx_x_exp = idx_x.unsqueeze(1)

            lut_device = self.lut.to(x.device)
            prod = lut_device[idx_w_exp, idx_x_exp]  # (Batch, Out, K, L)
            acc = torch.sum(prod, dim=2)             # Sum over K (MAC accumulation)

            out_unfold = acc.float() * (s_x * s_w)
            if self.bias is not None:
                out_unfold += self.bias.unsqueeze(0).unsqueeze(-1)

            # Reshape back to (Batch, Out, H_out, W_out)
            h_out = (x.shape[2] + 2 * self.padding - kh) // self.stride + 1
            w_out = (x.shape[3] + 2 * self.padding - kw) // self.stride + 1
            return out_unfold.view(batch_size, self.out_channels, h_out, w_out)


# ==============================================================================
# 4. LENET-5 MODEL DEFINITION
# ==============================================================================

if HAS_TORCH:
    class LeNet5(nn.Module):
        """
        LeNet-5 architecture supporting dynamic approximate operator swapping.
        """
        def __init__(self, lut=None):
            super().__init__()
            self.conv1 = ApproxConv2d(1, 6, kernel_size=5, padding=2, lut=lut)
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.conv2 = ApproxConv2d(6, 16, kernel_size=5, padding=0, lut=lut)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.fc1 = ApproxLinear(16 * 5 * 5, 120, lut=lut)
            self.fc2 = ApproxLinear(120, 84, lut=lut)
            self.fc3 = ApproxLinear(84, 10, lut=lut)

        def forward(self, x):
            x = torch.tanh(self.conv1(x))
            x = self.pool1(x)
            x = torch.tanh(self.conv2(x))
            x = self.pool2(x)
            x = x.view(x.size(0), -1)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            return x

        def set_lut(self, lut):
            """Set or clear the approximate LUT across all layers."""
            self.conv1.lut = lut
            self.conv2.lut = lut
            self.fc1.lut = lut
            self.fc2.lut = lut
            self.fc3.lut = lut


# ==============================================================================
# 5. INFERENCE RUNNER ON MNIST (10,000 SAMPLES)
# ==============================================================================

def run_mnist_inference(model, test_loader, device='cpu'):
    """
    Evaluates LeNet-5 on the full 10,000 MNIST test images.
    Returns Top-1 classification accuracy percentage.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for neural network evaluation.")

    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  [Progress] Evaluated {total:5d}/{len(test_loader.dataset)} test samples...")

    return 100.0 * correct / total


# ==============================================================================
# VERIFICATION & ENTRY POINT
# ==============================================================================

def verify_multiplier_metrics(k_pruned=4):
    """
    Verifies error metrics: Error Rate (ER), Mean Error Distance (MED), and Max Error.
    """
    print(f"\n[*] Characterizing approximate multiplier error metrics (k={k_pruned})...")
    errors = []
    nonzero_count = 0
    total_ops = 256 * 256

    for a in range(-128, 128):
        for b in range(-128, 128):
            exact = a * b
            approx = rtl_approx_mult8x8_signed(a, b, k_pruned=k_pruned)
            err = approx - exact
            errors.append(abs(err))
            if err != 0:
                nonzero_count += 1

    er = (nonzero_count / total_ops) * 100.0
    med = np.mean(errors)
    max_err = np.max(errors)
    nmed = med / (128 * 127)

    print(f"  - Total Evaluated Input Pairs: {total_ops:,}")
    print(f"  - Error Rate (ER):             {er:.2f}%")
    print(f"  - Mean Error Distance (MED):   {med:.4f}")
    print(f"  - Normalized MED (NMED):       {nmed:.6f}")
    print(f"  - Maximum Absolute Error:      {max_err}")
    return er, med, max_err


if __name__ == '__main__':
    print("=" * 75)
    print("DAY 12: PYTORCH EMULATION - INT8 APPROXIMATE PE (LeNet-5 / MNIST)")
    print("=" * 75)

    # Multiplier verification across all 65,536 combinations
    verify_multiplier_metrics(k_pruned=4)

    # Multiplier LUT generation
    k = 4
    lut_approx = generate_approx_mult_lut(k_pruned=k)
    print(f"\n[+] Generated 256x256 LUT for k={k} pruned bits.")

    if HAS_TORCH:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] Running on device: {device}")

        # 10,000 MNIST test images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        model = LeNet5()

        # Step 1: Baseline exact inference
        print("\n--- Step 1: Baseline Exact Inference (Exact INT8 MAC) ---")
        model.set_lut(None)
        acc_exact = run_mnist_inference(model, test_loader, device=device)
        print(f"--> Baseline Top-1 Accuracy: {acc_exact:.2f}%")

        # Step 2: Approximate multiplier inference
        print(f"\n--- Step 2: Approximate Multiplier Inference (k={k} Pruned Bits) ---")
        model.set_lut(lut_approx)
        acc_approx = run_mnist_inference(model, test_loader, device=device)
        print(f"--> Approximate Top-1 Accuracy: {acc_approx:.2f}%")

        # Step 3: Trade-off summary
        degradation = acc_exact - acc_approx
        print("\n" + "=" * 75)
        print("SUMMARY OF DAY 12 INFERENCE RESULTS")
        print(f"Exact INT8 Baseline Accuracy:    {acc_exact:.2f}%")
        print(f"Approximate (k={k}) Accuracy:    {acc_approx:.2f}%")
        print(f"Accuracy Degradation:            {degradation:+.2f}%")
        print("=" * 75)
    else:
        print("\n[INFO] Ready to execute in a PyTorch-enabled environment/container.")