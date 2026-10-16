"""
Day 6 - CNN Vector Prep
Train a tiny LeNet-5 on MNIST, quantize to INT8, and dump layer-1 / layer-2
weights + activations into lenet_int8.hex for a Verilog/SystemVerilog testbench.

Setup:   pip install torch torchvision
Run:     python lenet_int8_dump.py
Output:  lenet_int8.hex   (load with $readmemh)
         lenet_int8_map.txt (start address / size of every tensor + scales)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(0)

# ---------------------------------------------------------------- 1. Data
# MNIST = 28x28 grayscale handwritten digits (0-9). ToTensor() gives values 0..1.
tfm = transforms.ToTensor()
train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1000)


# ---------------------------------------------------------------- 2. Model
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # bias=False on the two conv layers keeps the hardware vectors simple
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2, bias=False)  # 1x28x28 -> 6x28x28
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)            # 6x14x14 -> 16x10x10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # -> 6x14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # -> 16x5x5
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


model = LeNet5()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------- 3. Train
for epoch in range(2):  # 2 epochs is plenty (~98% accuracy)
    model.train()
    for x, y in train_dl:
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch + 1}: test accuracy {correct / len(test_ds):.4f}")


# ---------------------------------------------------------------- 4. Quantize
# Symmetric per-tensor INT8:  q = round(x / scale),  scale = max|x| / 127
def quantize(t, scale, lo=-127, hi=127):
    return torch.clamp(torch.round(t / scale), lo, hi)


w1, w2 = model.conv1.weight.detach(), model.conv2.weight.detach()
s_w1 = w1.abs().max().item() / 127
s_w2 = w2.abs().max().item() / 127
w1_q, w2_q = quantize(w1, s_w1), quantize(w2, s_w2)

# Calibrate activation ranges on 1000 training images
with torch.no_grad():
    xs = torch.stack([train_ds[i][0] for i in range(1000)])
    a1 = F.relu(model.conv1(xs))
    a2 = F.relu(model.conv2(F.max_pool2d(a1, 2)))
s_x = 1 / 127                      # input pixels are 0..1
s_a1 = a1.max().item() / 127
s_a2 = a2.max().item() / 127

# ---------------------------------------------------------------- 5. Integer reference run
# Values are stored as floats but are always whole numbers, so this matches
# what INT8 hardware with a wide accumulator would compute.
img, label = test_ds[0]            # first test image (a "7")
x_in = img.unsqueeze(0)            # 1x1x28x28
x_q = quantize(x_in, s_x, 0, 127)

with torch.no_grad():
    # Layer 1: conv -> ReLU -> requantize to INT8
    acc1 = F.conv2d(x_q, w1_q, padding=2)                       # INT32 accumulators
    a1_q = torch.clamp(torch.round(F.relu(acc1) * (s_x * s_w1 / s_a1)), 0, 127)
    p1_q = F.max_pool2d(a1_q, 2)                                # layer-2 input

    # Layer 2: conv -> ReLU -> requantize to INT8
    acc2 = F.conv2d(p1_q, w2_q)
    a2_q = torch.clamp(torch.round(F.relu(acc2) * (s_a1 * s_w2 / s_a2)), 0, 127)

    # Sanity check: dequantized INT8 result should be close to the float model
    err = (a1_q * s_a1 - F.relu(model.conv1(x_in))).abs().max().item()
    print(f"layer-1 max abs error vs float: {err:.4f} (scale step = {s_a1:.4f})")

# ---------------------------------------------------------------- 6. Dump to hex
tensors = [
    ("input_img_int8      [1][28][28]", x_q),
    ("conv1_weights_int8  [6][1][5][5]", w1_q),
    ("conv1_out_int8      [6][28][28] (post-ReLU, pre-pool)", a1_q),
    ("conv2_in_int8       [6][14][14] (= pooled conv1 out)", p1_q),
    ("conv2_weights_int8  [16][6][5][5]", w2_q),
    ("conv2_out_int8      [16][10][10] (post-ReLU, pre-pool)", a2_q),
]

addr = 0
map_lines = []
with open("lenet_int8.hex", "w") as f:
    for name, t in tensors:
        flat = t.flatten().to(torch.int32).tolist()
        f.write(f"// {name}, {len(flat)} bytes\n@{addr:X}\n")
        for v in flat:
            f.write(f"{v & 0xFF:02x}\n")      # two's complement byte
        map_lines.append(f"0x{addr:05X}  {len(flat):6d} bytes  {name}")
        addr += len(flat)

with open("lenet_int8_map.txt", "w") as f:
    f.write("\n".join(map_lines) + "\n\n")
    f.write(f"scales: s_x={s_x:.8f} s_w1={s_w1:.8f} s_a1={s_a1:.8f} "
            f"s_w2={s_w2:.8f} s_a2={s_a2:.8f}\n")
    f.write(f"label of dumped image: {label}\n")

print("\n".join(map_lines))
print("wrote lenet_int8.hex and lenet_int8_map.txt")