import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset
import torch.nn.functional as F

class ManualVGG16:
    def __init__(self):
        self.W = {}

        # Conv layers (no bias), Kaiming init
        conv_shapes = [
            (64, 3, 3, 3), (64, 64, 3, 3),        # Block 1
            (128, 64, 3, 3), (128, 128, 3, 3),    # Block 2
            (256, 128, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3),  # Block 3
            (512, 256, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3),  # Block 4
            (512, 512, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3),  # Block 5
        ]

        for i, shape in enumerate(conv_shapes):
            self.W[f'conv{i+1}'] = torch.randn(*shape) * (2.0 / (shape[1]*shape[2]*shape[3]))**0.5
            self.W[f'conv{i+1}'].requires_grad = False

        # FC layers
        self.W['fc1'] = torch.randn(512, 512) * (2.0 / 512)**0.5
        self.W['fc2'] = torch.randn(512, 512) * (2.0 / 512)**0.5
        self.W['fc3'] = torch.randn(512, 10) * (2.0 / 512)**0.5
        for k in ['fc1', 'fc2', 'fc3']:
            self.W[k].requires_grad = False

    def forward(self, x):
        self.cache = {}
        self.input = x
        idx = 1

        for block, layers in enumerate([(1, 2), (3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13)], start=1):
            for lid in layers:
                x = F.conv2d(x, self.W[f'conv{lid}'], padding=1)
                self.cache[f'z{lid}'] = x
                x = F.relu(x)
            x, pool_idx = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
            assert pool_idx.dtype == torch.int64
            self.cache[f'pool{block}'] = (x, pool_idx)

        x = x.view(x.size(0), -1)
        self.cache['flat'] = x

        z1 = x @ self.W['fc1']
        self.cache['fc1_z'] = z1
        a1 = F.relu(z1)

        z2 = a1 @ self.W['fc2']
        self.cache['fc2_z'] = z2
        a2 = F.relu(z2)

        logits = a2 @ self.W['fc3']
        self.cache['logits'] = logits

        return logits

    def backward_propogation(self, grad_logits, lr=0.01):
        W = self.W
        c = self.cache

        # FC3
        grad_a2 = grad_logits @ W['fc3'].T
        dW_fc3 = c['fc2_z'].T @ grad_logits
        with torch.no_grad():
            W['fc3'] -= lr * dW_fc3

        # FC2
        grad_z2 = grad_a2 * (c['fc2_z'] > 0)
        grad_a1 = grad_z2 @ W['fc2'].T
        dW_fc2 = c['fc1_z'].T @ grad_z2
        with torch.no_grad():
            W['fc2'] -= lr * dW_fc2

        # FC1
        grad_z1 = grad_a1 * (c['fc1_z'] > 0)
        grad_flat = grad_z1 @ W['fc1'].T
        dW_fc1 = c['flat'].T @ grad_z1
        with torch.no_grad():
            W['fc1'] -= lr * dW_fc1

        # reshape to conv5 output shape
        grad = grad_flat.view(c['pool5'][0].shape)

        # backward through conv blocks
        for block in reversed(range(1, 6)):
            pooled_out, indices = c[f'pool{block}']
            first_lid = [1, 3, 5, 8, 11][block - 1]

            # MaxUnpool to restore pre-pool shape
            grad = F.max_unpool2d(grad, indices, kernel_size=2, stride=2, output_size=c[f'z{first_lid}'].shape)

            # Get conv layer ids in this block
            if block == 1:
                conv_ids = [1, 2]
            elif block == 2:
                conv_ids = [3, 4]
            elif block == 3:
                conv_ids = [5, 6, 7]
            elif block == 4:
                conv_ids = [8, 9, 10]
            elif block == 5:
                conv_ids = [11, 12, 13]

            for lid in reversed(conv_ids):
                grad = grad * (c[f'z{lid}'] > 0)  # ReLU

                # Get input to this conv layer
                if lid == 1:
                    input_ = self.input
                elif lid == conv_ids[0]:
                    input_ = c[f'pool{block - 1}'][0] if block > 1 else self.input
                else:
                    input_ = c[f'z{lid - 1}']

                # Compute weight gradient and update
                dW = torch.nn.grad.conv2d_weight(input_, W[f'conv{lid}'].shape, grad, padding=1)
                

                # Propagate grad to previous layer
                grad = F.conv_transpose2d(grad, W[f'conv{lid}'], padding=1)
                with torch.no_grad():
                    W[f'conv{lid}'] -= lr * dW

def train_manual():
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # or standard CIFAR-10 mean/std
        ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset , batch_size=32, shuffle=True)
    subset = Subset(dataset, indices=range(1000))
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    model = ManualVGG16()
    lr=0.01
    
    for epoch in range(20):
            correct = total = 0
            for x, y in loader:
                with torch.no_grad():
                    logits = model.forward(x)
                #print(logits.shape)
                with torch.no_grad():
                    probs = F.softmax(logits, dim=1)
                    probs[range(x.size(0)), y] -= 1   # compute logits, grad
                    probs /= x.size(0)
                with torch.no_grad():
                    model.backward_propogation(probs, lr=lr)

                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

            print(f"Epoch {epoch+1}: Accuracy = {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train_manual()