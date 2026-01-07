import torch
import torch.nn as nn

from taming.models import vqgan


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings : int, embedding_dim : int):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        flat_inputs = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_inputs, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        return quantized


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv_transpose1(x))
        x = self.sigmoid(self.conv_transpose2(x))
        return x


class VQGAN(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(VQGAN, self).__init__()
        self.encoder = Encoder()
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_q = self.vq(z)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        x_recon = self.decoder(z_q)
        return x_recon


import torch
import matplotlib.pyplot as plt


def generate_random_images(model, num_images=4, device='cpu'):
    """
    Generates images by sampling random codebook indices.
    Without a trained Transformer, this produces 'textured noise'.
    """
    model.eval()

    # 1. Define the latent spatial size
    # Your image is 64x64. You have 2 downsampling layers (stride 2).
    # 64 / 2 / 2 = 16. So the latent map is 16x16.
    h, w = 16, 16

    # 2. Generate random indices (integers between 0 and num_embeddings)
    # This simulates "generating" a sequence of tokens
    random_indices = torch.randint(
        low=0,
        high=model.vq._num_embeddings,
        size=(num_images, h, w),
        device=device
    )

    # 3. Lookup Embeddings
    # We grab the vectors corresponding to these random indices
    # Shape becomes: (Batch, H, W, Embedding_Dim)
    z_q = model.vq._embedding(random_indices)

    # 4. Permute for Decoder
    # Decoder expects (Batch, Channel, H, W)
    z_q = z_q.permute(0, 3, 1, 2).contiguous()

    # 5. Decode to Image
    with torch.no_grad():
        generated_images = model.decoder(z_q)

    return generated_images


# --- Helper to visualize ---
def show_images(tensor_images):
    # Move to CPU and permute to (H, W, C) for matplotlib
    images = tensor_images.cpu().permute(0, 2, 3, 1).numpy()
    plt.figure(figsize=(10, 2))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

def check_tensor_stats(name, tensor):
    print(f"--- Checking {name} ---")
    print(f"Shape: {tensor.shape}")
    print(f"Min: {tensor.min().item():.4f}")
    print(f"Max: {tensor.max().item():.4f}")
    print(f"Mean: {tensor.mean().item():.4f}")

    # Check if it's the Gray Square issue (Mean approx 0.5, Low Variance)
    if 0.45 < tensor.mean().item() < 0.55 and tensor.std().item() < 0.1:
        print("DIAGNOSIS: It is Gray because the model is outputting Sigmoid(0). Needs training!")

    # Check if it's a Normalization issue (Negative values)
    elif tensor.min().item() < 0:
        print("DIAGNOSIS: Contains negative values. You must un-normalize (add +1 div 2) before plotting.")
    else:
        print("DIAGNOSIS: Values look okay. Check your plotting code.")
    print("\n")

def main():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from tqdm import trange

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vqgan = VQGAN(num_embeddings=128, embedding_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(vqgan.parameters(), lr=0.001)

    print("Training VQGAN...")
    num_epochs = 25
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_images = vqgan(images)
            loss = criterion(recon_images, images)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    generated_images = generate_random_images(vqgan, num_images=8)
    show_images(generated_images) # 1. RECONSTRUCTION (Checking how well it memorized)
    # Grab a batch of real images
    real_images, _ = next(iter(loader))
    real_images = real_images[:4].to(device)

    check_tensor_stats("Real Images", real_images)
    show_images(real_images)
    print(real_images.shape)

    # Check Reconstructed Images
    with torch.no_grad():
        recon = vqgan(real_images)
    check_tensor_stats("Reconstructed Images", recon)

    print("Reconstructed Real Images:")
    show_images(recon)
    print(recon.shape)

    # 2. GENERATION (Dreaming new images)
    print("Generated Random Images (Texture Synthesis):")
    random_gen = generate_random_images(vqgan, num_images=4, device=device)
    show_images(random_gen)

if __name__ == "__main__":
    main()
