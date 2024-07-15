import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from vae.model import Encoder, Decoder, VAE
from vae.utils import preprocess_data


def main():
    # Hyperparameters
    batch_size = 64
    epochs = 30
    input_dim = 28 * 28
    latent_dim = 128
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    train_loader, _ = preprocess_data(batch_size)

    # Initialize models
    encoder = Encoder(input_dim, latent_dim).to(device)
    decoder = Decoder(latent_dim, input_dim).to(device)
    model = VAE(encoder, decoder).to(device)

    # Loss function
    def loss_function(x, x_hat, mean, logvar):
        reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
        regularization_loss = 0.5 * torch.sum(logvar.exp() + mean.pow(2) - logvar - 1)
        return reconstruction_loss + regularization_loss

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print("Start training VAE...")

    @torch.enable_grad()
    def train_epoch(model, train_loader, optimizer, device):
        model.train()
        overall_loss = 0
        for x, _ in tqdm(train_loader, desc="Training", leave=False):
            x = x.view(-1, input_dim).to(device)
            optimizer.zero_grad()
            x_hat, mean, logvar = model(x)
            loss = loss_function(x, x_hat, mean, logvar)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        return overall_loss

    for epoch in range(epochs):
        overall_loss = train_epoch(model, train_loader, optimizer, device)
        print(
            f"Epoch {epoch + 1}/{epochs}, Average Loss: {overall_loss / len(train_loader.dataset):.4f}"
        )

    print("Training finished!")

    # Save model weights
    torch.save(model.state_dict(), "vae_weights.pth")


if __name__ == "__main__":
    # torch.multiprocessing.freeze_support()
    main()
