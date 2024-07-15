import torch
from tqdm import tqdm
from vae.model import Encoder, Decoder, VAE
from vae.utils import preprocess_data, show_image
import matplotlib.pyplot as plt


def main():
    # Hyperparameters
    batch_size = 64
    input_dim = 28 * 28
    latent_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Prepare data
    _, test_loader = preprocess_data(batch_size)

    # Initialize models
    encoder = Encoder(input_dim, latent_dim).to(device)
    decoder = Decoder(latent_dim, input_dim).to(device)
    model = VAE(encoder, decoder).to(device)

    # Load model weights
    model.load_state_dict(torch.load("vae_weights.pth"))

    # Evaluate model
    x, x_hat = evaluate_model(model, test_loader, device, input_dim)
    show_image(x, x_hat)

    # Generate image from noise vector
    samples = generate_samples(decoder, latent_dim, batch_size, device)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(samples[32][0], cmap="gray")
    plt.title("Generated Sample 1")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(samples[23][0], cmap="gray")
    plt.title("Generated Sample 2")
    plt.axis("off")
    plt.show()


@torch.no_grad()
def evaluate_model(model, test_loader, device, input_dim):
    model.eval()
    for idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(-1, input_dim).to(device)
        x_hat, _, _ = model(x)
        if idx == 0:
            break

    x = x.view(-1, 1, 28, 28).cpu().numpy()
    x_hat = x_hat.view(-1, 1, 28, 28).cpu().numpy()
    return x, x_hat


@torch.no_grad()
def generate_samples(decoder, latent_dim, batch_size, device):
    noise = torch.randn(batch_size, latent_dim).to(device)
    samples = decoder(noise).view(-1, 1, 28, 28).cpu().numpy()
    return samples


if __name__ == "__main__":
    # torch.multiprocessing.freeze_support()
    main()
