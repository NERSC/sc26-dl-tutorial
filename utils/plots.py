import matplotlib.pyplot as plt


def generate_images(fields):
    _, tar, gen = [x.detach().float().cpu().numpy() for x in fields]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.title("Field 2")
    ax[0].imshow(tar[0, 2, :, :], cmap="turbo")
    ax[0].set_title("Synthetic target")
    ax[1].imshow(gen[0, 2, :, :], cmap="turbo")
    ax[1].set_title("ViT prediction")
    fig.tight_layout()
    return fig
