import os
import copy
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from torchinfo import summary
from model import LocalizationNN, test_dataloader  
from dataloader import PATH                        
from plot import plot                              
from captum.attr import IntegratedGradients

warnings.filterwarnings("ignore")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(weights_path: str, device: torch.device):
    model = LocalizationNN()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def _to_bchw(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 2:
        return img.unsqueeze(0).unsqueeze(0)
    if img.ndim == 3:
        if img.shape[0] in (1,3):
            return img.unsqueeze(0)
        return img.permute(2,0,1).unsqueeze(0)
    if img.ndim == 4:
        return img
    raise ValueError(f"Unexpected image shape: {tuple(img.shape)}")

def ig_saliency_single(model: nn.Module, image_tensor: torch.Tensor, target_idx: int, n_steps: int = 50) -> np.ndarray:
    img = _to_bchw(image_tensor)
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    img = img.detach().to("cpu").float().requires_grad_(True)
    baseline = torch.zeros_like(img)
    ig = IntegratedGradients(model_cpu)
    with torch.enable_grad():
        attr = ig.attribute(inputs=img, baselines=baseline, target=target_idx,
                            n_steps=n_steps, internal_batch_size=img.size(0))
    a = attr.detach().cpu().numpy().squeeze()
    a = np.abs(a)
    m, M = a.min(), a.max()
    if M <= m + 1e-12:
        return np.zeros_like(a)
    return (a - m) / (M - m)

def _norm_gray(gray):
    g = gray.astype(np.float32)
    g_min, g_max = g.min(), g.max()
    if g_max > g_min + 1e-12:
        g = (g - g_min) / (g_max - g_min)
    else:
        g = np.zeros_like(g)
    return g

def save_saliency_figure(sample_idx: int, image: torch.Tensor, sal0: np.ndarray, sal1: np.ndarray,
                         out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] == 1:
            g = image.squeeze(0).detach().cpu().numpy()
        elif image.ndim == 2:
            g = image.detach().cpu().numpy()
        else:
            g = image.detach().cpu().numpy().squeeze()
    else:
        g = image
    g = _norm_gray(g)

    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(g, cmap="gray")
    ax0.set_title("Input")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(sal0, cmap="magma")
    ax1.set_title("Saliency Map for X")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(sal1, cmap="magma")
    ax2.set_title("Saliency Map for Y")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    out_path = os.path.join(out_dir, f"sample{sample_idx:03d}_figure.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path

def interactive_viewer(image_list, sal0_list, sal1_list):
    n = len(image_list)
    if n == 0:
        print("[viewer] Nothing to show.")
        return

    g = _norm_gray(image_list[0])
    s0 = sal0_list[0]
    s1 = sal1_list[0]

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 3, figure=fig)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_s0  = fig.add_subplot(gs[0, 1])
    ax_s1  = fig.add_subplot(gs[0, 2])

    im_img = ax_img.imshow(g, cmap="gray")
    ax_img.set_title("Input")
    ax_img.axis("off")

    im_s0 = ax_s0.imshow(s0, cmap="magma")
    ax_s0.set_title("Saliency Map for X")
    ax_s0.axis("off")
    fig.colorbar(im_s0, ax=ax_s0, fraction=0.046, pad=0.04)

    im_s1 = ax_s1.imshow(s1, cmap="magma")
    ax_s1.set_title("Saliency Map for Y")
    ax_s1.axis("off")
    fig.colorbar(im_s1, ax=ax_s1, fraction=0.046, pad=0.04)

    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(ax_slider, "index", 0, n-1, valinit=0, valfmt="%0.0f")

    def update(_):
        i = int(slider.val)
        g = _norm_gray(image_list[i])
        im_img.set_data(g)
        im_s0.set_data(sal0_list[i])
        im_s1.set_data(sal1_list[i])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def run_inference_and_saliency(weights, iterations=100, saliency_n=5, save_dir="saliency_out",
                               make_figures=True, interactive=False, show_plot=False):
    device = get_device()
    print(f"[info] Using device: {device}")
    model = load_model(weights, device)

    preds, gts = [], []
    total_loss = torch.zeros(2)

    viewer_images, viewer_sal0, viewer_sal1 = [], [], []

    with torch.no_grad():
        for i in range(iterations):
            image = test_dataloader.dataset[i][0]
            truth = test_dataloader.dataset[i][1]

            img_bchw = _to_bchw(image).to(device).float()
            predict = model(img_bchw).squeeze(0).cpu().numpy()  # [2]
            gt = truth.cpu().numpy()

            total_loss += (truth.cpu() - torch.from_numpy(predict)).pow(2)

            print(f"sample {i:03d}  pred: {predict}   gt: {gt}")
            preds.append(predict)
            gts.append(gt)

            if i < saliency_n:
                s0 = ig_saliency_single(model, image, target_idx=0, n_steps=50)  # X
                s1 = ig_saliency_single(model, image, target_idx=1, n_steps=50)  # Y

                if make_figures:
                    out_fig = save_saliency_figure(i, image, s0, s1, out_dir=save_dir)
                    print(f"[saved] {out_fig}")

                if interactive:
                    if isinstance(image, torch.Tensor):
                        if image.ndim == 3 and image.shape[0] == 1:
                            g = image.squeeze(0).detach().cpu().numpy()
                        elif image.ndim == 2:
                            g = image.detach().cpu().numpy()
                        else:
                            g = image.detach().cpu().numpy().squeeze()
                    else:
                        g = image
                    viewer_images.append(g)
                    viewer_sal0.append(s0)
                    viewer_sal1.append(s1)

    avg_loss = (total_loss / iterations)
    print(f"\nFinal elementwise MSE: x={avg_loss[0].item():.6f}, y={avg_loss[1].item():.6f}")
    print(f"Final total MSE (sum): {(avg_loss.sum()).item():.6f}")

    preds = np.stack(preds); gts = np.stack(gts)
    if show_plot:
        plot(gts[:,0], gts[:,1], preds[:,0], preds[:,1])

    try:
        summary(model, input_size=[8,1,280,640])
    except Exception as e:
        print(f"[warn] summary() failed: {e}")

    if interactive and len(viewer_images) > 0:
        interactive_viewer(viewer_images, viewer_sal0, viewer_sal1)

def main():
    parser = argparse.ArgumentParser(description="Inference with clean saliency visualization (no overlays/text).")
    parser.add_argument("--weights", type=str, default=os.path.join("..", "pt", "localization_epoch15_20250625_182819.pt"))
    parser.add_argument("--iters", type=int, default=10, help="Number of test samples to evaluate.")
    parser.add_argument("--saliency-n", type=int, default=8, help="How many samples to save figures for.")
    parser.add_argument("--no-fig", action="store_true", help="Do not save per-sample figure PNGs.")
    parser.add_argument("--interactive", action="store_true", help="Open an interactive Matplotlib viewer.")
    parser.add_argument("--show-plot", action="store_true", help="Show XY scatter using your plot() util.")
    args = parser.parse_args()

    run_inference_and_saliency(
        weights=args.weights,
        iterations=args.iters,
        saliency_n=args.saliency_n,
        make_figures=not args.no_fig,
        interactive=args.interactive,
        show_plot=args.show_plot,
    )

if __name__ == "__main__":
    main()
