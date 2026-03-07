import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm

from .config import (
    BETA_END,
    BETA_START,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    MODEL_BASE_CHANNELS,
    MODEL_TIME_DIM,
    MODEL_VARIANT,
    SCHEDULE_TYPE,
    SAMPLES_DIR,
    TIMESTEPS,
)
from .data import get_mnist_dataloader
from .diffusion.ddpm import sample_ddpm
from .diffusion.forward import q_sample
from .diffusion.posterior import p_mean_from_eps, q_posterior_mean_variance
from .diffusion.schedule import DiffusionSchedule, build_schedule
from .models.classifier import SimpleMNISTClassifier
from .models.unet import TinyUNet
from .train import load_ddpm_checkpoint, train_ddpm
from .utils.seed import set_seed
from .utils.viz import save_image_grid

EVAL_CLASSIFIER_CKPT = CHECKPOINTS_DIR / "mnist_eval_classifier.pt"
RECOMMENDED_TASK7_SAMPLES = 10_000


@torch.no_grad()
def sample_and_save_grid(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    output_path: Path = SAMPLES_DIR / "final_samples.png",
    num_samples: int = 64,
) -> torch.Tensor:
    device = next(model.parameters()).device
    samples, _ = sample_ddpm(
        model=model,
        schedule=schedule,
        num_samples=num_samples,
        device=device,
        capture_steps=None,
    )
    save_image_grid(samples, output_path=output_path, nrow=8, source_range=(-1.0, 1.0))
    return samples


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    num_samples: int,
    batch_size: int = 64,
) -> torch.Tensor:
    device = next(model.parameters()).device
    chunks: list[torch.Tensor] = []
    remaining = num_samples

    while remaining > 0:
        n = min(batch_size, remaining)
        part, _ = sample_ddpm(
            model=model,
            schedule=schedule,
            num_samples=n,
            device=device,
            capture_steps=None,
        )
        chunks.append(part)
        remaining -= n

    return torch.cat(chunks, dim=0)


@torch.no_grad()
def save_denoising_trajectory(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    output_path: Path = SAMPLES_DIR / "trajectory.png",
    seed: int = 2026,
) -> None:
    # Capture i in {L, 3L/4, L/2, L/4, 1} mapped to 0-based indexing.
    one_based = [
        schedule.timesteps,
        max(1, int(round(schedule.timesteps * 0.75))),
        max(1, int(round(schedule.timesteps * 0.50))),
        max(1, int(round(schedule.timesteps * 0.25))),
        1,
    ]
    capture = sorted({i - 1 for i in one_based}, reverse=True)
    _, states = sample_ddpm(
        model=model,
        schedule=schedule,
        num_samples=1,
        device=next(model.parameters()).device,
        capture_steps=capture,
        seed=seed,
    )
    ordered = [states[s] for s in sorted(states.keys(), reverse=True)]
    images = torch.cat(ordered, dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(images, output_path=output_path, nrow=len(ordered), source_range=(-1.0, 1.0))


def save_loss_curve(losses: list[float], output_path: Path = FIGURES_DIR / "task7_loss_curve.png") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title("Task 7: Training Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def get_real_images(batch_size: int = 64, train: bool = False) -> torch.Tensor:
    loader = get_mnist_dataloader(batch_size=batch_size, train=train, shuffle=True)
    images, _ = next(iter(loader))
    return images


def collect_real_images(
    train: bool,
    num_samples: int,
    batch_size: int = 256,
    return_labels: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    loader = get_mnist_dataloader(batch_size=batch_size, train=train, shuffle=True)
    image_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    total = 0

    for images, labels in loader:
        image_parts.append(images)
        if return_labels:
            label_parts.append(labels)
        total += images.shape[0]
        if total >= num_samples:
            break

    images_cat = torch.cat(image_parts, dim=0)[:num_samples]
    if not return_labels:
        return images_cat

    labels_cat = torch.cat(label_parts, dim=0)[:num_samples]
    return images_cat, labels_cat


def compute_simple_metrics(generated: torch.Tensor, real: torch.Tensor) -> dict[str, float]:
    generated = generated.detach().cpu()
    real = real.detach().cpu()

    g_flat = generated.view(generated.shape[0], -1)
    r_flat = real.view(real.shape[0], -1)

    mean_gap = float(torch.abs(generated.mean() - real.mean()).item())
    std_gap = float(torch.abs(generated.std() - real.std()).item())
    nn_l2 = float(torch.cdist(g_flat, r_flat, p=2).min(dim=1).values.mean().item())
    diversity = float(torch.pdist(g_flat, p=2).mean().item()) if g_flat.shape[0] > 1 else 0.0

    return {
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "nearest_neighbor_l2": nn_l2,
        "pairwise_diversity_l2": diversity,
    }


def save_nearest_neighbor_grid(
    generated: torch.Tensor,
    real: torch.Tensor,
    output_path: Path = FIGURES_DIR / "task7_nearest_neighbors.png",
    num_pairs: int = 8,
) -> Path:
    generated = generated.detach().cpu()[:num_pairs]
    real = real.detach().cpu()

    g_flat = generated.view(generated.shape[0], -1)
    r_flat = real.view(real.shape[0], -1)
    nn_index = torch.cdist(g_flat, r_flat, p=2).argmin(dim=1)

    pairs = []
    for i in range(generated.shape[0]):
        pairs.append(generated[i : i + 1])
        pairs.append(real[nn_index[i] : nn_index[i] + 1])

    images = torch.cat(pairs, dim=0)
    save_image_grid(images, output_path=output_path, nrow=2, source_range=(-1.0, 1.0))
    return output_path


def train_mnist_classifier(
    epochs: int = 2,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> SimpleMNISTClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMNISTClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loader = get_mnist_dataloader(batch_size=batch_size, train=True, shuffle=True)

    model.train()
    for epoch in range(epochs):
        progress = tqdm(loader, desc=f"Classifier Epoch {epoch + 1}/{epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def load_or_train_mnist_classifier(
    checkpoint_path: Path = EVAL_CLASSIFIER_CKPT,
    force_retrain: bool = False,
    epochs: int = 2,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> tuple[SimpleMNISTClassifier, str, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMNISTClassifier().to(device)

    if checkpoint_path.exists() and not force_retrain:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model, str(checkpoint_path), False

    model = train_mnist_classifier(epochs=epochs, batch_size=batch_size, lr=lr)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    return model, str(checkpoint_path), True


@torch.no_grad()
def classifier_accuracy(
    classifier: SimpleMNISTClassifier,
    train: bool = False,
    batch_size: int = 256,
    max_batches: int = 20,
) -> float:
    device = next(classifier.parameters()).device
    loader = get_mnist_dataloader(batch_size=batch_size, train=train, shuffle=False)

    classifier.eval()
    total_correct = 0
    total_seen = 0

    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        logits = classifier(images)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_seen += labels.shape[0]

    if total_seen == 0:
        return 0.0
    return float(total_correct / total_seen)


@torch.no_grad()
def extract_features_and_logits(
    classifier: SimpleMNISTClassifier,
    images: torch.Tensor,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(classifier.parameters()).device
    classifier.eval()

    feats: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []

    n = images.shape[0]
    for start in range(0, n, batch_size):
        batch = images[start : start + batch_size].to(device)
        logits, features = classifier(batch, return_features=True)
        feats.append(features.detach().cpu())
        logits_list.append(logits.detach().cpu())

    return torch.cat(feats, dim=0), torch.cat(logits_list, dim=0)


def _feature_mean_cov(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = features.to(torch.float64)
    mean = x.mean(dim=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    return mean, cov


def _matrix_sqrt_spd(matrix: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    matrix = (matrix + matrix.T) * 0.5
    vals, vecs = torch.linalg.eigh(matrix)
    vals = torch.clamp(vals, min=eps)
    sqrt_vals = torch.sqrt(vals)
    return (vecs * sqrt_vals.unsqueeze(0)) @ vecs.T


def compute_dataset_fid(features_gen: torch.Tensor, features_real: torch.Tensor) -> float:
    mu_g, cov_g = _feature_mean_cov(features_gen)
    mu_r, cov_r = _feature_mean_cov(features_real)

    diff = mu_g - mu_r
    cov_g_sqrt = _matrix_sqrt_spd(cov_g)
    middle = cov_g_sqrt @ cov_r @ cov_g_sqrt
    cov_sqrt = _matrix_sqrt_spd(middle)

    fid = diff.dot(diff) + torch.trace(cov_g + cov_r - 2.0 * cov_sqrt)
    return float(fid.clamp(min=0).item())


def compute_dataset_kid(
    features_gen: torch.Tensor,
    features_real: torch.Tensor,
    subset_size: int = 100,
    num_subsets: int = 20,
) -> float:
    x = features_gen.to(torch.float64)
    y = features_real.to(torch.float64)
    dim = x.shape[1]
    m = min(subset_size, x.shape[0], y.shape[0])
    if m < 2:
        return 0.0

    def poly_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ((a @ b.T) / dim + 1.0) ** 3

    values = []
    for _ in range(num_subsets):
        idx_x = torch.randperm(x.shape[0])[:m]
        idx_y = torch.randperm(y.shape[0])[:m]
        xs = x[idx_x]
        ys = y[idx_y]

        k_xx = poly_kernel(xs, xs)
        k_yy = poly_kernel(ys, ys)
        k_xy = poly_kernel(xs, ys)

        m_float = float(m)
        term_xx = (k_xx.sum() - torch.diag(k_xx).sum()) / (m_float * (m_float - 1.0))
        term_yy = (k_yy.sum() - torch.diag(k_yy).sum()) / (m_float * (m_float - 1.0))
        term_xy = k_xy.mean()
        kid = term_xx + term_yy - 2.0 * term_xy
        values.append(kid)

    return float(torch.stack(values).mean().item())


@torch.no_grad()
def nearest_feature_labels(
    query_features: torch.Tensor,
    ref_features: torch.Tensor,
    ref_labels: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    query = query_features.to(torch.float32)
    ref = ref_features.to(torch.float32)
    labels = ref_labels.to(torch.long).view(-1)

    if ref.shape[0] != labels.shape[0]:
        raise ValueError("ref_features and ref_labels must have matching first dimension.")

    out_labels: list[torch.Tensor] = []
    for start in range(0, query.shape[0], batch_size):
        q = query[start : start + batch_size]
        dists = torch.cdist(q, ref, p=2)
        nn_idx = dists.argmin(dim=1)
        out_labels.append(labels[nn_idx])
    return torch.cat(out_labels, dim=0)


def compute_classifier_generation_metrics(
    logits_gen: torch.Tensor,
    proxy_labels_gen: torch.Tensor | None = None,
) -> dict[str, float]:
    probs = torch.softmax(logits_gen, dim=1)
    max_prob, pred = probs.max(dim=1)

    hist = torch.bincount(pred, minlength=10).float()
    hist = hist / max(hist.sum(), 1.0)
    entropy = -(hist * (hist + 1e-12).log()).sum()
    entropy_norm = entropy / math.log(10.0)

    metrics = {
        "generated_mean_confidence": float(max_prob.mean().item()),
        "generated_classifiable_fraction_0_5": float((max_prob >= 0.5).float().mean().item()),
        "generated_accuracy_proxy_0_5": float((max_prob >= 0.5).float().mean().item()),
        "generated_class_entropy": float(entropy.item()),
        "generated_class_entropy_normalized": float(entropy_norm.item()),
    }
    if proxy_labels_gen is not None:
        labels = proxy_labels_gen.to(torch.long).view(-1)
        if labels.shape[0] == pred.shape[0]:
            metrics["generated_nn_label_accuracy_proxy"] = float((pred == labels).float().mean().item())
    return metrics


def run_task7_diagnostics(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    losses: list[float],
    num_samples: int = 64,
    trajectory_seed: int = 2026,
) -> dict[str, str]:
    loss_path = save_loss_curve(losses, FIGURES_DIR / "task7_loss_curve.png")
    sample_path = SAMPLES_DIR / "task7_samples.png"
    traj_path = SAMPLES_DIR / "task7_trajectory.png"

    sample_and_save_grid(
        model=model,
        schedule=schedule,
        output_path=sample_path,
        num_samples=num_samples,
    )
    save_denoising_trajectory(
        model=model,
        schedule=schedule,
        output_path=traj_path,
        seed=trajectory_seed,
    )

    return {
        "loss_curve": str(loss_path),
        "sample_grid": str(sample_path),
        "trajectory": str(traj_path),
    }


@torch.no_grad()
def estimate_elbo_bpd(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    num_batches: int = 5,
    batch_size: int = 64,
    timestep_stride: int = 50,
    sigma_obs2: float = 1e-2,
) -> dict[str, float]:
    device = next(model.parameters()).device
    loader = get_mnist_dataloader(batch_size=batch_size, train=False, shuffle=True)
    model.eval()

    t_values = torch.arange(1, schedule.timesteps, timestep_stride, device=device, dtype=torch.long)
    if t_values.numel() == 0:
        t_values = torch.tensor([1], device=device, dtype=torch.long)
    diffusion_scale = (schedule.timesteps - 1) / float(t_values.numel())

    all_elbo = []
    all_bpd = []
    D = 28 * 28

    for i, (x0, _) in enumerate(loader):
        if i >= num_batches:
            break
        x0 = x0.to(device)
        b = x0.shape[0]

        alpha_bar_T = schedule.alpha_bars[-1]
        mu_T = torch.sqrt(alpha_bar_T) * x0
        var_T = (1.0 - alpha_bar_T).clamp(min=1e-12)
        lprior = 0.5 * (var_T + mu_T.pow(2) - 1.0 - torch.log(var_T))
        lprior = lprior.view(b, -1).sum(dim=1)

        t0 = torch.zeros(b, device=device, dtype=torch.long)
        x1 = q_sample(x0=x0, t=t0, schedule=schedule)
        eps0 = model(x1, t0)
        p_mean0, _, _ = p_mean_from_eps(x_t=x1, t=t0, eps_pred=eps0, schedule=schedule)
        recon = 0.5 * (((x0 - p_mean0) ** 2) / sigma_obs2 + math.log(2.0 * math.pi * sigma_obs2))
        lrecon = recon.view(b, -1).sum(dim=1)

        ldiff = torch.zeros(b, device=device)
        for t_scalar in t_values:
            t = torch.full((b,), int(t_scalar.item()), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = q_sample(x0=x0, t=t, schedule=schedule, noise=noise)
            q_mean, q_var = q_posterior_mean_variance(x0=x0, x_t=xt, t=t, schedule=schedule)
            eps_pred = model(xt, t)
            p_mean, p_var, _ = p_mean_from_eps(x_t=xt, t=t, eps_pred=eps_pred, schedule=schedule)

            q_var = q_var.clamp(min=1e-12)
            p_var = p_var.clamp(min=1e-12)
            kl = 0.5 * (
                torch.log(p_var) - torch.log(q_var) + (q_var + (q_mean - p_mean) ** 2) / p_var - 1.0
            )
            kl = kl.view(b, -1).sum(dim=1)
            ldiff += kl

        ldiff = ldiff * diffusion_scale
        elbo = lprior + lrecon + ldiff
        bpd = elbo / (D * math.log(2.0))
        all_elbo.append(elbo.detach().cpu())
        all_bpd.append(bpd.detach().cpu())

    elbo_cat = torch.cat(all_elbo, dim=0)
    bpd_cat = torch.cat(all_bpd, dim=0)
    return {
        "elbo_mean": float(elbo_cat.mean().item()),
        "elbo_std": float(elbo_cat.std().item()),
        "bpd_mean": float(bpd_cat.mean().item()),
        "bpd_std": float(bpd_cat.std().item()),
        "elbo_num_batches": float(num_batches),
        "elbo_timestep_stride": float(timestep_stride),
    }


def _task7_sample_count_note(num_samples: int, recommended: int = RECOMMENDED_TASK7_SAMPLES) -> str:
    if num_samples >= recommended:
        return f"num_samples={num_samples} (meets recommended >= {recommended})"
    return f"num_samples={num_samples} (below recommended >= {recommended}; prefer KID over FID)"


def run_task7_full_evaluation(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    losses: list[float],
    num_samples: int = RECOMMENDED_TASK7_SAMPLES,
    metrics_real_samples: int = RECOMMENDED_TASK7_SAMPLES,
    nn_train_samples: int = 60_000,
    classifier_epochs: int = 2,
    classifier_checkpoint_path: Path = EVAL_CLASSIFIER_CKPT,
    force_retrain_classifier: bool = False,
    trajectory_seed: int = 2026,
    elbo_num_batches: int = 5,
    elbo_timestep_stride: int = 50,
) -> dict[str, float | str]:
    artifacts = run_task7_diagnostics(
        model=model,
        schedule=schedule,
        losses=losses,
        num_samples=64,
        trajectory_seed=trajectory_seed,
    )

    classifier, clf_ckpt, clf_trained_now = load_or_train_mnist_classifier(
        checkpoint_path=classifier_checkpoint_path,
        force_retrain=force_retrain_classifier,
        epochs=classifier_epochs,
    )
    classifier_test_acc = classifier_accuracy(classifier, train=False)

    real_train_pair = collect_real_images(
        train=True,
        num_samples=metrics_real_samples,
        return_labels=True,
    )
    real_train, real_train_labels = real_train_pair  # type: ignore[misc]
    real_test = collect_real_images(train=False, num_samples=metrics_real_samples)
    real_train_nn = collect_real_images(train=True, num_samples=nn_train_samples)
    generated = generate_samples(model, schedule, num_samples=num_samples)

    gen_features, gen_logits = extract_features_and_logits(classifier, generated)
    train_features, _ = extract_features_and_logits(classifier, real_train)
    test_features, _ = extract_features_and_logits(classifier, real_test)
    pseudo_labels = nearest_feature_labels(gen_features, train_features, real_train_labels)

    fid_test = compute_dataset_fid(gen_features, test_features)
    fid_train = compute_dataset_fid(gen_features, train_features)
    kid_test = compute_dataset_kid(gen_features, test_features)
    kid_train = compute_dataset_kid(gen_features, train_features)

    nn_path = save_nearest_neighbor_grid(
        generated,
        real_train_nn,
        output_path=FIGURES_DIR / "task7_nearest_neighbors.png",
        num_pairs=8,
    )

    gen_metrics = compute_classifier_generation_metrics(gen_logits, proxy_labels_gen=pseudo_labels)
    simple_metrics = compute_simple_metrics(generated, real_test)
    elbo_stats = estimate_elbo_bpd(
        model=model,
        schedule=schedule,
        num_batches=elbo_num_batches,
        timestep_stride=elbo_timestep_stride,
    )

    results: dict[str, float | str] = {
        **artifacts,
        "nearest_neighbor_grid": str(nn_path),
        "classifier_test_accuracy": float(classifier_test_acc),
        "classifier_feature_extractor_checkpoint": clf_ckpt,
        "classifier_feature_extractor_retrained_now": float(int(clf_trained_now)),
        "feature_fid_test": float(fid_test),
        "feature_fid_train": float(fid_train),
        "feature_kid_test": float(kid_test),
        "feature_kid_train": float(kid_train),
        "feature_fid_train_minus_test": float(fid_train - fid_test),
        "feature_kid_train_minus_test": float(kid_train - kid_test),
        "eval_num_samples": float(num_samples),
        "real_reference_samples": float(metrics_real_samples),
        "nearest_neighbor_train_pool": float(real_train_nn.shape[0]),
        "task7_metric_sample_note": _task7_sample_count_note(num_samples),
        **gen_metrics,
        **simple_metrics,
        **elbo_stats,
    }
    return results


def run_task6_ablation(
    train_steps: int = 500,
    timesteps: int = TIMESTEPS,
    num_samples: int = 512,
    metrics_real_samples: int = 2_000,
    nn_train_samples: int = 20_000,
    classifier_epochs: int = 2,
    classifier_checkpoint_path: Path = EVAL_CLASSIFIER_CKPT,
    force_retrain_classifier: bool = False,
    unet_variant: str = MODEL_VARIANT,
    unet_time_dim: int = MODEL_TIME_DIM,
    unet_base_channels: int = MODEL_BASE_CHANNELS,
    elbo_num_batches: int = 3,
    elbo_timestep_stride: int = 100,
) -> list[dict[str, object]]:
    """
    Task 6 schedule ablation:
    compare linear beta schedule to a cosine alternative with everything else fixed.
    """
    variants = [
        {"name": "linear", "schedule_type": SCHEDULE_TYPE, "beta_end": BETA_END},
        {"name": "cosine", "schedule_type": "cosine", "beta_end": BETA_END},
    ]

    classifier, clf_ckpt, clf_trained_now = load_or_train_mnist_classifier(
        checkpoint_path=classifier_checkpoint_path,
        force_retrain=force_retrain_classifier,
        epochs=classifier_epochs,
    )
    classifier_test_acc = classifier_accuracy(classifier, train=False)

    real_train_pair = collect_real_images(
        train=True,
        num_samples=metrics_real_samples,
        return_labels=True,
    )
    real_train, real_train_labels = real_train_pair  # type: ignore[misc]
    real_test = collect_real_images(train=False, num_samples=metrics_real_samples)
    real_train_nn = collect_real_images(train=True, num_samples=nn_train_samples)
    real_train_features, _ = extract_features_and_logits(classifier, real_train)
    real_test_features, _ = extract_features_and_logits(classifier, real_test)

    results: list[dict[str, object]] = []
    for variant in variants:
        tag = variant["name"]
        schedule_type = str(variant["schedule_type"])
        beta_end = float(variant["beta_end"])

        model, schedule, losses = train_ddpm(
            steps=train_steps,
            timesteps=timesteps,
            unet_variant=unet_variant,
            unet_time_dim=unet_time_dim,
            unet_base_channels=unet_base_channels,
            schedule_type=schedule_type,
            beta_start=BETA_START,
            beta_end=beta_end,
            sample_every=0,
            checkpoint_path=CHECKPOINTS_DIR / f"task6_{tag}.pt",
        )

        generated = generate_samples(model, schedule, num_samples=num_samples)
        sample_path = SAMPLES_DIR / f"task6_{tag}_samples.png"
        save_image_grid(generated, output_path=sample_path, nrow=8, source_range=(-1.0, 1.0))

        gen_features, gen_logits = extract_features_and_logits(classifier, generated)
        pseudo_labels = nearest_feature_labels(gen_features, real_train_features, real_train_labels)
        fid_test = compute_dataset_fid(gen_features, real_test_features)
        fid_train = compute_dataset_fid(gen_features, real_train_features)
        kid_test = compute_dataset_kid(gen_features, real_test_features)
        kid_train = compute_dataset_kid(gen_features, real_train_features)

        nn_path = save_nearest_neighbor_grid(
            generated,
            real_train_nn,
            output_path=FIGURES_DIR / f"task6_{tag}_nearest_neighbors.png",
            num_pairs=8,
        )

        gen_metrics = compute_classifier_generation_metrics(gen_logits, proxy_labels_gen=pseudo_labels)
        simple_metrics = compute_simple_metrics(generated, real_test)
        elbo_stats = estimate_elbo_bpd(
            model=model,
            schedule=schedule,
            num_batches=elbo_num_batches,
            timestep_stride=elbo_timestep_stride,
        )

        results.append(
            {
                "name": tag,
                "schedule_type": schedule_type,
                "unet_variant": unet_variant,
                "unet_time_dim": float(unet_time_dim),
                "unet_base_channels": float(unet_base_channels),
                "beta_end": beta_end,
                "final_loss": float(losses[-1]),
                "sample_grid": str(sample_path),
                "nearest_neighbor_grid": str(nn_path),
                "classifier_test_accuracy": float(classifier_test_acc),
                "classifier_feature_extractor_checkpoint": clf_ckpt,
                "classifier_feature_extractor_retrained_now": float(int(clf_trained_now)),
                "feature_fid_test": float(fid_test),
                "feature_fid_train": float(fid_train),
                "feature_kid_test": float(kid_test),
                "feature_kid_train": float(kid_train),
                "feature_fid_train_minus_test": float(fid_train - fid_test),
                "feature_kid_train_minus_test": float(kid_train - kid_test),
                "eval_num_samples": float(num_samples),
                "real_reference_samples": float(metrics_real_samples),
                "nearest_neighbor_train_pool": float(real_train_nn.shape[0]),
                "loss_history": losses,
                **gen_metrics,
                **simple_metrics,
                **elbo_stats,
            }
        )

    return results


def run_task6_timestep_ablation(
    timestep_values: tuple[int, ...] = (1000, 500),
    seeds: tuple[int, ...] = (2026,),
    train_steps: int = 250,
    num_samples: int = 128,
    metrics_real_samples: int = 1_024,
    nn_train_samples: int = 5_000,
    classifier_epochs: int = 1,
    classifier_checkpoint_path: Path = EVAL_CLASSIFIER_CKPT,
    force_retrain_classifier: bool = False,
    unet_variant: str = MODEL_VARIANT,
    unet_time_dim: int = MODEL_TIME_DIM,
    unet_base_channels: int = MODEL_BASE_CHANNELS,
    elbo_num_batches: int = 2,
    elbo_timestep_stride: int = 100,
) -> list[dict[str, object]]:
    """
    Task 6 timestep-count ablation:
    compare models trained with different diffusion lengths L (same architecture/hparams).
    Multiple seeds can be used to reduce single-run noise.
    """
    classifier, clf_ckpt, clf_trained_now = load_or_train_mnist_classifier(
        checkpoint_path=classifier_checkpoint_path,
        force_retrain=force_retrain_classifier,
        epochs=classifier_epochs,
    )
    classifier_test_acc = classifier_accuracy(classifier, train=False)

    real_train_pair = collect_real_images(
        train=True,
        num_samples=metrics_real_samples,
        return_labels=True,
    )
    real_train, real_train_labels = real_train_pair  # type: ignore[misc]
    real_test = collect_real_images(train=False, num_samples=metrics_real_samples)
    real_train_nn = collect_real_images(train=True, num_samples=nn_train_samples)
    real_train_features, _ = extract_features_and_logits(classifier, real_train)
    real_test_features, _ = extract_features_and_logits(classifier, real_test)

    results: list[dict[str, object]] = []
    for seed in seeds:
        set_seed(int(seed))
        for t_steps in timestep_values:
            t_steps_int = int(t_steps)
            tag = f"t{t_steps_int}"
            run_tag = f"{tag}_seed{int(seed)}"

            model, schedule, losses = train_ddpm(
                steps=train_steps,
                timesteps=t_steps_int,
                unet_variant=unet_variant,
                unet_time_dim=unet_time_dim,
                unet_base_channels=unet_base_channels,
                schedule_type=SCHEDULE_TYPE,
                beta_start=BETA_START,
                beta_end=BETA_END,
                sample_every=0,
                checkpoint_path=CHECKPOINTS_DIR / f"task6_timestep_{run_tag}.pt",
            )

            generated = generate_samples(model, schedule, num_samples=num_samples)
            sample_path = SAMPLES_DIR / f"task6_timestep_{run_tag}_samples.png"
            save_image_grid(generated, output_path=sample_path, nrow=8, source_range=(-1.0, 1.0))

            gen_features, gen_logits = extract_features_and_logits(classifier, generated)
            pseudo_labels = nearest_feature_labels(gen_features, real_train_features, real_train_labels)
            fid_test = compute_dataset_fid(gen_features, real_test_features)
            fid_train = compute_dataset_fid(gen_features, real_train_features)
            kid_test = compute_dataset_kid(gen_features, real_test_features)
            kid_train = compute_dataset_kid(gen_features, real_train_features)

            nn_path = save_nearest_neighbor_grid(
                generated,
                real_train_nn,
                output_path=FIGURES_DIR / f"task6_timestep_{run_tag}_nearest_neighbors.png",
                num_pairs=8,
            )

            gen_metrics = compute_classifier_generation_metrics(gen_logits, proxy_labels_gen=pseudo_labels)
            simple_metrics = compute_simple_metrics(generated, real_test)
            elbo_stats = estimate_elbo_bpd(
                model=model,
                schedule=schedule,
                num_batches=elbo_num_batches,
                timestep_stride=elbo_timestep_stride,
            )

            results.append(
                {
                    "name": tag,
                    "run_name": run_tag,
                    "run_seed": float(seed),
                    "timesteps": float(t_steps_int),
                    "schedule_type": SCHEDULE_TYPE,
                    "unet_variant": unet_variant,
                    "unet_time_dim": float(unet_time_dim),
                    "unet_base_channels": float(unet_base_channels),
                    "final_loss": float(losses[-1]),
                    "sample_grid": str(sample_path),
                    "nearest_neighbor_grid": str(nn_path),
                    "classifier_test_accuracy": float(classifier_test_acc),
                    "classifier_feature_extractor_checkpoint": clf_ckpt,
                    "classifier_feature_extractor_retrained_now": float(int(clf_trained_now)),
                    "feature_fid_test": float(fid_test),
                    "feature_fid_train": float(fid_train),
                    "feature_kid_test": float(kid_test),
                    "feature_kid_train": float(kid_train),
                    "feature_fid_train_minus_test": float(fid_train - fid_test),
                    "feature_kid_train_minus_test": float(kid_train - kid_test),
                    "eval_num_samples": float(num_samples),
                    "real_reference_samples": float(metrics_real_samples),
                    "nearest_neighbor_train_pool": float(real_train_nn.shape[0]),
                    "loss_history": losses,
                    **gen_metrics,
                    **simple_metrics,
                    **elbo_stats,
                }
            )

    return results


def run_task6_diagnostics(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    losses: list[float],
    num_samples: int = 64,
    trajectory_seed: int = 2026,
) -> dict[str, str]:
    """
    Backward-compatible alias. PDF Task 7 owns diagnostics + evaluation.
    """
    return run_task7_diagnostics(
        model=model,
        schedule=schedule,
        losses=losses,
        num_samples=num_samples,
        trajectory_seed=trajectory_seed,
    )


def run_task6_full_evaluation(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    losses: list[float],
    num_samples: int = RECOMMENDED_TASK7_SAMPLES,
    metrics_real_samples: int = RECOMMENDED_TASK7_SAMPLES,
    nn_train_samples: int = 60_000,
    classifier_epochs: int = 2,
    classifier_checkpoint_path: Path = EVAL_CLASSIFIER_CKPT,
    force_retrain_classifier: bool = False,
    trajectory_seed: int = 2026,
    elbo_num_batches: int = 5,
    elbo_timestep_stride: int = 50,
) -> dict[str, float | str]:
    """
    Backward-compatible alias. Kept to avoid breaking existing notebooks/scripts.
    """
    return run_task7_full_evaluation(
        model=model,
        schedule=schedule,
        losses=losses,
        num_samples=num_samples,
        metrics_real_samples=metrics_real_samples,
        nn_train_samples=nn_train_samples,
        classifier_epochs=classifier_epochs,
        classifier_checkpoint_path=classifier_checkpoint_path,
        force_retrain_classifier=force_retrain_classifier,
        trajectory_seed=trajectory_seed,
        elbo_num_batches=elbo_num_batches,
        elbo_timestep_stride=elbo_timestep_stride,
    )


def run_task7_ablation(
    train_steps: int = 500,
    timesteps: int = TIMESTEPS,
    num_samples: int = 512,
    metrics_real_samples: int = 2_000,
    nn_train_samples: int = 20_000,
    classifier_epochs: int = 2,
    classifier_checkpoint_path: Path = EVAL_CLASSIFIER_CKPT,
    force_retrain_classifier: bool = False,
    elbo_num_batches: int = 3,
    elbo_timestep_stride: int = 100,
) -> list[dict[str, float | str]]:
    """
    Backward-compatible alias. PDF Task 6 is ablation.
    """
    return run_task6_ablation(
        train_steps=train_steps,
        timesteps=timesteps,
        num_samples=num_samples,
        metrics_real_samples=metrics_real_samples,
        nn_train_samples=nn_train_samples,
        classifier_epochs=classifier_epochs,
        classifier_checkpoint_path=classifier_checkpoint_path,
        force_retrain_classifier=force_retrain_classifier,
        elbo_num_batches=elbo_num_batches,
        elbo_timestep_stride=elbo_timestep_stride,
    )


def sliced_wasserstein_distance_2d(
    real: torch.Tensor,
    generated: torch.Tensor,
    num_projections: int = 64,
) -> float:
    """
    Task-C utility for toy 2D data.
    """
    x = real.to(torch.float64)
    y = generated.to(torch.float64)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]

    dirs = torch.randn(num_projections, 2, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)

    proj_x = x @ dirs.T
    proj_y = y @ dirs.T
    proj_x, _ = torch.sort(proj_x, dim=0)
    proj_y, _ = torch.sort(proj_y, dim=0)

    return float(torch.mean(torch.abs(proj_x - proj_y)).item())


def rbf_mmd_2d(
    real: torch.Tensor,
    generated: torch.Tensor,
    sigma: float = 1.0,
) -> float:
    """
    Task-C utility for toy 2D data.
    """
    x = real.to(torch.float64)
    y = generated.to(torch.float64)

    def rbf_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dist2 = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dist2 / (2.0 * sigma * sigma))

    k_xx = rbf_kernel(x, x)
    k_yy = rbf_kernel(y, y)
    k_xy = rbf_kernel(x, y)
    mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(mmd.item())


def save_toy_scatter(
    real: torch.Tensor,
    generated: torch.Tensor,
    output_path: Path = FIGURES_DIR / "task6_toy_scatter.png",
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = real.detach().cpu().numpy()
    y = generated.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=8, alpha=0.5, label="real")
    plt.scatter(y[:, 0], y[:, 1], s=8, alpha=0.5, label="generated")
    plt.legend()
    plt.title("Toy Data: Real vs Generated")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def load_trained_model(
    checkpoint_path: Path = CHECKPOINTS_DIR / "tiny_unet_mnist.pt",
) -> tuple[TinyUNet, DiffusionSchedule]:
    model, schedule = load_ddpm_checkpoint(checkpoint_path=checkpoint_path)
    return model, schedule
