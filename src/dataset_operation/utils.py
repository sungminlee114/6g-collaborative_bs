import numpy as np
import torch

_CFR_FROM_CIR_WARNED = False


def cir_to_cfr(
    cir_a: np.ndarray,
    cir_tau: np.ndarray,
    num_subcarriers: int,
    subcarrier_spacing: float,
) -> np.ndarray:
    """Compute CFR from CIR path coefficients (path-domain DFT).

    H(f_k) = sum_l a_l * exp(-j * 2*pi * f_k * tau_l)

    Args:
        cir_a:   (N_UE, n_rx, n_tx, n_paths) complex64 — path amplitudes
        cir_tau: (N_UE, n_paths) float32 — path delays [s]
        num_subcarriers: number of OFDM subcarriers
        subcarrier_spacing: subcarrier spacing [Hz]

    Returns:
        cfr: (N_UE, n_rx, n_tx, n_sc) complex64
    """
    global _CFR_FROM_CIR_WARNED
    if not _CFR_FROM_CIR_WARNED:
        print(
            "[dataset] CFR not found in channels.npz — computing from CIR on-the-fly.\n"
            "          This adds ~0.3s per snapshot load.  If this is too slow,\n"
            "          re-run data generation with CFR included, or use:\n"
            "            python scripts/strip_cfr.py --restore <data_dir>"
        )
        _CFR_FROM_CIR_WARNED = True

    # Subcarrier frequencies (baseband)
    if num_subcarriers % 2 == 0:
        start = -num_subcarriers // 2
        stop = num_subcarriers // 2
    else:
        start = -(num_subcarriers - 1) // 2
        stop = (num_subcarriers - 1) // 2 + 1
    freqs = np.arange(start, stop, dtype=np.float32) * subcarrier_spacing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freqs_t = torch.tensor(freqs, dtype=torch.float32, device=device)

    # Batched GPU computation — all UEs at once (or in chunks if OOM)
    n_ue = cir_a.shape[0]

    # Expand tau for broadcasting: (N_UE, n_paths) → (N_UE, 1, 1, n_paths)
    tau_expanded = cir_tau
    while tau_expanded.ndim < cir_a.ndim:
        tau_expanded = np.expand_dims(tau_expanded, axis=1)

    # Estimate memory: n_ue * n_rx * n_tx * n_paths * n_sc * 8 bytes (complex64)
    n_paths = cir_a.shape[-1]
    n_sc = len(freqs_t)
    elem_bytes = n_ue * int(np.prod(cir_a.shape[1:])) * n_sc * 8 * 3  # phase + a + exp
    free_vram = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else 8e9
    chunk_size = max(1, int(free_vram * 0.5) // max(elem_bytes // n_ue, 1))
    chunk_size = min(chunk_size, n_ue)

    cfr_chunks = []
    for i in range(0, n_ue, chunk_size):
        j = min(i + chunk_size, n_ue)
        a_t = torch.tensor(cir_a[i:j], dtype=torch.complex64, device=device)
        tau_t = torch.tensor(tau_expanded[i:j], dtype=torch.float32, device=device)
        # phase: (chunk, n_rx, n_tx, n_paths, n_sc)
        phase = -2j * torch.pi * tau_t[..., None] * freqs_t
        cfr_chunks.append(
            torch.sum(a_t[..., None] * torch.exp(phase), dim=-2).cpu().numpy()
        )
        del a_t, tau_t, phase
    return np.concatenate(cfr_chunks, axis=0)  # (N_UE, n_rx, n_tx, n_sc)


def load_cfr_from_npz(
    npz_path, num_subcarriers: int = 1024, subcarrier_spacing: float = 0.0
) -> np.ndarray:
    """Load CFR from channels.npz, computing from CIR if CFR key is missing.

    Args:
        npz_path: path to channels.npz
        num_subcarriers: needed only when CFR must be reconstructed
        subcarrier_spacing: needed only when CFR must be reconstructed

    Returns:
        cfr: (N_UE, n_rx, n_tx, n_sc) complex64
    """
    data = np.load(npz_path)
    if "cfr" in data.files:
        return data["cfr"]

    # Reconstruct from CIR
    assert subcarrier_spacing > 0, (
        f"channels.npz has no 'cfr' key and subcarrier_spacing not provided. "
        f"Available keys: {data.files}"
    )
    return cir_to_cfr(data["cir_a"], data["cir_tau"], num_subcarriers, subcarrier_spacing)


def complex_to_real(h_complex: np.ndarray) -> np.ndarray:
    """Complex channel → real representation: stack real/imag along new axis 0.

    Input: (...) complex
    Output: (2, ...) float
    """
    return np.stack([h_complex.real, h_complex.imag], axis=0).astype(np.float32)


def real_to_complex(h_real: np.ndarray) -> np.ndarray:
    """Real representation → complex channel.

    Input: (2, ...) float  (first dim = real/imag)
    Output: (...) complex
    """
    return h_real[0] + 1j * h_real[1]


def add_awgn(h: np.ndarray, snr_db: float) -> np.ndarray:
    """Add AWGN noise to channel at given SNR (dB).

    Input: (2, n_ant, n_sc) float — real representation
    Output: (2, n_ant, n_sc) float — noisy version (LS estimate)
    """
    signal_power = np.mean(h ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*h.shape).astype(np.float32) * np.sqrt(noise_power)
    return h + noise


def nmse(h_est: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    """Normalized MSE: ||h_est - h_true||^2 / ||h_true||^2.

    Computed per-sample then averaged over batch.
    Input shapes: (batch, 2, n_ant, n_sc)
    """
    err = (h_est - h_true).flatten(1)
    ref = h_true.flatten(1)
    return (err.pow(2).sum(1) / ref.pow(2).sum(1)).mean()


def nmse_db(h_est: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    """NMSE in dB."""
    return 10 * torch.log10(nmse(h_est, h_true))


def prepare_channel_sample(cfr_complex: np.ndarray, snr_db: float) -> dict:
    """Prepare a single channel estimation sample.

    Args:
        cfr_complex: (n_rx_ant, n_tx_ant, n_subcarriers) complex
        snr_db: SNR in dB

    Returns:
        dict with 'input' (noisy LS), 'target' (clean), 'snr_db'
    """
    # Flatten antenna dims: (n_rx_ant * n_tx_ant, n_subcarriers)
    n_rx, n_tx, n_sc = cfr_complex.shape
    h_flat = cfr_complex.reshape(n_rx * n_tx, n_sc)

    # To real: (2, n_ant_pairs, n_sc)
    h_real = complex_to_real(h_flat)

    # Add noise for LS estimate
    h_noisy = add_awgn(h_real, snr_db)

    return {
        'input': h_noisy,     # (2, n_ant_pairs, n_sc) float32
        'target': h_real,     # (2, n_ant_pairs, n_sc) float32
        'snr_db': snr_db,
    }
