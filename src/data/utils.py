import numpy as np
import torch


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
