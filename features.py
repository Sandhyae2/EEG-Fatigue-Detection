import numpy as np
import mne
from mne.preprocessing import ICA

def extract_features(file_path, data_format='int16'):

    raw = mne.io.read_raw_cnt(
        file_path,
        preload=False,
        data_format=data_format,
        recompute_n_samples=True,
        verbose=False
    )

    raw.load_data()

    filtered = raw.copy().filter(
        l_freq=1,
        h_freq=40,
        method='fir'
    )

    filtered.notch_filter(freqs=50)

    resampled = filtered.copy().resample(250)

    ica = ICA(
        n_components=15,
        random_state=42,
        max_iter='auto'
    )

    ica.fit(resampled)

    cleaned = resampled.copy()

    ica.exclude = []

    ica.apply(cleaned)

    epochs = mne.make_fixed_length_epochs(
        cleaned,
        duration=2,
        overlap=0,
        preload=True
    )

    psd = epochs.compute_psd(
        fmin=1,
        fmax=40,
        method='multitaper'
    )

    psd_data = psd.get_data()

    freqs = psd.freqs

    theta_idx = np.where((freqs >= 4) & (freqs < 8))[0]

    alpha_idx = np.where((freqs >= 8) & (freqs < 13))[0]

    beta_idx = np.where((freqs >= 13) & (freqs < 30))[0]

    theta_power = psd_data[:, :, theta_idx].mean(axis=2)

    alpha_power = psd_data[:, :, alpha_idx].mean(axis=2)

    beta_power = psd_data[:, :, beta_idx].mean(axis=2)

    theta_alpha_ratio = (
        theta_power /
        (alpha_power + 1e-10)
    )

    features = np.concatenate([
        theta_power,
        alpha_power,
        beta_power,
        theta_alpha_ratio
    ], axis=1)
    
    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)

    features = np.concatenate([mean_feat, std_feat])
    return features


