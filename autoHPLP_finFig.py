import numpy as np
import pywt
import matplotlib.pyplot as plt

# === Load data ===
data = np.loadtxt("Pry-Prot.dat", skiprows=1)
time = data[:, 0]
energy = data[:, 1]

# === Parameters ===
wavelet = 'db4'
autocorr_threshold = 0.3  # threshold for autocorrelation at chosen lag
lag = 1  # lag for autocorrelation
trim_fraction = 0.05  # trim 5% edges

# === Wavelet decomposition ===
max_level = pywt.dwt_max_level(len(energy), pywt.Wavelet(wavelet).dec_len)
coeffs = pywt.wavedec(energy, wavelet, mode='periodization', level=max_level)
print(f"Maximum decomposition level: {max_level}")

# === Compute HP autocorrelation for all levels ===
hp_ac_list = []

def autocorr(x, lag=1):
    x = x - np.mean(x)
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]

for level in range(1, max_level+1):
    # Reconstruct lowpass using approximation + zeros for higher detail levels
    lowpass = pywt.waverec(
        [coeffs[0]] + [np.zeros_like(d) if i >= level else d for i, d in enumerate(coeffs[1:])],
        wavelet, mode='periodization'
    )
    lowpass = lowpass[:len(energy)]
    highpass = energy - lowpass
    ac = autocorr(highpass, lag=lag)
    hp_ac_list.append(ac)
    print(f"Level {level} → HP autocorr (lag={lag}): {ac:.3f}")

# === Select sweet spot level based on autocorr ===
best_level = max_level  # default
for idx, ac in enumerate(hp_ac_list):
    if abs(ac) < autocorr_threshold:
        best_level = max(1, idx)  # choose previous level
        print(f"✅ Sweet spot selected at level {best_level} (HP autocorr below threshold at level {idx+1})")
        break

# === Reconstruct final components using best_level ===
lowpass = pywt.waverec(
    [coeffs[0]] + [np.zeros_like(d) if i >= best_level else d for i, d in enumerate(coeffs[1:])],
    wavelet, mode='periodization'
)
lowpass = lowpass[:len(energy)]
highpass = energy - lowpass

# === Trim 5% edges ===
n_points = len(energy)
cut = int(trim_fraction * n_points)
time_trim = time[cut:-cut]
energy_trim = energy[cut:-cut]
lowpass_trim = lowpass[cut:-cut]
highpass_trim = highpass[cut:-cut]

# === Plot HP autocorr vs decomposition level (excluding last level) ===
levels_to_plot = range(1, max_level)   # exclude final level (e.g., 9th)
hp_ac_to_plot = hp_ac_list[:-1]        # exclude last autocorr value

plt.figure(figsize=(8,5))
plt.plot(levels_to_plot, hp_ac_to_plot, 'o-', label=f'HP autocorr (lag={lag})')
plt.axhline(autocorr_threshold, color='r', linestyle='--', label=f'Threshold = {autocorr_threshold}')
plt.axvline(best_level, color='g', linestyle='--', label=f'Sweet spot = {best_level}')
plt.xlabel('Wavelet Decomposition Level')
plt.ylabel(f'HP Autocorrelation (lag={lag})')
plt.title('High-Pass Autocorrelation VS Decomposition Level')
plt.legend()
plt.tight_layout()
plt.show()

# === Plot original, LP, HP signals ===
plt.figure(figsize=(10,7))

plt.subplot(3,1,1)
plt.plot(time_trim, energy_trim, color='black')
plt.axhline(np.mean(energy_trim), color='gray', linestyle='--', label='Mean energy')
plt.title('Original Energy (trimmed)')
plt.ylabel('Energy')
plt.legend()

plt.subplot(3,1,2)
plt.plot(time_trim, lowpass_trim, color='blue')
plt.title('Low-frequency Component')
plt.ylabel('Energy')

plt.subplot(3,1,3)
plt.plot(time_trim, highpass_trim, color='red')
plt.axhline(np.mean(highpass_trim), color='gray', linestyle='--', label='Mean HP')
plt.title('High-frequency Component (sweet spot)')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()

plt.tight_layout()
plt.show()

# === Save filtered data ===
np.savetxt("data_lowfreq_autocorr.txt", np.column_stack((time_trim, lowpass_trim)),
           fmt="%.6f", delimiter="\t", header="time\tenergy_lowpass")
np.savetxt("data_highfreq_autocorr.txt", np.column_stack((time_trim, highpass_trim)),
           fmt="%.6f", delimiter="\t", header="time\tenergy_highpass")

print("✅ Saved files: data_lowfreq_autocorr.txt and data_highfreq_autocorr.txt")
