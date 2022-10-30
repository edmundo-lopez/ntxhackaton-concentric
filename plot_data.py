""" PLOT DATA SAVED TO IDENTIFY INTERESTING FEATURES """

import mne
import numpy as np
from scipy.signal import welch

import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BoardIds


##
# DATA ANALYSIS


def raw_from_data(data):
    """ Get Raw MNE object from data .txt """
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD.value)
    eeg_data = data[eeg_channels, :]
    eeg_data /= 1000000

    # Creating MNE objects from brainflow data arrays
    ch_types = ['eeg'] * len(eeg_channels)
    ch_names = BoardShim.get_eeg_names(BoardIds.MUSE_2_BOARD.value)
    sfreq = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)
    raw.notch_filter(np.arange(50, sfreq/2, 50))

    return raw


ch_names = BoardShim.get_eeg_names(BoardIds.MUSE_2_BOARD.value)
sfreq = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)

data = {"OT1": np.loadtxt('data/eyes_open_task.txt'),
        "OT2": np.loadtxt('data/eyes_open_task2.txt'),
        "OT3": np.loadtxt('data/eyes_open_task3.txt'),
        "OC1": np.loadtxt('data/eyes_open_calm.txt'),
        "OC2": np.loadtxt('data/eyes_open_calm2.txt'),
        "CC": np.loadtxt('data/eyes_closed.txt')}

raw_dict = {}
for task in data:
    raw_dict[task] = raw_from_data(data[task])

# fig, axs = plt.subplots(len(ch_names))

band_dict = {
    "alpha": [8, 12],
    "beta": [20, 30]
}

low_filt = band_dict["beta"][0]
high_filt = band_dict["beta"][1]

all_pxx = np.zeros((len(ch_names), len(raw_dict)))

fig, axs = plt.subplots(len(ch_names))

for i, channel in enumerate(ch_names):
    for j, task in enumerate(raw_dict):
        raw_dict[task].filter(low_filt, high_filt)
        f, pxx = welch(raw_dict[task].get_data()[i], sfreq)
        all_pxx[i, j] = np.mean(pxx)
        axs[i].set_title(channel)
        axs[i].semilogy(f, pxx, label=task)
        axs[i].set_xlim([low_filt, high_filt])

plt.legend()
plt.show()

##
# Remove outlier
all_pxx[3, 1] = np.median(all_pxx[:, 1])

##
plt.figure()
plt.semilogy(all_pxx, linestyle='None', marker='o')
plt.xticks([0, 1, 2, 3], labels=ch_names)
plt.legend(raw_dict.keys())
plt.ylabel('beta power')
plt.show()


##
plt.figure()
mean_mat = all_pxx.mean(axis=0)
std_mat =all_pxx.std(axis=0)
plt.errorbar(raw_dict.keys(), mean_mat, yerr=std_mat, marker='o', capsize=10)
plt.ylabel('beta power')
plt.show()

##
all_pxx_task = all_pxx[:, :3].flatten()
all_pxx_calm = all_pxx[:, 3:].flatten()
plt.figure(figsize=[3, 5])
plt.boxplot([all_pxx_task, all_pxx_calm], labels=["Task", "Calm"])
plt.ylabel("Beta band power")
plt.tight_layout()
plt.show()