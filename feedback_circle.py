""" VISUAL NEUROFEEDBACK FOR MEDITATION """

import time
import mne
import numpy as np
from scipy.signal import welch

import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# RECORD SIGNAL

# Total recording time
total_time = 20
# Filter
low_filt = 20
high_filt = 30

# Seconds for window
window_secs = 4

# Board parameters
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
board = BoardShim(38, params)

eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD.value)

# Creating MNE objects from brainflow data arrays
ch_types = ['eeg'] * len(eeg_channels)
ch_names = BoardShim.get_eeg_names(BoardIds.MUSE_2_BOARD.value)
sfreq = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Calibration session
cal_time = 5

board.prepare_session()
board.start_stream()
time.sleep(cal_time)
data_cal = board.get_board_data()
board.stop_stream()
board.release_session()
print("Calibration ended!")
eeg_data = data_cal[eeg_channels, :]
eeg_data /= 1000000  # BrainFlow returns uV, convert to V for MNE
raw = mne.io.RawArray(eeg_data, info)
raw.filter(low_filt, high_filt)
f, pxx = welch(raw.get_data(), sfreq)
cal_pxx = np.mean(pxx) * 2
print("Mean power: ", cal_pxx)

# Recording session
board.prepare_session()
board.start_stream()

fig = plt.figure(figsize=[10, 10])
fig_board = plt.axes(xlim=(-1, 1), ylim=(-1, 1))

x = 0
y = 0

time.sleep(2)
for i in range(total_time):
    time.sleep(1)
    data_live = board.get_current_board_data(sfreq)

    eeg_data = data_live[eeg_channels, :]
    eeg_data /= 1000000  # BrainFlow returns uV, convert to V for MNE
    raw = mne.io.RawArray(eeg_data, info)
    raw.filter(low_filt, high_filt)
    f, pxx = welch(raw.get_data(), sfreq)

    pxx_norm = np.mean(pxx)/cal_pxx
    circle = plt.Circle((x, y), pxx_norm, fc='pink')
    fig_board.add_patch(circle)
    plt.axis('off')
    plt.draw()
    plt.pause(1)
    circle.remove()

plt.show()

board.stop_stream()
board.release_session()