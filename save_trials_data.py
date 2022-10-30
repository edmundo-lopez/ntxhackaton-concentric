""" SAVE EEG DATA FOR DATA ANALYSIS """

import time
import mne
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# RECORD DATA
total_time = 30

BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
board = BoardShim(38, params)
board.prepare_session()

board.start_stream()
time.sleep(total_time)
data = board.get_board_data()
board.stop_stream()
board.release_session()

##
# PLOT DATA FOR QUALITY CHECK
eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD.value)
eeg_data = data[eeg_channels, :]
eeg_data /= 1000000

# Creating MNE objects from brainflow data arrays
ch_types = ['eeg'] * len(eeg_channels)
ch_names = BoardShim.get_eeg_names(BoardIds.MUSE_2_BOARD.value)
sfreq = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(eeg_data, info)
# its time to plot something!
raw.plot_psd(average=True)
raw.plot()
filtered = raw.filter(1, 30)
filtered.plot_psd(fmin=1, fmax=30)

##
# SAVE DATA
np.savetxt('data/eyes_open_calm2.txt', data)