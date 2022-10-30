# Concentric

A visual neurofeedback application for meditation.

## Description

After a brief calibration period, the meditator can visualize a pink circle on screen. 
The radius of the circle will change depending on his level of concentration, as measured in the EEG recorded by the MUSE2 device.
The duration of the session and calibration period can be adjusted for a longer practice.

The application is only compatible with MUSE2 devices.

### Executing program

* To use the neurofeedback application, execute `feedback_circle.py`
* To save EEG data collected from a MUSE2 device, execute `save_trials_data.py`
* To plot EEG data and extract features, execute `plot_data.py`

### Dependencies

Brainflow, numpy, matplotlib

