# Visual Noise Remover - Audio Denoising Tool

Visual Noise Remover is a Python application that provides a graphical interface for removing noise from audio files using spectral editing techniques. The tool allows users to visually select noise regions in the spectrogram and apply various noise reduction algorithms to clean the audio.

# Features

- Visual Noise Selection: Select noise regions directly on the spectrogram

- Multiple Noise Reduction Methods:

  - Spectral Subtraction

  - Wiener Filter

  - Convolution Filter

- Adjustable Parameters:

  - Noise threshold

  - Reduction strength

- Audio Preview:

  - Play original or processed audio segments

  - Adjustable preview position and duration

- Spectrogram Visualization:

  - Interactive display

  - Color-coded selections

- File Support:

  - Load WAV, AIFF, FLAC, OGG, MP3 files

  - Save processed audio in multiple formats

# Dependencies
Make sure you have the following installed:
```
pip install numpy matplotlib librosa soundfile sounddevice pydub
```

# Images

![image](https://github.com/user-attachments/assets/5d8f050d-d936-4e6f-a647-1f42560b5f50)
![image](https://github.com/user-attachments/assets/eec3b6b8-88f1-42d4-a5d5-a5d27b087ac7)
