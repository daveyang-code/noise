import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import soundfile as sf
import librosa
import sounddevice as sd
import threading
import os


class VisualNoiseRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Noise Remover")

        # Initialize variables
        self.audio = None
        self.audio_original = None
        self.sr = None
        self.file_name = None
        self.preview_duration = 3.0  # seconds
        self.audio_start_point = 0  # seconds
        self.n_fft = 2048
        self.hop_length = 512
        self.stft = None
        self.mag = None
        self.phase = None
        self.db = None
        self.selections = []
        self.current_selection = None
        self.attenuation_level = 30  # Default attenuation in dB
        self.noise_profile = None
        self.noise_threshold = 1.5  # Noise threshold multiplier
        self.method = "spectral_subtraction"  # Default method

        # Reduction methods
        self.reduction_methods = {
            "spectral_subtraction": self.spectral_subtraction,
            "wiener_filter": self.wiener_filter,
            "convolution_filter": self.convolution_filter,
        }

        # Create GUI layout
        self.create_widgets()

    def create_widgets(self):
        # Create main frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.spectrogram_frame = ttk.Frame(self.main_frame)
        self.spectrogram_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create spectrogram display
        self.create_spectrogram_display()

        # Create control panel
        self.create_control_panel()

        # Create menu
        self.create_menu()

    def create_menu(self):
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_audio)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        self.root.config(menu=menubar)

    def create_spectrogram_display(self):
        # Create matplotlib figure
        self.fig = plt.figure(figsize=(10, 6))
        self.ax_spec = self.fig.add_subplot(111)

        # Initialize empty spectrogram with proper dimensions
        empty_spectrogram = np.zeros((100, 100))  # Temporary empty array
        self.img = self.ax_spec.imshow(
            empty_spectrogram, origin="lower", aspect="auto", cmap="viridis"
        )
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_title(
            "Spectrogram - Select areas containing ONLY noise to create a noise profile"
        )

        # Add colorbar
        self.cbar = self.fig.colorbar(self.img, ax=self.ax_spec, format="%+2.0f dB")

        # Add preview line
        self.preview_line = None

        # Create canvas for matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.spectrogram_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Set up the selection tool
        self.rect_selector = RectangleSelector(
            self.ax_spec,
            self.on_select,
            useblit=True,
            button=[1],
            interactive=True,
            props=dict(facecolor="red", edgecolor="red", alpha=0.3),
        )

        # Connect click handler for preview position
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def create_control_panel(self):
        # Create notebook for different control sections
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill=tk.X, padx=5, pady=5)

        # Noise Profile Tab
        self.profile_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.profile_frame, text="Noise Profile")

        # Reduction Tab
        self.reduction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reduction_frame, text="Reduction")

        # Preview Tab
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="Preview")

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready: Open an audio file to begin")
        self.status_bar = ttk.Label(
            self.control_frame, textvariable=self.status_var, relief=tk.SUNKEN
        )
        self.status_bar.pack(fill=tk.X, padx=5, pady=5)

        # Populate the tabs
        self.create_profile_tab()
        self.create_reduction_tab()
        self.create_preview_tab()

    def create_profile_tab(self):
        # Create noise profile controls
        ttk.Label(self.profile_frame, text="Noise Profile Controls").grid(
            row=0, column=0, columnspan=2, pady=5
        )

        self.create_profile_btn = ttk.Button(
            self.profile_frame,
            text="Create Noise Profile",
            command=self.create_noise_profile,
        )
        self.create_profile_btn.grid(row=1, column=0, padx=5, pady=5)

        self.clear_selections_btn = ttk.Button(
            self.profile_frame, text="Clear Selections", command=self.clear_selections
        )
        self.clear_selections_btn.grid(row=1, column=1, padx=5, pady=5)

        self.reset_btn = ttk.Button(
            self.profile_frame, text="Reset to Original", command=self.reset
        )
        self.reset_btn.grid(row=1, column=2, padx=5, pady=5)

        # Threshold slider
        self.threshold_slider = ttk.Scale(
            self.profile_frame,
            from_=1.0,
            to=5.0,
            value=self.noise_threshold,
            orient=tk.HORIZONTAL,
            command=lambda v: self.update_threshold(float(v)),
        )
        self.threshold_slider.grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW
        )

        self.threshold_label = ttk.Label(
            self.profile_frame, text=f"Noise Threshold: {self.noise_threshold:.2f}x"
        )
        self.threshold_label.grid(row=3, column=0, columnspan=2, pady=5)

    def create_reduction_tab(self):
        # Create noise reduction controls
        ttk.Label(self.reduction_frame, text="Noise Reduction Controls").grid(
            row=0, column=0, columnspan=3, pady=5
        )

        # Method selection
        self.method_var = tk.StringVar(value="Spectral Subtraction")
        ttk.Label(self.reduction_frame, text="Method:").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )

        self.method_menu = ttk.Combobox(
            self.reduction_frame,
            textvariable=self.method_var,
            values=["Spectral Subtraction", "Wiener Filter", "Convolution Filter"],
        )
        self.method_menu.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.method_menu.bind("<<ComboboxSelected>>", self.set_method)

        # Attenuation slider
        self.attenuation_slider = ttk.Scale(
            self.reduction_frame,
            from_=0,
            to=100,
            value=self.attenuation_level,
            orient=tk.HORIZONTAL,
            command=lambda v: self.update_attenuation(float(v)),
        )
        self.attenuation_slider.grid(
            row=2, column=0, columnspan=3, padx=5, pady=5, sticky=tk.EW
        )

        self.attenuation_label = ttk.Label(
            self.reduction_frame,
            text=f"Reduction Strength: {self.attenuation_level:.1f} dB",
        )
        self.attenuation_label.grid(row=3, column=0, columnspan=3, pady=5)

        # Apply reduction button
        self.apply_reduction_btn = ttk.Button(
            self.reduction_frame,
            text="Apply Reduction",
            command=self.apply_noise_reduction,
        )
        self.apply_reduction_btn.grid(
            row=4, column=0, columnspan=3, padx=5, pady=5, sticky=tk.EW
        )

    def create_preview_tab(self):
        # Create preview controls
        ttk.Label(self.preview_frame, text="Audio Preview Controls").grid(
            row=0, column=0, columnspan=3, pady=5
        )

        # Preview position
        ttk.Label(self.preview_frame, text="Preview Position (s):").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )

        self.preview_pos_var = tk.DoubleVar(value=self.audio_start_point)
        self.preview_pos_entry = ttk.Entry(
            self.preview_frame, textvariable=self.preview_pos_var, width=8
        )
        self.preview_pos_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Preview duration
        ttk.Label(self.preview_frame, text="Duration (s):").grid(
            row=2, column=0, padx=5, pady=5, sticky=tk.W
        )

        self.preview_dur_var = tk.DoubleVar(value=self.preview_duration)
        self.preview_dur_entry = ttk.Entry(
            self.preview_frame, textvariable=self.preview_dur_var, width=8
        )
        self.preview_dur_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        # Preview buttons
        self.preview_orig_btn = ttk.Button(
            self.preview_frame, text="Preview Original", command=self.preview_original
        )
        self.preview_orig_btn.grid(row=3, column=0, padx=5, pady=5, sticky=tk.EW)

        self.preview_proc_btn = ttk.Button(
            self.preview_frame, text="Preview Processed", command=self.preview_processed
        )
        self.preview_proc_btn.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)

        self.stop_preview_btn = ttk.Button(
            self.preview_frame, text="Stop Preview", command=self.stop_preview
        )
        self.stop_preview_btn.grid(row=3, column=2, padx=5, pady=5, sticky=tk.EW)

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Open Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.aiff *.aif *.flac *.ogg *.mp3"),
                ("All Files", "*.*"),
            ],
        )

        if file_path:
            try:
                self.file_name = file_path
                self.audio, self.sr = librosa.load(file_path, sr=None)
                self.audio_original = self.audio.copy()

                # Compute spectrogram
                self.stft = librosa.stft(
                    self.audio, n_fft=self.n_fft, hop_length=self.hop_length
                )
                self.mag = np.abs(self.stft)
                self.phase = np.angle(self.stft)
                self.db = librosa.amplitude_to_db(self.mag, ref=np.max)

                # Update spectrogram display
                self.img.set_data(self.db)
                self.img.set_extent([0, self.audio.shape[0] / self.sr, 0, self.sr / 2])
                self.img.set_clim(
                    vmin=np.min(self.db), vmax=np.max(self.db)
                )  # Update color scale

                # Update axis limits
                self.ax_spec.set_xlim(0, self.audio.shape[0] / self.sr)
                self.ax_spec.set_ylim(0, self.sr / 2)

                # Update colorbar
                self.cbar.update_normal(self.img)

                # Redraw the canvas
                self.canvas.draw()

                self.status_var.set(
                    f"Loaded: {os.path.basename(file_path)} - {self.sr}Hz, {len(self.audio)/self.sr:.2f}s"
                )

                # Enable controls
                self.enable_controls(True)

            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {str(e)}")
                self.status_var.set("Error loading file")

    def enable_controls(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.create_profile_btn.config(state=state)
        self.clear_selections_btn.config(state=state)
        self.reset_btn.config(state=state)
        self.apply_reduction_btn.config(state=state)
        self.preview_orig_btn.config(state=state)
        self.preview_proc_btn.config(state=state)
        self.stop_preview_btn.config(state=state)

    def on_click(self, event):
        """Handle mouse clicks on the spectrogram."""
        if event.inaxes != self.ax_spec or event.button != 1:
            return

        # Check if it's not a RectangleSelector event
        if self.rect_selector.active:
            return

        # Set the audio preview start point
        self.audio_start_point = event.xdata
        self.preview_pos_var.set(f"{self.audio_start_point:.2f}")

        # Update the preview marker
        if self.preview_line:
            self.preview_line.remove()

        # Create vertical line at click position
        self.preview_line = self.ax_spec.axvline(
            x=self.audio_start_point, color="yellow", linewidth=2
        )
        self.status_var.set(f"Preview point set at {self.audio_start_point:.2f}s")
        self.canvas.draw()

    def on_select(self, eclick, erelease):
        """Called when a rectangle selection is made."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Convert time to frames and frequency to bins
        t1 = max(0, int(x1 * self.sr / self.hop_length))
        t2 = min(self.stft.shape[1], int(x2 * self.sr / self.hop_length))
        f1 = max(0, int(y1 * self.n_fft / self.sr))
        f2 = min(self.stft.shape[0], int(y2 * self.n_fft / self.sr))

        # Store the selection
        self.current_selection = {
            "time": (t1, t2),
            "freq": (f1, f2),
            "rect": plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, facecolor="red", edgecolor="red", alpha=0.3
            ),
        }
        self.selections.append(self.current_selection)
        self.ax_spec.add_patch(self.current_selection["rect"])
        self.canvas.draw()

        # Update status text
        self.status_var.set(
            f"Added selection: {len(self.selections)} noise areas selected"
        )

    def update_threshold(self, val):
        """Update the noise threshold value."""
        self.noise_threshold = val
        self.threshold_label.config(text=f"Noise Threshold: {val:.2f}x")

    def update_attenuation(self, val):
        """Update the attenuation level from the slider."""
        self.attenuation_level = val
        self.attenuation_label.config(text=f"Reduction Strength: {val:.1f} dB")

    def set_method(self, event=None):
        """Set the noise reduction method."""
        method_map = {
            "Spectral Subtraction": "spectral_subtraction",
            "Wiener Filter": "wiener_filter",
            "Convolution Filter": "convolution_filter",
        }
        self.method = method_map[self.method_var.get()]
        self.status_var.set(f"Method set to: {self.method_var.get()}")

    def clear_selections(self):
        """Clear all selection rectangles without resetting audio."""
        for selection in self.selections:
            selection["rect"].remove()
        self.selections = []
        self.canvas.draw()
        self.status_var.set("All selections cleared")

    def create_noise_profile(self):
        """Create a noise profile from the selected areas."""
        if not self.selections:
            self.status_var.set(
                "No areas selected! Select areas containing ONLY noise first."
            )
            return

        # Initialize noise profile with zeros
        self.noise_profile = np.zeros((self.n_fft // 2 + 1,), dtype=np.float32)
        total_frames = 0

        # Collect noise statistics from all selected regions
        for selection in self.selections:
            t1, t2 = selection["time"]
            f1, f2 = selection["freq"]

            # Extract the noise region
            noise_region = self.mag[f1:f2, t1:t2]

            # Add to the noise profile (average power spectrum)
            frames_in_region = noise_region.shape[1]
            if frames_in_region > 0:
                for f_idx in range(f1, f2):
                    if f_idx < self.noise_profile.shape[0]:
                        # Accumulate the power
                        self.noise_profile[f_idx] += (
                            np.mean(noise_region[f_idx - f1, :] ** 2) * frames_in_region
                        )

                total_frames += frames_in_region

        # Normalize by total frames
        if total_frames > 0:
            self.noise_profile /= total_frames

            # Take the square root to get magnitude
            self.noise_profile = np.sqrt(self.noise_profile)

            # Highlight the selections in a different color to show they're now part of the noise profile
            for selection in self.selections:
                selection["rect"].set_facecolor("blue")
                selection["rect"].set_edgecolor("cyan")
                selection["rect"].set_alpha(0.4)

            self.status_var.set(
                "Noise profile created. Click 'APPLY Reduction' to remove noise."
            )
            self.canvas.draw()
        else:
            self.status_var.set(
                "Error: Could not create noise profile from selections."
            )

    def spectral_subtraction(self, stft, noise_profile, reduction_factor):
        """Apply spectral subtraction method."""
        mag = np.abs(stft)
        phase = np.angle(stft)

        # Reshape noise profile to broadcast across time
        noise_profile_reshaped = noise_profile[:, np.newaxis]

        # Apply noise threshold
        noise_mask = noise_profile_reshaped * self.noise_threshold

        # Subtract noise profile from magnitude
        reduced_mag = np.maximum(
            mag - noise_mask * reduction_factor,
            mag * 0.05,  # Floor to avoid complete silence (5% of original)
        )

        # Reconstruct complex STFT
        return reduced_mag * np.exp(1j * phase)

    def wiener_filter(self, stft, noise_profile, reduction_factor):
        """Apply Wiener filter method."""
        power = np.abs(stft) ** 2
        phase = np.angle(stft)

        # Reshape noise profile to broadcast across time
        noise_power = (noise_profile**2)[:, np.newaxis] * self.noise_threshold

        # Wiener filter formula: H = S / (S + N)
        gain = np.maximum(
            1 - (noise_power * reduction_factor / (power + 1e-10)), 0.05  # Minimum gain
        )

        # Apply gain to original STFT
        return gain * stft

    def convolution_filter(self, stft, noise_profile, reduction_factor):
        """Apply convolution-based filter (advanced spectral subtraction)."""
        mag = np.abs(stft)
        phase = np.angle(stft)

        # Create 2D noise profile that can vary with frequency
        noise_profile_2d = noise_profile[:, np.newaxis] * self.noise_threshold

        # Estimate spectral characteristics
        signal_power = mag**2
        noise_power = noise_profile_2d**2

        # Calculate SNR (signal-to-noise ratio) for each time-frequency bin
        snr = np.maximum(signal_power / (noise_power + 1e-10) - 1.0, 0)

        # Apply soft-thresholding based on SNR
        gain = np.sqrt(snr / (snr + 1))

        # Apply additional attenuation based on user setting
        reduction_scale = (
            1.0 - (reduction_factor / 100.0) * 0.95
        )  # Scale to 0.05-1.0 range
        gain = np.maximum(gain * reduction_scale, 0.01)

        # Apply gain and reconstruct complex STFT
        return gain * mag * np.exp(1j * phase)

    def apply_noise_reduction(self):
        """Apply noise reduction using the created noise profile."""
        if self.noise_profile is None:
            self.status_var.set(
                "No noise profile created! Use 'CREATE Noise Profile' first."
            )
            return

        # Convert attenuation level to reduction factor
        reduction_factor = self.attenuation_level / 100.0

        # Apply the selected noise reduction method
        method_func = self.reduction_methods.get(self.method, self.spectral_subtraction)
        reduced_stft = method_func(self.stft, self.noise_profile, reduction_factor)

        # Inverse STFT to get the processed audio
        self.audio = librosa.istft(
            reduced_stft, hop_length=self.hop_length, length=len(self.audio_original)
        )

        # Update the spectrogram display
        self.mag = np.abs(reduced_stft)
        self.phase = np.angle(reduced_stft)
        self.db = librosa.amplitude_to_db(self.mag, ref=np.max)
        self.img.set_array(self.db)
        self.canvas.draw()

        self.status_var.set(
            f"Noise reduction applied using {self.method.replace('_', ' ').title()}"
        )

    def reset(self):
        """Reset the audio and clear selections."""
        self.audio = self.audio_original.copy()

        # Clear all selections
        for selection in self.selections:
            selection["rect"].remove()
        self.selections = []

        # Reset noise profile
        self.noise_profile = None

        # Recompute spectrogram
        self.stft = librosa.stft(
            self.audio, n_fft=self.n_fft, hop_length=self.hop_length
        )
        self.mag = np.abs(self.stft)
        self.phase = np.angle(self.stft)
        self.db = librosa.amplitude_to_db(self.mag, ref=np.max)

        # Update the display
        self.img.set_array(self.db)
        self.canvas.draw()

        self.status_var.set("Reset to original audio")

    def save_audio(self):
        """Save the processed audio to a file."""
        if self.audio is None:
            messagebox.showerror("Error", "No audio to save!")
            return

        output_file = filedialog.asksaveasfilename(
            title="Save Processed Audio",
            defaultextension=".wav",
            filetypes=[
                ("WAV Files", "*.wav"),
                ("AIFF Files", "*.aiff *.aif"),
                ("FLAC Files", "*.flac"),
                ("OGG Files", "*.ogg"),
                ("All Files", "*.*"),
            ],
        )

        if output_file:
            try:
                sf.write(output_file, self.audio, self.sr)
                self.status_var.set(f"Audio saved to {os.path.basename(output_file)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving audio: {str(e)}")
                self.status_var.set(f"Error saving audio: {str(e)}")

    def preview_original(self):
        """Play a short preview of the original audio."""
        try:
            # Stop any ongoing playback
            sd.stop()

            # Get preview position and duration from UI
            self.audio_start_point = float(self.preview_pos_var.get())
            self.preview_duration = float(self.preview_dur_var.get())

            # Calculate start and end samples for preview
            start_sample = int(self.audio_start_point * self.sr)
            end_sample = start_sample + int(self.preview_duration * self.sr)

            # Ensure we don't exceed audio length
            if end_sample > len(self.audio_original):
                end_sample = len(self.audio_original)

            # Extract preview segment
            preview_segment = self.audio_original[start_sample:end_sample]

            # Play the preview
            sd.play(preview_segment, self.sr)

            self.status_var.set(
                f"Previewing original audio from {self.audio_start_point:.2f}s"
            )

            # Highlight preview region on spectrogram
            self.highlight_preview_region()

        except Exception as e:
            messagebox.showerror("Error", f"Error playing preview: {str(e)}")
            self.status_var.set(f"Error playing preview: {str(e)}")

    def preview_processed(self):
        """Play a short preview of the processed audio."""
        try:
            # Stop any ongoing playback
            sd.stop()

            # Get preview position and duration from UI
            self.audio_start_point = float(self.preview_pos_var.get())
            self.preview_duration = float(self.preview_dur_var.get())

            # Calculate start and end samples for preview
            start_sample = int(self.audio_start_point * self.sr)
            end_sample = start_sample + int(self.preview_duration * self.sr)

            # Ensure we don't exceed audio length
            if end_sample > len(self.audio):
                end_sample = len(self.audio)

            # Extract preview segment
            preview_segment = self.audio[start_sample:end_sample]

            # Play the preview
            sd.play(preview_segment, self.sr)

            self.status_var.set(
                f"Previewing processed audio from {self.audio_start_point:.2f}s"
            )

            # Highlight preview region on spectrogram
            self.highlight_preview_region()

        except Exception as e:
            messagebox.showerror("Error", f"Error playing preview: {str(e)}")
            self.status_var.set(f"Error playing preview: {str(e)}")

    def stop_preview(self):
        """Stop audio preview playback."""
        sd.stop()
        self.status_var.set("Preview stopped")

    def highlight_preview_region(self):
        """Temporarily highlight the preview region on the spectrogram."""
        # Remove any existing highlight
        for artist in self.ax_spec.get_children():
            if isinstance(artist, plt.Rectangle) and getattr(
                artist, "preview_rect", False
            ):
                artist.remove()

        # Calculate preview region
        start_time = float(self.preview_pos_var.get())
        end_time = start_time + float(self.preview_dur_var.get())

        # Ensure end time doesn't exceed audio length
        if end_time > len(self.audio) / self.sr:
            end_time = len(self.audio) / self.sr

        # Add a rectangle highlight
        rect = plt.Rectangle(
            (start_time, 0),
            end_time - start_time,
            self.sr / 2,
            facecolor="yellow",
            edgecolor="white",
            alpha=0.2,
            linewidth=2,
        )
        rect.preview_rect = True  # Mark as preview rectangle
        self.ax_spec.add_patch(rect)
        self.canvas.draw()

        # Set up a timer to remove the highlight after preview duration
        threading.Timer(self.preview_duration, self.remove_preview_highlight).start()

    def remove_preview_highlight(self):
        """Remove the preview region highlight."""
        for artist in self.ax_spec.get_children():
            if isinstance(artist, plt.Rectangle) and getattr(
                artist, "preview_rect", False
            ):
                artist.remove()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = VisualNoiseRemoverApp(root)
    root.mainloop()
