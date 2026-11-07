# Recall the basics

## Chapter 1: What sound actually is
A wave is something that has frequency, wavelength and Amplitude. Sound is a mechanical wave:
- It needs a medium. Sound can't travel through a vacuum, which is why space is silent. Light can.
- In air, it's a longitudinal wave. Molecules move back and forth along the direction the wave travels, not up and down like a ripple on water.
- Hearing is a narrower idea. Humans hear roughly 20 Hz to 20 kHz. Vibrations below that are infrasound and above it are ultrasound. Physically they're still sound, but they aren't audible to us.

## Chapter 2: The part that matters most for speech synthesis, VOICE !
Your picture assumes a sound has a frequency. That's true only for a pure tone, like a tuning fork. Real sounds, and especially voices, are many frequencies at once, each with its own amplitude and phase, all changing over time.

- Pitch: When you say "ah", your vocal cords vibrate at a fundamental frequency (F0), about 85–180 Hz for typical male voices and 165–255 Hz for typical female voices. That's what sets the pitch.
- Harmonics: The vibration also produces harmonics at 2×, 3×, 4× F0 and so on.
- Formants: Your throat and mouth shape boost some frequency bands, called formants. Different formant patterns are what make "ee" sound different from "oo", even at the same pitch.
- Timbre: the quality that makes your voice sound like you, comes from this whole mix.

*Spectrograms:* This is exactly why the Fourier transform is the foundation of speech technology. It breaks a complex waveform into its component frequencies. A spectrogram shows that breakdown over time, and it's what most TTS models actually learn to generate.

A quick experiment that builds intuition: record yourself saying a vowel, then view it as a spectrogram (Audacity can do this, or a few lines of Python with librosa). You'll see the fundamental frequency, the stack of harmonics and the formant bands directly.

## Chapter 3: What's actually inside a spectrogram
Your recorded audio starts as a waveform, a long list of amplitude values over time. At a 22,050 Hz sample rate, that's 22,050 numbers per second, each saying how far the air pressure is from normal at that instant.

To build a spectrogram, you run a short-time Fourier transform (STFT):

- Cut the waveform into short, overlapping slices (frames), usually about 20–50 ms each. Speech changes quickly, but within a slice that short it's roughly stable.
- Run a Fourier transform on each slice. This answers: "In this tiny moment, how much of each frequency is present?"
- Stack the results side by side. The result is a 2D grid of numbers, like an image:

Axis	| What it represents
X-axis	Time | one column per frame
Y-axis 	Frequency bins | from low (bottom) to high (top)
Value in each cell (colour)	Magnitude| how much energy that frequency has at that moment, usually on a log (decibel) scale

For example, with typical TTS settings (22,050 Hz audio, 1,024-sample window, 256-sample hop), you get about 86 columns per second and 513 frequency bins per column.

*Magnitude:* how strong the frequency is. The MEL spectrogram keeps this.
*Phase:* where in its cycle each wave is at that moment. The MEL spectrogram throws this away.
Without phase, you can't exactly rebuild the waveform. A mel spectrogram also loses fine frequency detail when bins are merged. So a spectrogram is a lossy representation: it tells you what frequencies are there but not exactly how they line up.


## Chapter 4:  The MEL spectrogram (what TTS models actually use)
TTS models usually work with a mel spectrogram, which compresses the regular spectrogram in two ways:

*Mel scale:*  the frequency bins are merged according to how humans hear. We're good at telling 200 Hz from 300 Hz but bad at telling 8,000 Hz from 8,100 Hz, so low frequencies keep fine detail and high frequencies get grouped coarsely. The 513 bins typically become 80 mel bins.
Log compression: values are converted to a log scale, since perceived loudness is roughly logarithmic.
This gives a compact, perception-focused summary of the audio, which is much easier for a neural network to predict than raw samples.
But it's not the final representation !

Try it yourself: load a recording with librosa, compute a mel spectrogram, then turn it back into audio with librosa.feature.inverse.mel_to_audio (which uses Griffin-Lim). Listen to the result next to the original. The metallic quality you hear is the missing phase, and it shows why vocoders exist.

Getting audio back means estimating the missing information (Phase):

- *Griffin-Lim (classical):* iteratively guesses the phase. It works, but sounds metallic or robotic.
- *Neural vocoders (HiFi-GAN, BigVGAN, Vocos):* learn to generate a realistic waveform from a mel spectrogram. This is what makes modern TTS sound natural.

Text → phonemes → [acoustic model] → mel spectrogram → [vocoder] → waveform → speaker

The acoustic model (for example Tacotron 2 or FastSpeech 2) handles what is said and how: content, timing, pitch, prosody.
The vocoder handles sounding real: rebuilding phase and fine detail.

Not every modern system works this way. End-to-end models like VITS go straight from text to waveform internally. Codec-based models like VALL-E use a neural audio codec to turn audio into discrete tokens, then generate those tokens instead of a spectrogram. In every case, though, the last step produces a waveform.