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

---

## November 8, 2026: analysis snapshot archived before compaction

Historical research and equations follow verbatim. References and implementation
claims describe their original snapshot; the dated decision update below supersedes
old recommendations. Original analysis SHA256: `27bebba706607e52a1ffde3fb77221f16543a0fbd0774456b2c38c074bfe97b1`.

# Expressive Own-Voice Speech Synthesis for YapIt

**A technical review and reproducible model-selection study**

Implementation file/line references below belong to the original research
snapshot. For the maintained folder layout and shared-utility/model-loader
boundaries, use [the developer guide](dev.readme.md).

| Metadata | Value |
|---|---|
| Research date | November 7, 2026 |
| Audience | Software developers evaluating a local personal-voice narration system |
| Owner | Arka |
| Status | Literature and implementation review complete; comparative listening experiment not yet performed |
| Decision scope | English travel narration, preserving Arka's voice, on Windows with an RTX 4070 SUPER eGPU or a CPU-only travelling configuration |
| Evidence policy | Primary papers, official repositories/model cards, and explicitly identified local measurements |

### Abstract

YapIt's existing Qwen3-TTS integration establishes a working voice-cloning baseline, not evidence that Qwen is the best expressive narrator. The central requirement is **controlling delivery while preserving a particular speaker**, rather than generating an arbitrary emotional voice. This review distinguishes numeric prosody controls, emotion-conditioned acoustic models, reference-conditioned synthesis, and models supporting speaker reference plus independent instructions. FastSpeech 2, EmotiVoice, and a specific Emotion-Aware-TTS implementation illuminate useful mechanisms but do not provide equivalent zero-shot own-voice workflows. Current alternatives include VoxCPM2, CosyVoice 3, and IndexTTS 2.5. Their released interfaces better match independently controllable personal-voice narration, with different dependency, licensing, and deployment costs.

**Recommendation:** prioritize **VoxCPM2 and Fun-CosyVoice3-0.5B-2512** for a controlled local comparison against Qwen's neutral and expressive-reference baselines. Treat IndexTTS 2.5 as the explicit emotion/duration-control challenger, conditional on accepting its custom license and isolated Python 3.11 runtime. No model is declared the perceptual winner without listening tests. CPU environment portability and real-time CPU synthesis are separate acceptance questions. [9, 13-23]

## Problem Statement

### 1. Research questions and non-negotiable requirements

The question is not whether a model advertises "emotion." It is whether YapIt can produce **recognizably Arka**, speaking the approved words, with an appropriate and reproducible delivery:

| Research question | Required evidence |
|---|---|
| Can delivery change without changing speaker identity? | Matched speaker reference, contrasting styles, held-out speaker comparisons, and human listening |
| Does "lively travel vlog" sound natural rather than exaggerated? | Style-fit and naturalness ratings on actual travel scripts |
| Can relaxed narration and ASMR-like delivery remain intelligible? | Separate listening and transcription checks; an emotion label alone is insufficient |
| Can an approved transcript fit its video? | Measured duration without omitted words or excessive post-generation stretching |
| Does it run on the actual eGPU and laptop? | Native-Windows installation, real synthesis, memory/latency measurements, and explicit CPU-mode tests |
| Can it be maintained and redistributed appropriately? | Released checkpoints, dependency compatibility, code/weight licenses, and maintenance status |

These are **product requirements and proposed tests**, not claims that every candidate already meets them. There must be no silent replacement of Arka with a stock speaker, silent CPU fallback from an explicitly selected GPU mode, or transcript rewriting to conceal duration failures.

### 2. What is implemented and what was actually observed

The following separates code evidence from quality claims:

| Finding | Evidence and limitation |
|---|---|
| Default synthesis uses `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `src\speech.py:16,31-73`; reference audio and its transcript construct the clone prompt |
| Audio is generated independently for each script beat | `src\pipeline.py:92-104`; cross-beat expressive continuity is not evaluated |
| Reference-driven style selection is now implemented provisionally | `src\models.py:66-90`; `src\voices.py:168-251`; this selects a different recording, not an emotion-trained model |
| Missing styles and style use with a fine-tuned checkpoint fail explicitly | `src\voices.py:179-190,230-238`; no silent default-reference substitution |
| Style provenance is recorded | `src\pipeline.py:75-90`; the description explicitly requires listening review |
| The existing CPU render is 34.20004 s; the source soundtrack is 28.02792 s | Local `soundfile.info` measurements of `data\generated_audio\7ab3df348947460c84dad25125e1aa14\raw.wav` and `tests\data\arka.ladakhi_marmots.wav` |
| The duration rejection was correct | `34.20004 / 28.02792 = 1.2202`, outside the allowed post-processing factor; mismatch also exceeds 5 s. `src\media.py:97-113` |
| The user found the earlier delivery insufficiently expressive | User listening feedback, **not** a MOS study or controlled comparison |
| Hardware is an NVIDIA GeForce RTX 4070 SUPER, driver 610.88, 12282 MiB VRAM | Local `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader` |
| Style-interface regression tests pass | 25 focused tests across `test_voices.py`, `test_pipeline.py`, and `test_interfaces.py`; these verify routing and safety, **not** emotion quality |

No expressive reference recording has yet been supplied for the selected lively-vlog style. No local head-to-head experiment against the shortlisted models has been run. Therefore neither the new style option nor the GPU environment proves expressive quality.

The current training wrapper performs supervised personal-voice adaptation through upstream Qwen scripts. Its records contain `audio`, `text`, and `ref_audio`, not a multi-style emotion-supervision schema. It currently rejects native-Windows training and requires Linux/WSL2 plus CUDA/bfloat16/FlashAttention. Its minimum-data checks are **project guardrails**, not guarantees of good adaptation. `src\training.py:16-99`.

### 3. Terminology and the ambiguous "emovide"

For this investigation:

| Term | Operational meaning |
|---|---|
| Speaker identity | The voice should remain recognizable as Arka, including characteristic timbre and accent |
| Prosody | Timing, pitch contour, stress, rhythm, and pauses |
| Emotion | An affective state such as happiness or sadness; not necessarily the right narration style |
| Delivery style | Communicative manner, such as curious vlog narration or calm conversational explanation |
| Paralinguistic event | A non-word vocal action such as a breath or chuckle |
| Zero-shot cloning | Conditioning a pretrained model on a new speaker reference without optimizing its weights |
| Fine-tuning / SFT | Supervised fine-tuning: updating pretrained weights using paired examples |
| LoRA | Low-rank adaptation: training smaller weight updates instead of updating every original parameter |
| RTF | Real-time factor: synthesis time divided by generated audio duration; below 1 is faster than real time |
| MOS | Mean opinion score: an average human rating collected under a specified listening protocol |

The definitions above establish evaluation axes; they do not imply that a particular architecture perfectly separates them.

**"emovide" is unresolved rather than silently corrected.** Plausible references are **EmotiVoice**, a released engine; **EMOVIE**, a Mandarin emotional-speech paper/dataset and EMSpeech model; or **EmoV-DB**, an emotional speech dataset, not a synthesizer. This review covers all three identities where relevant. EMOVIE reports 9,724 Mandarin samples; the specific English Emotion-Aware-TTS project uses EmoV-DB. These are not interchangeable products. [3-7]

### 4. Research method and scope

The literature was checked against current official repository instructions and released model cards on the research date. Paper publication, repository activity, package version, and model release are distinguished. A README-only update is not treated as a new trained checkpoint. A feature present in one model variant is not attributed automatically to every variant.

The evidence classes throughout this document are:

- **Implemented/measured locally:** source inspection, regression tests, existing audio durations, and hardware detection.
- **Documented by authors:** architectures, exposed APIs, release announcements, model-card resource estimates, and benchmark results.
- **Proposed:** experimental design, acceptance thresholds, workload assumptions, and integration plan.
- **Unknown:** actual Arka voice quality, Indian-English accent preservation, native-Windows feasibility for uninstalled candidates, and their local latency/memory.

This is a scoped decision review, not a systematic survey claiming to exhaust every 2026 TTS release. Repository `main`/`master` documentation is mutable: record exact code and weight revisions before executing the benchmark.

## Implementation

### 1. Architecture taxonomy: why the candidates are not equivalent

An organizing model for the requirement is:

$$
x \sim p_\theta(x \mid y, s, e, \pi, d)
$$

Here `x` is synthesized audio, `y` the approved text, `s` speaker information, `e` requested emotion/delivery, `pi` prosodic context, and `d` a duration constraint. This is an **analytical notation**, not a claim that every backend exposes these five arguments.

The important distinction is between a model that receives one reference containing both identity and delivery, and a model that can condition on identity separately from the requested delivery. Strong conditioning can still leak one attribute into another; "disentangled" in an architecture description is a hypothesis to test perceptually.

#### FastSpeech 2: explicit variance control, not native emotional cloning

FastSpeech 2 encodes phonemes, expands them using predicted duration, adds pitch/energy conditioning, and predicts a mel spectrogram in parallel. A vocoder turns that representation into audio; FastSpeech 2s is the direct-waveform variant. Training uses aligned ground-truth durations, pitch, and energy; inference predicts them. The revised paper uses a continuous wavelet transform for pitch modeling. The widely used `ming024` implementation is a community implementation, not an official Microsoft release. [1, 2]

A simplified description is:

$$
H=\operatorname{Encoder}(y), \qquad
\hat M=\operatorname{Decoder}\bigl(\operatorname{LR}(H,\hat d)+E_p(\hat p)+E_a(\hat a)\bigr)
$$

`LR` is the length regulator; `p` is pitch and `a` acoustic energy. The equation abstracts implementation ordering. Numeric scaling changes timing, pitch, or energy, but does not establish a learned "curious travel narrator" control. The original work leaves emotion/style/speaker extensions to future work. Thus comparing original FastSpeech 2 directly to pretrained own-voice foundation models without adding speaker/style training would be an unequal experiment. [1]

#### Emotion-conditioned FastSpeech variants: labels must match the learned speech

The specific `ionut-cmd/Emotion-Aware-TTS` project pairs a modified FastSpeech 2 with text-emotion analysis. Its acoustic model adds speaker and emotion embeddings, while its classifier uses RoBERTa trained on GoEmotions. The speech training uses four English EmoV-DB speakers and five categories: amused, anger, disgust, neutral, and sleepiness. Manual emotion-ID inference is also exposed. This is a real trained research system, not merely a wrapper pretending to add emotion. [6, 7]

However, four trained speaker IDs are not arbitrary reference-audio cloning. The trained label set also does not establish ASMR or lively-vlog controllability. The paper reports timing and waveform RMSE, not a human naturalness/emotion study or an own-voice identity benchmark. Its cross-system speed table involves different GPUs, so it cannot determine the fastest candidate here. [6]

**Engineering inference:** text sentiment is not the same as desired delivery. A neutral statement about marmots may appropriately be narrated with curiosity. Prefer explicit user style plus contextual intent over automatically forcing a text classifier's label.

#### EmotiVoice: prompt-conditioned acoustic generation with a closed speaker table

EmotiVoice's released path combines a speaker-ID embedding with BERT-derived style/content embeddings and acoustic predictors, using a PromptTTS-related model and joint acoustic/vocoder inference. Its advertised large voice inventory is a lookup table of known speakers, not a reference encoder for any new speaker. An own-voice recipe exists, but it involves training on paired personal recordings; emotionally expressive personal output needs appropriate emotional examples. [3, 4]

The observed September 3, 2026 repository update changed Chinese documentation, not model code. The roadmap's next-model and longer-text items remain unchecked. Its historical Python 3.8 setup and checkpoint-license uncertainty increase integration work. EmotiVoice remains a useful **prompt-control and adaptation baseline**, not the shortest path to zero-shot Arka narration. [3-5]

#### Codec-language models: expressive information and API availability are different

A codec-language model generates speech codes autoregressively:

$$
p(z\mid c)=\prod_t p(z_t\mid z_{<t},c), \qquad x=\operatorname{Decode}(z)
$$

`c` represents the available conditioning. This factorization explains why generation has sequential cost and why conditioning matters; it is not a speed comparison.

Qwen3-TTS uses discrete multi-codebook speech modeling with its 12 Hz tokenizer. **YapIt's Base checkpoint provides reference cloning; it does not expose the 1.7B CustomVoice instruction workflow.** The official released-model table also does not advertise instruction control for the 0.6B models. CustomVoice's stock speakers and VoiceDesign's newly created persona do not automatically satisfy "keep Arka." [8-10]

Using an expressive recording of Arka as the Base reference is a sensible low-cost experiment. It transfers an example, rather than separately specifying identity and emotion. Temperature/top-p changes stochastic sampling, not a calibrated emotion axis. Qwen is worth retaining as the working comparator, not defending as an already established winner. `src\speech.py:63-92`; [9].

#### Hybrid and continuous-latent models: finer control with different costs

CosyVoice combines language-model speech-token generation with chunk-aware flow matching. Its third-generation paper improves the tokenizer using supervision including recognition, emotion, language, audio events, and speaker analysis. The released instruction interface can receive both a speaker recording and an instruction. However, the May 2025 paper discusses scaling to **1.5B**, whereas the current public `2512` checkpoint selected here is **0.5B**. Paper-scale measurements must not be relabeled as measurements of that checkpoint. [21-23]

IndexTTS 2.5 separates text-to-semantic generation from semantic-to-mel rendering and a vocoder. Its report describes reducing semantic frame rate from 50 to 25 Hz, replacing the semantic-to-mel backbone with Zipformer, and applying GRPO post-training. GRPO is a reinforcement-learning method comparing groups of sampled outputs. The released interface provides separate speaker/emotion references, an eight-dimensional emotion vector, optional text-to-emotion conversion, and a duration factor. These controls directly address the requirements but do not prove perfect identity/emotion separation. [16-20]

VoxCPM2 generates continuous speech-latent patches with hierarchical semantic/acoustic language modeling and local diffusion. "Tokenizer-free" means no **external discrete speech tokenizer**; its paper still includes an internal finite-scalar-quantization bottleneck. Separate reference conditioning and style-guided cloning are exposed in one model. Its 16 kHz reference encoding and 48 kHz decoding do not imply that an old recording gains missing original information or that higher sample rate proves more natural speech. [13-15]

Flow matching learns a velocity field for transforming noise into conditioned speech representations. Conceptually:

$$
\mathcal L_{\mathrm{FM}} =
\mathbb E\left[\|v_\theta(x_t,t,c)-u_t\|^2\right],
\qquad \frac{dx_t}{dt}=v_\theta(x_t,t,c).
$$

`u_t` is the target training velocity. This explains why iterative sampling steps introduce a quality/cost lever. A diffusion or flow model is not automatically more emotional than a codec model; the conditioning, data, and inference path must support the requested behavior. [15, 30]

#### Style diffusion and reference continuation

StyleTTS 2 samples a latent style through diffusion and uses a pretrained speech-language-model discriminator for training. Its multispeaker model supports reference-based zero-shot adaptation, with a documented fine-tuning route. Its strong published listening results are dataset-specific, not evidence on Arka's recordings. Deployment involves phonemization and package/license choices that deserve explicit review. [28, 29]

F5-TTS uses conditional flow matching with a diffusion Transformer and reference audio/text. The latest base checkpoint advertised by its official README is **F5-TTS v1**, released March 12, 2025. Multi-style inference selects different references; that is not automatically independent speaker/emotion control. Its permissive code license does **not** make the pretrained weights commercially unrestricted: those are CC-BY-NC. [30, 31]

### 2. Emotion taxonomy, delivery style, and content context

**Research correction:** the ten labels initially proposed for YapIt (`curious`,
`awe`, `clear-explainer`, and so on) are product presets, not a published universal
emotion taxonomy. They must not be advertised as native categories trained into
every candidate. The relevant research instead distinguishes affective labels,
acoustic attributes, free-form style descriptions, and contextual intent.

| Source | What it actually uses | Consequence for YapIt |
|---|---|---|
| IndexTTS2/2.5 public emotion-vector interface | Eight ordered dimensions: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm | A directly documented bounded control interface; there is no native `travel`, `product`, `curious`, or `awe` vector dimension [17] |
| EmoV-DB and the reviewed Emotion-Aware-TTS project | Neutral, sleepiness, anger, disgust, amused | A corpus-specific set, not a complete creator taxonomy; sleepy is not the desired default for calm narration [6, 7] |
| CosyVoice3 official instruction examples | Happy, sad, angry instructions; separate very-soft/loud voice and slow/fast speech instructions, plus dialect/style examples | The public examples distinguish emotion from acoustic delivery. They are examples, not proof of arbitrary English-style generalization [35] |
| VoxCPM2 public cloning controls | Natural-language instructions for emotion, speed, and style, independent of the reference used for timbre | A free-form instruction interface rather than a public closed emotion-ID list; performance must be tested for each requested combination [36] |
| InstructTTSEval | Acoustic-Parameter Specification, Descriptive-Style Directive, and Role-Play/context; its paper organizes 12 features into several tiers | Evidence for evaluating separate low-level controls, high-level descriptions, and scenario adaptation, not for declaring a fixed set of ten emotions [33, 34] |

InstructTTSEval is an evaluation benchmark, **not a training-label specification
for every model evaluated on it**. Its expressive references include film/TV
material, which is not automatically representative of natural personal vlogs.
Its automatic evaluation uses external model judges; YapIt's private audio must
not be uploaded merely to reproduce that judge. Local listening and local metrics
remain the selection method. [33]

The 2026 Emotional-Context-Speech work further supplies conversational context,
personal experience, open-vocabulary emotion labels, and paralinguistic
descriptions, with authors reporting benefits from context-aware synthesis. It
supports the principle that isolated sentence labels omit relevant information;
it does **not** prove that adding a `travel` flag improves every shortlisted model
or establish performance on Arka's recordings. [37]

#### Proposed control model for `add_emotion`

Keep the underlying concepts separate, with friendly presets composing them:

| Layer | Proposed representation | Interpretation |
|---|---|---|
| Output identity | `profile`, default `arka` | The enrolled voice, not a filename-only label |
| Affect | `emotion_type` | Backend-independent requested emotion, mapped only to validated backend controls |
| Delivery | Conversational, storytelling, explanatory, cinematic | How the words are communicated; these are proposed product categories |
| Vocal mode | Spoken, soft-spoken, whisper | Whispering must be separately validated; it is not a synonym for calm/sleepy |
| Expressive strength | Subtle by default; moderate optional | Product-level settings, not universal numerical intensity calibration |
| Context | Optional domain, content subtype, audience/intent, pronunciation hints | Guidance for phrasing and emphasis, never permission to rewrite facts |

A practical affect vocabulary can begin with **neutral, happy, calm, sad, and
surprised**, and expose angry/fearful/disgusted/melancholic only where the backend
passes explicit capability tests. This is a **proposed scoped vocabulary** assembled
from the documented interfaces, not a claim that all five initial labels are
already available in YapIt's current Qwen Base path. `neutral` need not correspond
to a vector dimension: its backend mapping may mean no emotion override.

Keep creator presets as a separate convenience layer:

| Proposed preset | Requested intent, not a scientific equivalence | Useful context |
|---|---|---|
| `lively-vlog` | Curious, friendly exploration with varied emphasis and restrained energy | Travel experiences, discovering a place or trying a feature |
| `relaxed-travel` | Calm, warm, unhurried conversation without flattening intonation | Slow travel and relaxing commentary |
| `cinematic-landscape` | Quiet wonder, space between phrases, non-theatrical delivery | Landscape reveals and cinematic scenes |
| `reflective-journey` | Thoughtful personal storytelling, without automatically making it sad | Memories and meaningful travel experiences |
| `clear-itinerary` | Neutral/helpful explanation with clear emphasis on days, times, and logistics | Itineraries and practical travel advice |
| `energetic-review` | Credible enthusiasm with articulate feature/benefit emphasis | Positive tech, makeup, kitchen, and household reviews |
| `balanced-review` | Measured explanation of strengths and limitations; no forced positivity | Comparisons, reservations, recommendations |
| `asmr-landscape` | Calm, intimate delivery with explicitly selected soft-spoken/whisper mode | Relaxing landscape ASMR commentary |

These are **proposed presets pending listening validation**. In particular, do
not implement `awe` or curiosity by inventing untested emotion-vector mixtures.
Use an appropriate instruction/reference path and evaluate the result. Preserve
existing reference-labeled presets: their metadata alone cannot distinguish a
soft-spoken reference from an actual whispered one.

#### Is travel/product context required?

**No: an explicit domain should be optional.** This is the recommended product
design. The approved transcript already supplies vocabulary; optional context
can make intended delivery less ambiguous:

- Travel experience: conversational discovery and personal emphasis.
- Itinerary: intelligible names, numbers, durations, and practical instructions.
- Cinematic landscape: lower energy and considered pauses, without compulsory whisper.
- Product review/demo: articulate feature names and contrasts, without compulsory excitement.
- Makeup/food/household use: relatable demonstrations; domain alone must not dictate a sales voice.

Domain, emotion, and energy must not be locked together. A travel itinerary can
be brisk and clear; a product review can be calm or skeptical. When no domain is
supplied, use generic instructions rather than treating an inferred domain as an
authoritative classification. Pronunciation hints can guide approved words but
must not silently change place names, brands, specifications, or claims.

#### Agreed workflow and remaining implementation gate

The owner selected **one-step unreviewed drafts for audio-only inputs** and
**natural timing by default**. Proposed behavior:

```text
add_emotion AUDIO --emotion-type calm [--transcript PATH]
                 [--profile arka] [--context travel-itinerary]
                 [--vocal-mode spoken|soft-spoken|whisper]
```

This command is **not implemented**. Audio-only mode would run local ASR then
regenerate, clearly labeling both transcript and audio as unreviewed. Supplied
transcripts would be authoritative. Standalone Rescript continues to produce
review-required transcripts without generating audio. No automatic rewriting,
source truncation, music/ambience preservation, or exact pause reconstruction is
implied. Matching source duration would be an explicit mode with existing
stretching/correctness gates.

The `arka` default chooses the output voice. If a deliberate different profile is
selected, the system must not describe the result as preserving a different source
speaker. Source/profile consent and identity need explicit validation.

Backend feasibility remains a gate. Qwen Base cannot be given arbitrary emotion
instructions merely by adding a flag. VoxCPM2's documented controllable-cloning
mode is the first prototype route; its guide warns that Hi-Fi mode ignores the
control instruction, so the adapter must not accidentally select that mode.
Voice identity and style adherence still require local tests. [9, 36]

### 3. Released-capability and decision matrix

**Documented support is not measured quality.** "Training required" describes the path to a new speaker, not a guarantee that sufficient personal data exists.

| Candidate actually reviewed | Personal-voice path | Emotion/delivery mechanism | Role and principal tradeoff |
|---|---|---|---|
| FastSpeech 2, paper v8; community implementation | Train/adapt a speaker-capable implementation | Numeric duration/pitch/energy; emotional extension needs training | Architectural baseline, not an equivalent zero-shot product [1, 2] |
| EmotiVoice, current released repository | Own-data fine-tuning; pretrained speaker-ID inventory | Text prompt embedding | Useful explicit-style baseline; older stack, personal expressive data, unverified weight terms [3-5] |
| EMOVIE / EMSpeech | Mandarin research system, not verified English own-voice tooling | Text-predicted emotion | Disambiguation and historical research; not selected [6, 7] |
| `ionut-cmd/Emotion-Aware-TTS` | Four trained English speakers; new speaker adaptation not documented | Five trained emotion categories, manual or classifier-driven | Educational/research baseline; not arbitrary voice cloning [6, 7] |
| Qwen3-TTS 12Hz 1.7B Base | Speaker audio plus exact transcript | Reference-conditioned delivery; no native Base instruction interface | Implemented comparator; weakest independent-control fit among the shortlist [9, 10] |
| Qwen3-TTS 1.7B CustomVoice / VoiceDesign | Stock speaker or designed persona; own-speaker training is a separate route | Instructions | Not a drop-in Arka-preserving substitution [9] |
| VoxCPM2, 2B | Reference cloning or continuation with reference transcript | Style-guided cloning in released API | **Priority challenger:** requirement fit plus Apache-2.0 weights; higher memory and stochastic-style risk [13-15] |
| Fun-CosyVoice3-0.5B-2512 | Zero-shot reference cloning | Instruction plus speaker reference; vocal-event controls | **Priority alternative:** smaller released model; Windows dependencies and actual English instruction fidelity untested [21-23] |
| IndexTTS 2.5 | Separate speaker reference | Separate emotion reference/vector/text; native duration factor | Strongest explicit controls; custom license and incompatible Python pins require isolation [16-20] |
| StyleTTS 2, multispeaker | Reference adaptation or fine-tuning | Style diffusion and reference blending | Valuable naturalness baseline; greater integration/training complexity [28, 29] |
| F5-TTS v1 Base | Reference audio/text | Reference-selected styles, flow sampling | Useful reference-continuation baseline; pretrained weight restriction [30, 31] |
| Chatterbox V3 / Turbo / Nano | Reference cloning | Variant-specific controls and vocal tags | Nano is a CPU experiment, not a proven travelling solution; upstream retirement risk [25-27] |
| OmniVoice, released repository | Reference cloning; separate voice-design mode | Documented attributes and vocal-event syntax | Horizon candidate; the reviewed examples do not establish arbitrary emotion instructions combined with cloning [32] |

**Freshness checks that materially change the decision:**

- Qwen's official release announcement is January 22, 2026; its Base and instruction-capable variants are distinct. [9]
- VoxCPM2's repository announces an **April 2026** release; its technical report was submitted **June 5, 2026**. These are different events, not a contradictory release date. [14, 15]
- IndexTTS **2.5**, not 2, is the current reviewed release, announced **August 10, 2026**; the paper's latest revision is **August 11, 2026**. The older IndexTTS 2 README said precise-duration control was not enabled in that release. [16, 17]
- CosyVoice's reviewed public recommendation is the **December 2025 `2512`** base/RL family. The base checkpoint is selected first; RL is a separate ablation, not assumed better on speaker identity. [22, 23]
- Chatterbox's current repository says it became **retired, unmaintained research in August 2026**. Its model card and parts of the README retain older promotional material. Do not infer current support or a commercial service from those stale paragraphs. [25, 26]

### 3. Verified control interfaces and integration hazards

The following are abbreviated API shapes, **not code already integrated into YapIt**:

```python
# VoxCPM2: reference identity with separately supplied style guidance.
model.generate(
    text="(warm, curious, conversational delivery)" + approved_text,
    reference_wav_path=reference_audio,
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)

# CosyVoice3: the released example passes instruction and speaker recording.
model.inference_instruct2(
    approved_text,
    "You are a helpful assistant. " + instruction + "<|endofprompt|>",
    reference_audio,
    stream=False,
)

# IndexTTS2.5: speaker and emotion recordings are distinct inputs.
tts.infer(
    spk_audio_prompt=reference_audio,
    emo_audio_prompt=emotion_audio,
    emo_alpha=0.7,
    text=approved_text,
    lang="EN",
    output_path=generated_audio_path,
)
```

Sources: [14, 17, 22]. Parameter values illustrate an initial configuration, not tuned recommendations or achieved quality.

VoxCPM places instructions in a model-specific text prefix. YapIt must preserve the original `Script.beats[].text` and compile control metadata separately inside the backend. Explicitly test that instructions are **not spoken**. Compare its controllable-cloning mode against continuation mode rather than assuming both maximize identity and style simultaneously. [13, 14]

CosyVoice examples demonstrate the combined interface, including dialect/speed instructions; their existence does not prove that an English "curious vlog" instruction is followed reliably. Verify the selected checkpoint and instruction syntax locally. [22, 23]

IndexTTS 2.5's text-emotion helper is optional and requires `use_qwen_emo=True` when constructing the model. Its `duration_factor` range is 0.5-2.0, with values above 1 slowing speech. It is a relative duration control, not a guarantee of exact video alignment. `use_random=True` is documented as reducing cloning fidelity. Use deterministic conditioning first and record any extra helper-model memory. [17, 18]

Chatterbox Turbo/Nano have an especially important trap: their `generate` method accepts `exaggeration` and `cfg_weight`, but logs that these controls are unsupported and ignored. Their reference preparation also requires **more than five seconds** of audio. Thus a three-second YapIt reference cannot be reused blindly, and original Chatterbox's expressiveness guidance cannot be transferred to Nano by copying argument names. [27]

### 4. Windows, eGPU, CPU, and licensing feasibility

| Candidate | Verified upstream requirement or estimate | YapIt-specific implication |
|---|---|---|
| Qwen baseline | Upstream recommends isolated Python 3.12; current code supports CPU and CUDA selection [9] | Existing integration works; new CPU/GPU environments establish runtime portability, not model quality |
| VoxCPM2 | Package metadata requires Python >=3.10, Torch >=2.5; README guidance is narrower than that metadata. Model card estimates approximately 8 GB VRAM [13, 14] | Plausible on the 12 GB-class eGPU, **unmeasured**; isolate and resolve its Gradio/TorchCodec dependencies rather than modifying Qwen's environment |
| CosyVoice3 | Official setup uses Python 3.10; optional `ttsfrd` example wheel is Linux-specific, with a documented alternative frontend [22, 23] | Use an isolated runtime and test native Windows; do not require the optional Linux wheel or claim WSL2 is already validated |
| IndexTTS2.5 | Manifest requires Python >=3.10,<3.12, Torch 2.8, Transformers 4.52.1; model card estimates roughly 6 GB inference VRAM [18, 19] | Cannot install into YapIt's current Python 3.12/Qwen stack as-is; Python 3.11 worker is the straightforward candidate |
| Chatterbox Nano | Model card targets CPU; tested upstream environment is Python 3.11/Debian 11; public checkpoint and `nano=True` API exist [26, 27] | Worth a travelling-mode spike, but native-Windows latency, identity, and dependencies remain unmeasured |
| EmotiVoice / Emotion-Aware-TTS | Older acoustic-model toolchains and limited published deployment evidence [3-7] | Higher adaptation/setup cost; not justified before the zero-shot shortlist |

VoxCPM2's reported 4090 RTF around 0.30 and Nano's reported three-times-real-time eight-core result are **author claims on other hardware**. They are not forecasts for this eGPU or laptop. Similarly, inference memory estimates do not establish training feasibility. [13, 26]

**Proposed memory policy:** load only one synthesis model at a time and unload/offload ASR before synthesis. For scale intuition only, 2 billion BF16 parameters occupy approximately `2e9 * 2 = 4e9` bytes before decoder models, activations, caches, CUDA overhead, and display use. This lower bound is not a peak-VRAM estimate. A 12 GB-class card does not justify keeping every candidate resident.

**License gate, not legal advice:**

| Artifact | Verified status | Decision consequence |
|---|---|---|
| Qwen Base weights | Apache-2.0 model-card metadata [10] | Favorable starting point; keep required notices and review included dependencies |
| VoxCPM2 code/weights | Apache-2.0 repository/model card [13, 14] | Favorable for a maintainable backend; consent and output disclosure remain product requirements |
| CosyVoice3 `2512` weights | Apache-2.0 model-card metadata [23] | Favorable; review ancillary components independently |
| IndexTTS2.5 | Custom Bilibili model agreement, not MIT/Apache [18, 20] | Review before adoption or redistribution; includes scale thresholds, downstream obligations, and restrictions on improving other AI models |
| EmotiVoice | Apache-2.0 code; downloaded checkpoint terms not independently verified [3, 4] | Do not claim blanket commercial clearance |
| Emotion-Aware-TTS | MIT repository; checkpoint/data terms not independently established [6, 7] | Research use requires resolving actual artifact terms |
| StyleTTS2 | MIT code; additional pretrained-model consent/disclosure instructions; some inference packaging involves GPL dependencies [29] | Evaluate exact package and weights rather than calling the entire stack uniformly MIT |
| F5-TTS | MIT code, CC-BY-NC pretrained weights [31] | Commercial deployment needs a different licensed-weight strategy |
| Chatterbox Nano | MIT model-card metadata; archived research support status [25, 26] | Permissive licensing does not eliminate maintenance cost |

The IndexTTS2.5 card links an agreement whose definition still names `bilibili indextts2`. Record the actual downloaded agreement and seek clarification if redistribution or derivative-training scope matters; do not infer unrestricted permissions from a repository description. [18, 20]

### 5. Recommendation and explicit tradeoffs

**VoxCPM2 is the first technical challenger**, because the released controllable-cloning interface directly combines speaker reference and style instructions, and its weight license avoids IndexTTS's custom restrictions. The costs are a larger model, stochastic style fidelity, output-rate conversion, and an untested Windows dependency stack. Its advertised memory estimate makes a local experiment reasonable, not guaranteed. [13-15]

**CosyVoice3 `2512` is the second challenger**, because its released instruction-plus-reference path provides another way to satisfy the same requirement with a smaller model. The costs are frontend/dependency integration and uncertainty about English style adherence; do not extrapolate the 1.5B paper or streaming figures to the released 0.5B model on this machine. [21-23]

**IndexTTS2.5 is the specialist challenger** if exact control is prioritized enough to justify license review and a separate interpreter. Its emotion references/vectors and duration factor are more explicit than Qwen Base's single-reference path. A speaker-preserving English lively/relaxed result still needs verification. [16-20]

**Retain Qwen as a reversible baseline.** Compare the same enrolled voice with neutral and genuinely expressive personal references before discarding a working integration. If the expressive-reference baseline satisfies the owner, avoiding another backend is a legitimate simplicity win. It is not a reason to skip the comparison.

For CPU travel, first retain the existing CPU-capable Qwen mode with honest latency reporting. Evaluate Nano separately only if measured latency is unacceptable; model switching can change identity/style and must never be invisible. Do not promise that the GPU quality winner will also be the CPU speed winner.

### 6. Architecture and artifact model

The diagram below shows the **current implemented narration route**, not fictitious deployed candidate services. Candidate adapters are proposed in the task breakdown.

```mermaid
flowchart LR
    E["CLI or MCP entry point"] -->|"synchronous validated call"| P["Pipeline"]
    P -->|"voice and style selection"| V["Voice profiles and references"]
    V -->|"selected speaker audio and transcript"| Q["QwenSpeech"]
    P -->|"approved beat text"| Q
    Q -->|"24 kHz waveform per beat"| P
    P -->|"concatenated raw audio"| F["Media and duration gate"]
    F -->|"accepted WAV and MP3"| D["DataStore artifact folders"]
    P -->|"script and provenance"| D
    D -->|"local files for listening"| H["Owner review"]
```

Evidence: `src\speech.py:19-95`; `src\pipeline.py:29-166`; `src\voices.py:226-251`; `src\storage.py:41-171`; `src\media.py:97-131`.

**CLI/MCP entry point.** Accepts validated local inputs and style selection; calls Pipeline synchronously through Python. MCP paths are confined to configured workspace/data roots. It must not bypass missing-style or consent validation. `src\cli.py`; `src\server.py`; `src\storage.py:90-108`.

**Pipeline.** Coordinates voice selection, unchanged script text, per-beat generation, pauses, duration fitting, and provenance. Its communication is synchronous Python calls to voice selection, speech, and media helpers, followed by local-file writes. `src\pipeline.py:46-155`.

**Voice profiles and references.** JSON metadata and local recordings identify the speaker and optional human-labeled styles. Selection supplies the reference directory and transcript to QwenSpeech; registration stores raw/prepared audio plus a transcript sidecar without training. `src\voices.py:168-251`.

**QwenSpeech.** Lazily loads one model, builds the reference prompt, and returns validated/resampled NumPy audio. Python calls to Qwen are synchronous; its lock protects that backend instance, **not every independent process using the GPU**. `src\speech.py:23-95`.

**Media and duration gate.** Runs local media tools synchronously and checks fitted and encoded durations. This is a correctness barrier, not a shortcut for forcing all output into the requested runtime. `src\media.py:22-33,97-131`; `src\pipeline.py:106-118`.

**DataStore artifact folders.** Stores raw/reference recordings in `data\audio`, text artifacts in `data\transcripts`, and synthesized audio in `data\generated_audio`. New generated artifacts are flat, with `<profile>.<source>` names and numeric duplicate suffixes; operational metadata is flat under `workspace\outputs`. Existing voice/enrollment/import storage remains structured and old artifacts are not migrated. Communication is local filesystem I/O; there is no selected distributed database or broker. `src\storage.py:41-171`.

**Owner review.** The human listens to local files and judges voice, delivery, and words. Automated routing/signal checks cannot supply this judgment; the review step is a required product gate, not an existing automated service.

The ER diagram is a **logical model of existing Pydantic objects and file artifacts**, not a relational schema or claim of database constraints:

```mermaid
erDiagram
    VOICE ||--o{ STYLE_REFERENCE : registers
    VOICE ||--o{ NARRATION_JOB : selected_by
    SCRIPT ||--|{ BEAT : contains
    SCRIPT ||--o{ NARRATION_JOB : copied_into
    NARRATION_JOB ||--o{ GENERATED_AUDIO : produces
    STYLE_REFERENCE ||--|| REFERENCE_AUDIO : points_to
    NARRATION_JOB ||--o| PROVENANCE_JSON : records
    VOICE {
        string name
        string reference_audio
        string reference_text
    }
    STYLE_REFERENCE {
        string audio
        string text
        boolean consent
    }
    BEAT {
        string text
        string visual
        int pause_ms
    }
```

Evidence: `src\models.py:26-54,66-90`; `src\pipeline.py:66-90,105-155`; `src\voices.py:196-222`. Here a narration job means a scripted synthesis attempt that reached artifact creation; prompt-only jobs and earlier validation failures are outside this diagram. Failed synthesis can produce zero final audio files.

**Storage/CAP applicability.** CAP describes consistency versus availability under partitions in distributed storage. This local single-machine design selects neither a distributed database nor a message queue, so labeling its directories "CP" or "AP" would be misleading. No DB/queue comparison or CAP classification is needed. The relevant file-storage tradeoffs are durability, atomic writes, path confinement, backup, and concurrent access.

### 7. Scale, RPS, backpressure, and single points of failure

**Planning estimates, not observed usage:** one owner, one active synthesis job, ten 45-second narrations per day, and ten 30-second imported recordings per day. Using mono 24 kHz PCM16:

```text
One 45 s waveform = 45 * 24,000 * 2 = 2.16 MB
Three generated WAV versions * 10 jobs = 64.8 MB/day
Ten 30 s raw recordings = 10 * 30 * 24,000 * 2 = 14.4 MB/day
Combined illustrative growth = 79.2 MB/day ~= 2.38 GB per 30 days
```

This excludes MP3, video, caches, alternate candidate outputs, source recordings with other formats, and model checkpoints. Canonical folder locations are implemented; the volume is an assumption.

For an assumed interactive burst of **one submission per minute**:

```text
Job arrival rate lambda = 1/60 = 0.0167 requests/s
Two logical input-file reads/job ~= 0.033 reads/s
Approximately ten artifact writes/job ~= 0.167 writes/s
```

These read/write counts simplify Pipeline's operations and exclude model-loading I/O. Storage RPS is not the expected bottleneck. For output duration `D` and measured warm RTF `r`, service capacity is approximately `mu = 1/(r*D)`. At hypothetical `D=45, r=1`, utilization is `lambda/mu = 0.75`; at hypothetical `r=4`, utilization is `3`, so an unbounded backlog would grow. Neither RTF is a local result.

**Proposed backpressure:** admit one GPU synthesis process at a time; reject or explicitly wait on additional submissions with a bounded pending count. Cancel between beats, bound worker timeouts, and retain failure metadata. A shared process/GPU admission mechanism would be new work: the existing per-object Qwen lock is insufficient across CLI/MCP processes. Run ASR and synthesis sequentially on the eGPU to avoid memory contention.

| Single point of failure | Consequence | Proposed mitigation |
|---|---|---|
| Detachable eGPU/driver | Mid-job CUDA failure or unavailable GPU mode | Preflight and explicit failure; restart only with owner-selected CPU mode, not hidden fallback |
| Laptop disk | Loss of recordings, transcripts, and outputs | Back up canonical data/profile folders; preserve raw originals and checksums |
| Model download/cache | Offline travel or cold start cannot load weights | Pin and pre-cache selected weights; test offline startup |
| Single synthesis worker | Crash blocks a job | Timeouts, failed-job metadata, and bounded explicit restart |
| Owner review | No validated expressive reference or listening decision | Keep output marked unreviewed; do not substitute automated emotion scores for consent/quality approval |

## Success Criteria

### 1. Reproducible local experiment

**Prerequisite:** obtain an owner-confirmed expressive recording with its exact spoken transcript. Do not manufacture a style reference from tones, another speaker, or an unverified soundtrack. Use a common **6-10 second** high-quality speaker excerpt for the principal experiment, subject to each model's documented requirements; this also satisfies Nano's more-than-five-second requirement. Keep longer style references as a separate ablation. [9, 27]; `src\voices.py:168-251`.

Proposed primary conditions:

| Condition | Purpose |
|---|---|
| Qwen Base + default reference | Existing-model control |
| Qwen Base + matched expressive reference | Isolates the effect of reference delivery |
| VoxCPM2 + same identity reference + requested style | Tests independently instructed cloning |
| CosyVoice3 `2512` + same identity reference + requested style | Tests a second instruction-capable architecture |

Add IndexTTS2.5 after its license/runtime gate. Keep equivalent information available where interfaces differ: all backends receive the same approved text and reference recordings, but some consume exact reference transcripts and others do not. Record this asymmetry rather than pretending identical API arguments.

Use **12 held-out scripts x 3 styles x 2 reproducible runs x 4 conditions = 288 renders**. The three styles are lively-vlog, relaxed-travel, and ASMR-like soft delivery. Include the corrected marmot script, unfamiliar geographic names, numbers, punctuation, a firm wildlife-advice sentence, and multi-beat continuity cases. At an assumed average 20 seconds, this is `288 * 20 = 5,760 s = 96 min` of generated audio, not 96 minutes of compute.

The marmot fixture is useful regression data, but it is not automatically a fully verified gold transcript: the earlier lake-name transcription remains unconfirmed. Preserve "The Ladakhi Marmot" and "burrows to greet you," and have the owner verify other proper names before scoring them as ASR errors.

Begin with one script/style per backend for installation and memory feasibility. Do not launch all 288 jobs before confirming references, licenses, and basic output validity.

### 2. Controls, metrics, and analysis

**Freeze the experiment:** code/model revisions, environment lock or package inventory, reference checksums, exact text, style wording, chunk boundaries, supported seed settings, dtype/device, and decoder parameters. Pin revisions before execution. A fixed seed is not a guarantee of bit-identical output across different hardware or library versions.

Perform two distinct evaluations:

| Evaluation | What it isolates |
|---|---|
| Matched per-beat generation | Backend quality under YapIt's current chunking pattern |
| Native paragraph/context generation ablation | Whether chunking loses continuity; keep timing and spoken-text checks equally strict |

Use fixed initial decoding settings rather than giving one model unlimited cherry-picked retries. Publish failures and first-pass success rates. Tune on a separate development set; do not tune and score the same held-out scripts.

| Dimension | Proposed measurement | Interpretation limits |
|---|---|---|
| Text fidelity | WER/CER with a fixed local ASR model, plus manual proper-name verification | ASR bias and accent errors require listening; never use the uncertain source ASR output as unquestioned truth |
| Identity | Fixed speaker-embedding cosine similarity against several held-out real Arka clips; owner same-speaker judgment | Embedding scores are evaluator-dependent, not a universal identity threshold |
| Style fit | Blind human 1-5 ratings and preference comparisons | Classifier emotion accuracy alone does not establish desired narration |
| Naturalness | Separate 1-5 listening ratings | Do not conflate louder or more exaggerated with better |
| Intelligibility | Manual words and independent transcription, especially soft speech | Low energy or pitchless whisper can undermine automated metrics |
| Continuity | Boundary clicks, repeated words, abrupt pitch/energy shifts, inconsistent speaker/style | Single short demos cannot establish long-form quality |
| Duration | Raw/final length, approved-word completeness, tempo factor, per-beat timings | Native duration controls do not excuse content omission |
| Performance | Cold/warm latency, synchronized CUDA timing, RTF, process RAM, peak allocated/reserved VRAM, failures | Other vendors' GPU results are not local measurements |

The CosyVoice paper uses ASR error rates and WavLM embedding similarity, illustrating useful objective axes, not sufficient emotion-quality evidence. [21]

For listening, randomize anonymized conditions, include real-recording anchors, and loudness-match comparison copies without altering the originals or introducing clipping. LUFS is a perceptual loudness measure useful for matching playback levels. Record normalization parameters separately. Use at least three listeners where possible, including the owner for personal identity; if only the owner participates, report this explicitly as a small personal preference study.

Analyze **paired per-script/style differences**, not an average that hides catastrophic speaker drift. Report distributions, confidence intervals obtained by resampling script groups, all failure counts, and first-pass versus retry-assisted quality. Two runs on the same text are not independent evidence equivalent to two different scripts.

### 3. Proposed decision gates

These thresholds are **engineering targets to approve before testing**, not published model performance:

| Gate | Proposed acceptance |
|---|---|
| Safety/privacy | Local audio stays local; consent recorded; paths confined; generated content labeled |
| Words and controls | No omitted safety advice, invented phrases, or spoken instruction tokens; proper names reviewed |
| Signal validity | No empty/nonfinite/silent output; no unacceptable clipping or boundary artifacts |
| Speaker identity | Owner accepts every released clip as recognizably their voice; embedding distributions do not show systematic degradation versus accepted baseline |
| Style improvement | Median blinded style-fit improvement of at least 0.5/5 over default Qwen, with median naturalness >=4/5 and no material identity loss |
| Timing | Existing +/-5 s and 0.90-1.10 post-stretch gates remain; exact video synchronization, if requested, needs a separately agreed tighter tolerance |
| GPU operation | Actual CUDA synthesis completes within available VRAM; no hidden CPU fallback; record a proposed warm RTF target <=1 separately from quality gates |
| CPU travel | Offline CPU synthesis succeeds without the eGPU; report latency; real-time CPU capability requires its own measured acceptance |
| Operational quality | Missing style, invalid input, failed worker, and GPU disconnection produce explicit errors and preserved job provenance |

A candidate failing identity or text fidelity cannot win by speed or emotion ratings. If no challenger clearly improves the accepted expressive-reference Qwen baseline, retain the simpler implementation.

### 4. Results not yet available and threats to validity

The benchmark-results table is intentionally not populated. Missing evidence includes actual expressive reference recordings, native-Windows installation/synthesis for challengers, local memory/RTF, blind listener ratings, Indian-English accent retention, and CPU model quality.

Material confounders include reference recording quality, unequal conditioning interfaces, loudness, chunk boundaries, model-specific frontend pronunciation, stochastic retries, training/test overlap, and automated evaluators' accent/style biases. The present 25 passing style tests do not resolve any of those perceptual questions.

## Task Breakdown

Each item is sized as an approximately **one-to-two-week deliverable**, not a request to launch training or all candidates immediately. Arka owns approval of recordings, target delivery, and final listening judgments; engineering implements the experiment and reversible adapters.

| Work item | Dependency | Deliverable and exit condition |
|---|---|---|
| 1. Freeze reference and evaluation corpus | None | Consent-confirmed default/expressive recordings, exact transcripts, proper-name review, held-out scripts, checksums, and pre-agreed listening rubric |
| 2. Validate isolated runtimes | None; recordings needed for final synthesis smoke | Finish existing CPU/GPU baseline modes; pre-cache weights; verify offline CPU, actual CUDA synthesis, `pip check`, eGPU-unavailable behavior; resolve challenger dependency/license gates |
| 3. Prototype shortlisted adapters | 1 and 2 | VoxCPM2/CosyVoice3 experimental backends preserve script text, resample to YapIt's 24 kHz contract, retain canonical artifacts, reject invalid output, and record backend/control provenance |
| 4. Execute and analyze the benchmark | 3 | Reproducible renders and anonymized listening copies; paired analysis of identity/style/words/timing/performance; no cherry-picked demo claims |
| 5. Promote the winner reversibly | 4 | Explicit backend selection with a preserved baseline, documented CPU/GPU behavior, bounded execution, focused regression tests, and owner-approved marmot narration |
| 6. Add specialist control or adaptation only if justified | 4; Index license approval if selected | IndexTTS2.5 control ablation, or style-labeled personal fine-tuning with held-out evaluation; proceed only when a documented failure motivates the extra complexity |

**Proposed adapter boundary:** preserve `SpeechBackend.synthesize(text) -> np.ndarray` initially; style/reference settings belong to backend construction, not modifications to approved beat text. For incompatible Python environments, a persistent local subprocess per job can use bounded JSON requests and WAV artifacts under `data\generated_audio`; resample/validate in the parent. This is proposed IPC (interprocess communication), not an implemented service. Keep worker logs on stderr, impose timeouts, and avoid loading a model anew for every beat. `src\speech.py:19-20`; `src\pipeline.py:92-104`.

**No premature training:** start with released zero-shot conditioning. The VoxCPM model card's few-minute LoRA claim and StyleTTS2's reference training examples are author recipes, not measured promises for this GPU or for emotional coverage. If adaptation is necessary, collect style-balanced own-voice examples and preserve style-specific holdouts rather than adding neutral recordings and expecting missing emotions to emerge. [13, 29]

## Reference Links

Primary sources below were reviewed directly or by the bounded literature-research thread. Mutable repository and model-card links must be revision-pinned for the actual experiment. Paper versions are given where reviewed; code license, paper license, checkpoint license, and dataset license are separate artifacts.

1. Yi Ren et al. **FastSpeech 2: Fast and High-Quality End-to-End Text to Speech.** ICLR 2021; arXiv v8, August 8, 2022. [Paper](https://arxiv.org/abs/2006.04558v8); [author project](https://speechresearch.github.io/fastspeech2/).
2. ming024. **FastSpeech2 community implementation.** [Repository](https://github.com/ming024/FastSpeech2). Not Microsoft-official.
3. NetEase Youdao. **EmotiVoice.** [Repository](https://github.com/netease-youdao/EmotiVoice); [acoustic-model implementation](https://github.com/netease-youdao/EmotiVoice/blob/main/models/prompt_tts_modified/model_open_source.py); [inference](https://github.com/netease-youdao/EmotiVoice/blob/main/inference_am_vocoder_joint.py).
4. NetEase Youdao. **Voice Cloning with your personal data.** [Official wiki recipe](https://github.com/netease-youdao/EmotiVoice/wiki/Voice-Cloning-with-your-personal-data).
5. NetEase Youdao. **EmotiVoice roadmap and history.** [Roadmap](https://github.com/netease-youdao/EmotiVoice/blob/main/ROADMAP.md); [commit history](https://github.com/netease-youdao/EmotiVoice/commits/main/).
6. Suparna De, Ionut Bostan, Nishanth Sastry. **Making Social Platforms Accessible: Emotion-Aware Speech Generation with Integrated Text Analysis.** ASONAM 2024; arXiv v1, October 24, 2024. [Paper](https://arxiv.org/abs/2410.19199); [implementation](https://github.com/ionut-cmd/Emotion-Aware-TTS).
7. Emotional dataset/name disambiguation: **EMOVIE** [paper](https://arxiv.org/abs/2106.09317); **EmoV-DB** [repository](https://github.com/numediart/EmoV-DB), [OpenSLR 115](https://www.openslr.org/115/).
8. Qwen Team. **Qwen3-TTS Technical Report.** 2026. [Paper](https://arxiv.org/abs/2601.15621). Released capabilities are checked against [9], not inferred solely from the report.
9. Qwen Team. **Qwen3-TTS official README and released-model/API table.** [Repository](https://github.com/QwenLM/Qwen3-TTS); [README](https://github.com/QwenLM/Qwen3-TTS/blob/main/README.md).
10. Qwen Team. **Qwen3-TTS-12Hz-1.7B-Base model card.** [Released weights and metadata](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).
11. YapIt implementation evidence: `src\speech.py:16-95`; `src\pipeline.py:46-155`; `src\media.py:97-131`; `src\training.py:16-99`.
12. YapIt storage/style evidence: `src\storage.py:41-171`; `src\models.py:26-90`; `src\voices.py:130-251`; `tests\test_voices.py`; `tests\test_pipeline.py`; `tests\test_interfaces.py`.
13. OpenBMB. **VoxCPM2 released model card.** [Model and metadata](https://huggingface.co/openbmb/VoxCPM2), including author-reported memory/RTF and adaptation guidance.
14. OpenBMB. **VoxCPM current official repository.** [README/API/release news](https://github.com/OpenBMB/VoxCPM/blob/main/README.md); [dependency manifest](https://github.com/OpenBMB/VoxCPM/blob/main/pyproject.toml).
15. Yixuan Zhou et al. **VoxCPM2 Technical Report.** June 5, 2026, v1. [Paper](https://arxiv.org/abs/2606.06928v1); [full text](https://arxiv.org/html/2606.06928v1).
16. Yunpei Li et al. **IndexTTS 2.5 Technical Report.** August 11, 2026 revision, v5. [Paper](https://arxiv.org/abs/2601.03888v5); [full text](https://arxiv.org/html/2601.03888v5).
17. Index Speech Team. **IndexTTS current README, version 2.5 controls and release news.** [Official repository](https://github.com/index-tts/index-tts/blob/main/README.md).
18. IndexTeam. **IndexTTS-2.5 released model card.** [Checkpoint and runtime/control requirements](https://huggingface.co/IndexTeam/IndexTTS-2.5).
19. Index Speech Team. **Dependency manifest.** [Python/Torch/Transformers constraints](https://github.com/index-tts/index-tts/blob/main/pyproject.toml).
20. Bilibili. **Model Use License Agreement supplied with IndexTTS-2.5.** [Actual model license](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/main/LICENSE).
21. Zhihao Du et al. **CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training.** May 27, 2025, v2. [Paper](https://arxiv.org/abs/2505.17589v2); [full text](https://arxiv.org/html/2505.17589v2).
22. Tongyi speech team. **CosyVoice current official code and examples.** [README](https://github.com/QwenAudio/CosyVoice/blob/main/README.md); [CosyVoice3 `inference_instruct2` example](https://github.com/QwenAudio/CosyVoice/blob/main/example.py).
23. FunAudioLLM. **Fun-CosyVoice3-0.5B-2512 model card.** [Released base checkpoint, license, roadmap, and benchmark table](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512). RL checkpoint results are distinct.
24. YapIt local fixtures and measurements: `tests\data\arka.ladakhi_marmots.wav`; `tests\data\arka.ladakhi_marmots.transcript.json`; `data\generated_audio\7ab3df348947460c84dad25125e1aa14\raw.wav`. The latter is an unversioned local artifact, not a public benchmark.
25. Resemble AI. **Chatterbox current README.** [Retirement notice, V3/Turbo/Nano distinction](https://github.com/resemble-ai/chatterbox/blob/master/README.md).
26. Resemble AI. **Chatterbox Nano model card.** [Weights, MIT metadata, and author CPU claim](https://huggingface.co/ResembleAI/chatterbox-nano).
27. Resemble AI. **Turbo/Nano implementation.** [Reference-duration assertion and ignored CFG/exaggeration arguments](https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/tts_turbo.py).
28. Yinghao Aaron Li et al. **StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models.** NeurIPS 2023. [Paper](https://arxiv.org/abs/2306.07691).
29. StyleTTS2 authors. **Official README.** [Inference, adaptation, and code/pretrained-model conditions](https://github.com/yl4579/StyleTTS2/blob/main/README.md).
30. Yushen Chen et al. **F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.** 2024. [Paper](https://arxiv.org/abs/2410.06885).
31. F5-TTS authors. **Official README.** [v1 release, reference/multistyle inference, and code/weight license distinction](https://github.com/SWivid/F5-TTS/blob/main/README.md).
32. OmniVoice authors. **Official released repository.** [Cloning/design APIs and attribute controls](https://github.com/k2-fsa/OmniVoice/blob/master/README.md). Included as a horizon check, not a proven emotion-control winner.
33. Kexin Huang et al. **InstructTTSEval: Benchmarking Complex Natural-Language Instruction Following in Text-to-Speech Systems.** 2025, v1. [Paper](https://arxiv.org/html/2506.16381v1); [official benchmark repository](https://github.com/KexinHUANG19/InstructTTSEval).
34. InstructTTSEval authors. **Dataset card.** [Acoustic-Parameter Specification, Descriptive-Style Directive, and Role-Play tasks](https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval).
35. CosyVoice authors. **Official instruction examples in source.** [Emotion, softness/loudness, speed, and dialect examples](https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/utils/common.py).
36. VoxCPM authors. **Official Usage Guide.** [Controllable-cloning mode, style instructions, and Hi-Fi mode caveat](https://voxcpm.readthedocs.io/en/latest/usage_guide.html); [official cookbook](https://voxcpm.readthedocs.io/en/latest/cookbook.html).
37. Haiyang Sun et al. **Beyond Sentence-level Labels: Integrating Conversational Context and Personal Experience for Natural Emotional Expression.** Findings of ACL 2026. [Primary publication and abstract](https://aclanthology.org/2026.findings-acl.940/).


---

## November 8, 2026: model selection, configuration and single-profile specialization

**Current decision:** retain Arka's Qwen 0.6B epoch-2 as the measured GPU
baseline. Compare no-training Pocket `english_2026-10` first, then
`english_18m_2026-10` for CPU size. Keep 7M/2M as later experiments. VoxCPM2 is
the independent-style challenger, IndexTTS 2.5 the specialist-control option.
Do not resume the reviewed CosyVoice `2512` checkpoint under the new 2026+
selection rule. These decisions supersede the preceding historical shortlist;
they do not assert a cross-model listening winner.

### Verified findings and their limits

| Finding | Evidence | Decision / limitation |
|---|---|---|
| Stable Pocket SDK v4.0.1 was published October 21, 2026; its model loader supports `english_2026-10` and the dated 18M/7M/2M selectors. | [R1, R2] | Default to a dated selector; a floating alias or SDK update does not itself prove newer weights. |
| Pocket's small English models have 18.2M / 6.9M / 1.9M parameters, decoder included. | [R3] | Published sizes, not measured Windows RAM, speed or Arka quality. Start with 18M before more aggressive reduction. |
| Pocket v4.0.1 supports `load_model(quantize=True)` with dynamic INT8 transformer linear weights; optimized performance uses its optional quantization dependencies. | [R2] | This is an upstream capability. YapIt's adapter currently forwards language/config only, so there is no YapIt quantization switch yet. Decoder, activations and other allocations are not thereby all INT8. |
| Current official Qwen 0.6B Base SHA matches the recorded training Base SHA; Hub last modification is January 29, 2026. | [R4]; `workspace\outputs\arka.training.training.json` | Preserve the existing trained checkpoint. A matching Base revision does not validate its personalization quality. |
| VoxCPM2 was released April 2026; official examples use `cfg_value=2.0`, `inference_timesteps=10`, and document SFT/LoRA. | [R5] | Useful initial experiment settings, not tuned YapIt defaults. The few-minute adaptation claim is upstream, not a measured Arka result. |
| IndexTTS 2.5 was announced August 10, 2026. The reviewed CosyVoice release remains `2512`. | [R6, R7] | Index needs its existing license/runtime gate. The December 2025 CosyVoice checkpoint is excluded from new selections, regardless of recent repository edits. |
| Pocket training documentation demonstrates 200 hours for intelligibility and describes larger data requirements for production quality, depth distillation and teacher caching. | [R8] | Not evidence that 15 Arka clips can train a high-quality miniature model from scratch. Released small models are the lower-cost first route. |

No model was downloaded, loaded, quantized or trained for this documentation
update. Pocket remains mocked-unit tested only. Qwen training remains the
recorded full-weight run: 15 targets, one holdout, three epochs, batch one,
learning rate `2e-6`, continuation with new AdamW. Latest-only checkpoint cleanup
was manual, not an automatic training policy.

### Weight-size arithmetic: lower bounds, not runtime forecasts

For parameter count \(N\) and storage precision \(q\) bits:

$$
M_{\mathrm{weights}} \approx Nq/8.
$$

This is derived arithmetic, not a benchmark. Packed formats add scales,
metadata, alignment and sometimes duplicate/dequantized tensors; runtime also
needs activations, caches, codec/decoder assets and allocator workspace.

| Nominal parameter count | FP32, 4 bytes | BF16/FP16, 2 bytes | INT8, 1 byte | INT4, 0.5 byte |
|---|---|---|---|---|
| 0.6B | 2.40 GB | 1.20 GB | 0.60 GB | 0.30 GB |
| 100M | 400 MB | 200 MB | 100 MB | 50 MB |
| 18.2M | 72.8 MB | 36.4 MB | 18.2 MB | 9.1 MB |

Decimal GB/MB are used. These columns do not imply supported precision in each
SDK. Qwen's current GPU adapter selects BF16/FP16 and CPU selects FP32; Pocket
uses SDK defaults. Its selective INT8 option is not whole-model INT8.
The retained Arka checkpoint is about 2.33 GiB on disk, including exported
assets; do not equate that directory size with the 0.6B weight lower bound.

Peak memory is more usefully decomposed as:

$$
M_{\mathrm{peak}} \approx
M_{\mathrm{resident\ weights}} + M_{\mathrm{conditioning}} +
M_{\mathrm{activations}} + M_{\mathrm{cache}} + M_{\mathrm{workspace}}.
$$

One resident synthesis model and sequential ASR/TTS avoid competing allocations.
Changing chunk/token budgets can reduce some runtime terms, not the dense
parameter count. Exact-text caching improves repeated requests, not first-pass
weight footprint.

### Why one language/profile does not automatically shrink weights

An English language argument and an Arka speaker selector choose conditioning,
not a smaller tensor layout. Dense multilingual layers and codecs share
representations. Deleting language IDs, vocabulary rows or codec codebooks
without rebuilding mappings and evaluating a new architecture is not a safe
compression recipe. English travel/product speech must still cover geographic
names, brand names, numbers and Arka's accent.

Reference/exported-state conditioning avoids a full independently trained
checkpoint per profile when its output is acceptable. It cannot reproduce
arbitrary changes from full-weight personalization. Latest-selected retention
and shared immutable assets reduce disk duplication, not necessarily live RAM.

For a frozen linear weight \(W \in \mathbb R^{d_{\mathrm{out}}\times
d_{\mathrm{in}}}\), LoRA represents an update:

$$
W' = W + BA,\qquad
P_{\mathrm{adapter}} = r(d_{\mathrm{in}}+d_{\mathrm{out}}).
$$

This explains smaller trainable/exported updates. The base weight remains
resident; merging the update keeps the original dense shape. LoRA therefore is
not an inference-size reduction by itself. Its target modules, rank, precision
and training recipe must be validated per architecture. Source: [R9].

Structural pruning/distillation is separate new-model work. Prefer a released
small English model first; only investigate a teacher/student specialization
after a measured memory/quality failure. Use held-out Arka recordings and scripts
covering names, numbers, restrained enthusiasm, calm explanations and boundary
continuity. Do not tune and evaluate on the same 15 training clips.

### Configuration and promotion order

1. Preserve accepted Qwen epoch-2, English, explicit CUDA mode, existing
   BF16/SDPA/CUDA graphs and sampling defaults.
2. During an authorized comparison, use identical approved text and a clean
   consented reference with dated standard Pocket; reuse its voice state.
3. Change only the Pocket model to 18M. Accept it only if owner identity,
   naturalness, all words/names, signal validity and duration gates pass.
4. If memory still fails, assess documented quantization or smaller variants
   separately; measure cold/warm time, resident/peak memory and first-pass failures.
5. Fine-tune only for a demonstrated identity/style problem; optimize codec-data
   reuse, stage measurement and atomic export before changing the optimizer.

The 33.65s -> 10.10s identical-waveform CUDA-graph measurement is a controlled
Qwen workload, not evidence that Qwen beats Pocket. The 16 mocked backend units
passed in 0.60s; they establish selection/lifecycle behavior, not perceptual quality.
Sources: session `files\4070-cuda-graph-benchmark.json`;
`src\yapper\tests\test_backends.py`; current test output.

### Primary sources checked November 8, 2026

- [R1] Pocket latest stable release metadata:
  `https://api.github.com/repos/kyutai-labs/pocket-tts/releases/latest`.
- [R2] Pocket v4.0.1 loader, selectors and quantization option:
  `https://github.com/kyutai-labs/pocket-tts/blob/v4.0.1/pocket_tts/models/tts_model.py`.
- [R3] Pocket official model sizes:
  `https://github.com/kyutai-labs/pocket-tts/blob/main/README.md`.
- [R4] Official Qwen 0.6B Base metadata:
  `https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
- [R5] VoxCPM official release/API and fine-tuning documentation:
  `https://github.com/OpenBMB/VoxCPM/blob/main/README.md`.
- [R6] IndexTTS official release history:
  `https://github.com/index-tts/index-tts/blob/main/README.md`.
- [R7] CosyVoice official current checkpoint recommendation:
  `https://github.com/QwenAudio/CosyVoice/blob/main/README.md`.
- [R8] Pocket official training/distillation scope:
  `https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md`.
- [R9] LoRA author implementation and paper:
  `https://github.com/microsoft/LoRA`; `https://arxiv.org/abs/2106.09685`.

Mutable source links must be rechecked and revisions pinned before any actual
model experiment. Corrections to this entry must be appended, not edited into it.


---

## November 8, 2026: first authorized Pocket Arka-reference video

This entry corrects the earlier "mocked tests only / no inference" status without
rewriting it. The owner authorized a Puerto Princesa vlog and selected a
45-second shot-placeholder edit guide, not destination footage.

**Result:** Pocket SDK v4.0.1, `english_2026-10`, Windows CPU runtime, four Torch
threads, seed 42, Arka's existing reference; no Qwen checkpoint supplied to Pocket,
no training, music or cloud script call. Generated 49.55 seconds of narration
and a 50.30-second vertical guide (1080x1920, H.264, 30 fps, AAC audio), with
natural timing and `tempo_factor=1.0`. The existing Qwen worker was not restarted.

**Measured, single run after weight download:** 7.13 seconds model/reference
load; 16.99 seconds across six synthesis calls; 25.42 seconds audio renderer;
31.02 seconds full pipeline. These nested timings must not all be added together.
They are not a matched comparison against the earlier Qwen workload.

Video decoded fully without errors. The 24 kHz narration WAV was finite,
non-silent, with peak 0.2052 and zero clipped samples at the checked threshold.
A preview frame confirmed legible edit-guide labels. No listening/ASR evaluation
was performed: pronunciation, all spoken words, speaker similarity and engaging
delivery still require review.

**Access finding:** free weights are not necessarily ungated. The SDK caught
failed cloning-weight downloads and loaded its non-cloning fallback, then rejected
Arka conditioning. Direct download exposed the actual 403. Browser sign-in,
repository agreement/access approval and local token login are separate steps;
all required user actions were completed before the successful run. No stock
voice was substituted and no token was placed in chat or tracked files.

The cloned model is pinned by the SDK config to
`kyutai/pocket-tts`, revision `db04f0e653473e5c7333c69686ac92b7460b0c1d`,
`languages/english_2026-10/model.safetensors`. Its config specifies FP32 and
default temperature 0.3; those defaults were retained. Sources:
`pocket_tts\config\english_2026-10.yaml` in installed v4.0.1;
`https://github.com/kyutai-labs/pocket-tts/blob/v4.0.1/pocket_tts/config/english_2026-10.yaml`;
direct authenticated download and generated provenance.

**Script grounding:** UNESCO verifies the Palawan limestone cave system,
dramatic formations/large chambers, total underground river length of 8.2 km,
direct sea outflow and lower-river tidal influence. The script does not describe
8.2 km as the tourist boat route. Source:
`https://whc.unesco.org/en/list/652/`.

**Decision:** keep Pocket as a technically working CPU reference-cloning option,
not an approved identity/style winner. The 18M/7M/2M variants, quantization,
distillation and LoRA remain untested here.

Local evidence: `workspace\outputs\arka.puerto_princesa.pocket_2026_10.3.status.json`,
its provenance and timeline; `.repo-stats\yapper\functions.jsonl`; session
`files\puerto-princesa-pocket-result.json`,
`files\puerto-princesa-pocket-verification.json` and preview PNG.
Video: `data\generated_video\arka.puerto_princesa.pocket_2026_10.3.guide.mp4`.


---

## November 8, 2026: Pocket personal fine-tuning route

**Correction/clarification:** the official repository now includes a same-language
fine-tuning recipe, not only its large-data from-scratch quickstart. The latter's
200-hour example is not a demonstrated minimum for adapting Arka's voice.

At inspected source revision `28770c06f7deb1b65617834e60a6add7a4ff5b6d`:
- `training/configs/finetune.yaml` starts from the released 24-layer English
  teacher, keeps its text tokenizer/embedding, and accepts aligned train/validation
  JSONL manifests. `finetune_language.yaml` resets the embedding for a different
  language and is not the appropriate English-personalization starting point.
- The published path is teacher fine-tuning followed by optional six-layer
  `depth_distill.yaml`. Do not claim direct few-minute adaptation of the current
  October six-layer inference model is already validated.
- Template settings are LR `2e-5`, Muon LR `5e-4`, batch 64, accumulation 1,
  20,000 steps, warmup 1,000, EMA decay 0.999, max duration 30 seconds and
  max voice prompt 5 seconds. They are not a tuned prescription for 15 clips.
- The template selects `english_2026-04_24l`; the inspected configuration directory
  also provides the newer `english_2026-09_24l`, pinned to weight revision
  `983151f13aaeab1b13c1e5e3c2c383d49a9edf3f`. Select/validate the latest compatible
  teacher and its complete configuration before execution, not merely a date alias.
- Training is officially supported on Linux, not native Windows/Mac. The
  documented consumer setup is batch 16 with accumulation 4 at approximately
  16 GB VRAM. This does not establish a working setup on Arka's 12 GB card;
  smaller micro-batches and other memory choices need explicit feasibility checks.
- EMA `model.safetensors` is an inference-format export; resume `.pt` checkpoints
  and optimizer state have different purposes. A custom YAML referencing the
  exported weights can be selected through YapIt's Pocket inference adapter.

**Decision:** no training was requested or started by the how-to question.
Prepare clean, verified, style-balanced personal audio and independent holdouts;
compare the existing reference-cloning result first. YapIt's `voice-train`
still accepts only Qwen Base models; a Pocket training integration remains work.

Primary sources:
`https://github.com/kyutai-labs/pocket-tts/blob/28770c06f7deb1b65617834e60a6add7a4ff5b6d/training/README.md`;
`https://github.com/kyutai-labs/pocket-tts/blob/28770c06f7deb1b65617834e60a6add7a4ff5b6d/training/configs/finetune.yaml`;
`https://github.com/kyutai-labs/pocket-tts/blob/28770c06f7deb1b65617834e60a6add7a4ff5b6d/pocket_tts/config/english_2026-09_24l.yaml`;
local `src\training\trainer.py`.


---

## November 8, 2026: owner prefers the Pocket generation

After listening to the Puerto Princesa Pocket generation, Arka reported that it
was better than Qwen. This is owner preference on one actual generated clip,
not a randomized matched-text/reference comparison or a complete spoken-word,
accent and speaker-identity audit.

**Decision:** prioritize the standard `english_2026-10` Pocket reference-cloning
configuration for similar narration experiments; keep the existing Qwen epoch-2
as a reversible baseline. Do not automatically change global/worker defaults,
train or quantize because of this feedback. No preference evidence exists yet
for the 18M/7M/2M variants.

The positive reference-cloning result weakens the case for immediate fine-tuning:
first identify a reproducible identity/style failure on held-out scripts.
The separately verified teacher fine-tuning/distillation route remains available
research, not implemented training or a demonstrated fit on the 12 GB GPU.

Source: owner feedback in this session; generated
`data\generated_video\arka.puerto_princesa.pocket_2026_10.3.guide.mp4`.


---

## November 8, 2026: generic CosyVoice onboarding and real Arka-reference sample

**Decision and release policy.** The owner explicitly requested
`FunAudioLLM/CosyVoice2-0.5B`, overriding the earlier 2026+ agent-selection rule
for this model. The official roadmap dates CosyVoice2 to December 2024 and
Fun-CosyVoice3-0.5B-2512 to December 2025. No official 2026 weight release was
found in the inspected official inventory/roadmap; recent repository or card
edits are not weight releases. Requested CosyVoice2 weights were downloaded at
revision `eec1ae6c79877dbd9379285cf8789c9e0879293d`, not substituted with another
speaker/model. Existing SDK source is
`074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` with its Matcha-TTS submodule.

**Generic boundary.** A `cosyvoice` factory implements the existing
`ModelBackend` lifecycle/metadata/cache contract. It handles compatible
CosyVoice2/3 variants without changing Pipeline, Editor, worker or CLI behavior.
Other architectures still need trusted factories/plugins: a different Hub ID
cannot make incompatible model weights interchangeable. Reusable
`RuntimeProcess` and `isolated_runtime.serve` provide private bounded JSON
request/response and an SDK session with `load`/`synthesize`; audio moves through
an adapter-owned temporary WAV, not JSON/base64 or an external service.
SDK imports/loading are lazy; reference conditioning is retained across beats;
close, failure and timeout release the owned process/files. Cache identity
includes model/reference/runtime/source/transport assets. Actual component
precision is reported rather than assuming all CUDA modules use FP16.

**Provisioning and compatibility.** Dependencies were installed only into
`.venv-cosyvoice`, Python 3.10.22, through the approved feed. Qwen/Pocket
environments were not changed. Installed Torch/Torchaudio are 2.3.1+cpu,
ONNX Runtime is 1.18.0, and CUDA availability is false. This is CPU validation,
not GPU readiness. Older Whisper build tooling requires `setuptools<81`;
the manifest constraint also covered isolated build dependencies. `pip check`
exposed missing `tiktoken`; SDK import exposed Matcha's required `gdown` and
`wget`. These were added to the inference manifest; final `pip check` passed.

Wetext was initially installed from the upstream stack, but its normalizer
attempted a Modelscope asset download during SDK construction even with
`text_frontend=False`. The bounded owned attempt was stopped. Because narration
is approved supplied text and the SDK supports absent normalization, Wetext
was removed from this isolated environment/manifest. Final synthesis used the
SDK's supported no-normalizer path, without rewriting vendor code. Optional
GUI/server/training/TensorRT/vLLM dependencies were not installed. Installing
extra normalizers later can reintroduce their initialization downloads.

**Measured output, not a quality verdict.** The authorized sample says:
Puerto Princesa's underground river feels mysterious, beautiful, and thrilling;
welcome to Palawan. The actual script JSON preserves the supplied wording.
It used `data\arka\audio\arka.audio_sample_1.wav` and its exact enrolled transcript,
not Qwen's checkpoint, other clips, a stock voice, training, music or a writing
service. Output:
`data\generated_audio\arka.cosyvoice2.validation.2.mp3` and `.wav`.
Narration is 8.52 seconds at tempo factor 1.0, 24 kHz mono; all LLM/flow/HiFT
components reported FP32. Pipeline time was 138.58 seconds, including a measured
26.36-second model/reference load. Upstream per-beat progress reported 70.31 and
41.30 seconds; these nested timings must not be added to the whole pipeline
as independent costs. This slow CPU sample is not a matched Pocket/Qwen
benchmark, and no memory optimization or CUDA performance claim follows.

WAV and decoded MP3 were finite, non-silent and unclipped. WAV peak was
0.4591064453125; decoded MP3 peak was 0.8161669373512268; WAV clipped fraction
was zero. Isolated PID 14168 exited, and no managed worker state remained.
The earlier long-running Qwen worker PID 15396 had already been stopped
gracefully. No resident worker was restarted. Neither listening/ASR word
verification nor objective speaker-identity evaluation was performed; Arka
must review pronunciation, identity and delivery. CosyVoice3 selection is
supported in this same adapter but has not been executed in the new integration.
Global backend defaults remain unchanged; Pocket remains the owner's accepted
candidate for comparable travel narration.

**Validation.** 35 focused mocked CosyVoice/backend units passed in 1.10 seconds;
Ruff lint/format checks passed. Cases cover registration, conditioning reuse,
native-rate normalization, invalid/silent audio, output confinement, timeout,
failure recovery and owned-process/file cleanup. This is not a full repository
audit or training implementation.

Sources:
- Official roadmap and setup: https://github.com/FunAudioLLM/CosyVoice
- Requested model: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
- Latest reviewed family: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- Local integration: `src\yapper\backends.py`, `cosyvoice.py`,
  `cosyvoice_runtime.py`, `isolated.py`, `isolated_runtime.py`,
  `tests\test_cosyvoice.py`, `scripts\cosyvoice-runtime.requirements.txt`.
- SDK no-normalizer fallback: `models\cosyvoice_source\cosyvoice\cli\frontend.py`.
- Persistent evidence: `workspace\outputs\arka.cosyvoice2.validation.2.status.json`,
  `.provenance.json`, `.timeline.json`; session
  `files\cosyvoice2-validation-result.json`, `cosyvoice2-validation-verification.json`.


---

## November 8, 2026: KV caching versus vLLM for sequential personalized narration

Read-only inspection confirms KV caching is already present; vLLM is not
enabled in the current YapIt adapters. These are separate concepts. KV caching
reuses earlier attention keys/values during token generation. Model residency
avoids reloading weights; reference conditioning avoids re-extracting speaker
features; finished-speech caching skips exact repeated synthesis altogether.

- Qwen: installed SDK talker/code-predictor configurations default to
  `use_cache=True`; decoder updates `past_key_values`. YapIt's accelerated
  codebook loop explicitly carries `output.past_key_values` into the next step
  with `use_cache=True`. This is code/config evidence, not a new runtime trace.
- Pocket: streaming attention stores K/V in model state. YapIt retains the
  state from `get_state_for_audio_prompt`; SDK `generate_audio` defaults to
  `copy_state=True`, preserving reference-prefix state for subsequent beats.
  It does not simply reuse a previous narration's whole generated history.
- CosyVoice2: `Qwen2Encoder.forward_one_step` sets `use_cache=True` and accepts/
  returns `past_key_values`. The standard inference loop starts with
  `cache=None` and carries it across generated tokens. YapIt caches extracted
  zero-shot reference conditioning across beats, but this is not persistent
  transformer KV-prefix sharing across independent beats.

The local official CosyVoice source offers `load_vllm`, exports the speech LLM
and attaches a vLLM engine to it. Its flow model and HiFT vocoder still use
their separate SDK paths. YapIt currently passes `load_vllm=False` and
`load_trt=False`; neither optimization was benchmarked here.

**Decision:** keep SDK KV caching. For one English profile and sequential reels,
do not replace every SDK with vLLM. Prefer the accepted Pocket CPU baseline;
if CosyVoice speed warrants more work, first measure a separately authorized
plain-CUDA baseline and LLM/flow/vocoder stages, then compare a supported vLLM
configuration on identical script/reference. vLLM can help batch-one workloads
through optimized execution too; lack of concurrency does not prove no benefit.
However, serving/batching features alone do not establish a latency win here.

Official stable GPU installation requires Linux and Python 3.11-3.14, with no
native Windows support; WSL is a documented route. It is not a drop-in install
into the present Python 3.10/Torch 2.3.1 CosyVoice environment. Choose a compatible
SDK/runtime combination in a separate environment rather than blindly selecting
the newest engine or changing working Qwen/Pocket dependencies.
Automatic prefix caching reuses exact shared prompt computation; upstream
explicitly says it speeds prefill, not generation of new tokens. Whether
CosyVoice's embedding-based prompts can reuse an equivalent prefix in a chosen
engine/configuration remains unverified.

No dependency changes, inference processes, model loads, downloads or benchmarks
were started for this investigation.

Sources:
- `src\yapper\inference.py:128-154`, `pocket.py:61-74`,
  `cosyvoice_runtime.py:34-39`.
- Installed `qwen_tts\core\models\configuration_qwen3_tts.py:198-200,381-383`;
  `modeling_qwen3_tts.py:1064-1083`.
- Installed `pocket_tts\modules\attention.py:9-19,184-197`;
  `models\tts_model.py:582-588,764-765`.
- `models\cosyvoice_source\cosyvoice\llm\llm.py:242-254,504-541`;
  `cosyvoice\cli\model.py:281-289`.
- https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/
- https://docs.vllm.ai/en/stable/


---

## November 8, 2026: actual TTS inputs and CosyVoice2 prefix ordering

YapIt passes script text and prepared voice conditioning to SDKs, not manually
constructed token IDs. Pocket prepares reference state from audio; Qwen Base
and CosyVoice zero-shot cloning also use the recording's exact transcript.
SDKs encode text into text tokens and reference audio into model-specific
speech tokens/latents and speaker/acoustic features.

The inspected CosyVoice2 non-streaming LLM path concatenates prompt transcript
and new script tokens, then constructs start embedding, those text embeddings,
task embedding and reference speech-token embeddings. Thus the new script
comes BEFORE the reference speech tokens. Even with the same recording, the
speech-token block does not have an identical preceding context across
different scripts; its complete KV state cannot simply be reused as a prefix.
The reference transcript/start portion may share a prefix, subject to engine
matching rules. Extracted speaker/acoustic/reference tokens remain reusable
through the existing cached zero-shot state. This resolves the earlier
input-order uncertainty for this specific inspected CosyVoice2 path, not for
every TTS architecture or a future engine implementation.

Sources: `models\cosyvoice_source\cosyvoice\llm\llm.py:475-496`;
`cosyvoice\cli\frontend.py:170-191`; `src\yapper\speech.py:122-127,164-176`;
`src\yapper\pocket.py:61-74`. No inference or dependency changes were performed.


---

## November 8, 2026: optimization scope when Arka is shared across models

**Decision:** share canonical profile source assets, but keep derived conditioning
specific to the backend, immutable model revision/configuration and reference/
transcript/style/preprocessing identity. Different architectures or incompatible
variants cannot safely share arbitrary speaker embeddings, codec tokens or
transformer KV tensors merely because the person is the same.

First proposed enhancement is adapter-owned persistent prepared-profile caching.
Pocket already accepts a profile `pocket_voice_state` pointing to a safetensors
export; upstream documents that this avoids repeatedly encoding the recording.
Its exporter is available in the installed SDK. YapIt does not yet automatically
prepare/export/manage state caches for every model. CosyVoice currently caches
zero-shot features in memory; its SDK `save_spkinfo` writes inside the model folder,
so do not use that as a shared-model/profile persistence strategy. A future
adapter-owned profile cache should not rewrite immutable model assets. Qwen Base
currently retains the prepared clone prompt in memory for its adapter lifetime.

Exact generated-speech caching is already implemented in the optional worker,
with selected backend/model, adapter identity and exact text in the key. A shared
cache usable by one-shot generation is a proposed improvement, not implemented.
It must not serve output from a different requested model or voice configuration.
Compatible reference preprocessing can be reused, but no speed gain was measured.
Keep model-specific computation/serialization inside adapters, not core workflows.

For multiple sequential reels, an explicitly owned injected adapter can remain
loaded across jobs and then be closed by the caller. This avoids an indefinite
worker; automatic idle expiry is not implemented. Existing worker replacement
retains only one active model. Keeping all models simultaneously resident is
not justified by sharing a profile, especially on the 12 GB GPU.

**Measured upper bound:** the latest CosyVoice2 CPU pipeline took 138.58s,
including 26.36s load/reference preparation. Eliminating that WHOLE stage would
still leave 112.22s (about 1.23x maximum end-to-end speedup for that hypothetical
change). A profile-state cache removes only some of the stage, not weight
loading or subsequent speech generation. Do not promise a large decoding
speedup from sharing a speaker. A separately authorized plain-GPU and per-stage
comparison is the more important next investigation for CosyVoice latency;
vLLM/precision/quantization still require their own quality and runtime evidence.

Sources:
- `src\yapper\pocket.py:17-24,61-74`, `speech.py:122-127`;
  `cosyvoice_runtime.py` (cached zero-shot preparation);
  `worker.py:172-195,225-227`; `generation.py` (caller owns injected adapters).
- Installed `pocket_tts\models\model_state.py:1-17`.
- `models\cosyvoice_source\cosyvoice\cli\cosyvoice.py:69-78`.
- https://github.com/kyutai-labs/pocket-tts/blob/main/docs/CLI%20Commands/export_voice.md
- https://huggingface.co/docs/transformers/cache_explanation
- `workspace\outputs\arka.cosyvoice2.validation.2.status.json`;
  measured 26.36s load reported in this session.

No inference, dependency changes or new worker was started for this assessment.
