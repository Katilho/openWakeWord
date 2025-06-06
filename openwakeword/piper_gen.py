import argparse
import io
import logging
import os
import random
import time
import wave
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import requests
import soundfile as sf
import torch
from piper.download import (
    VoiceNotFoundError,
    ensure_voice_exists,
    find_voice,
    get_voices,
)
from piper.voice import PiperVoice

# This will reduce most library logs
logging.basicConfig(level=logging.WARNING)  # or logging.ERROR for even less output

# Keep your specific logger at INFO level if you want to see your own messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Option 2: More granular control - silence specific noisy libraries
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("librosa").setLevel(logging.WARNING)
logging.getLogger("piper").setLevel(logging.WARNING)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF INFO and WARNING messages


class PiperGenerator:
    MODELS_DIR = Path("models")  # Define models directory as a class attribute

    def __init__(
        self,
        models: List[str],
        extra_models_paths: Optional[list[str | Path]] = None,
    ):
        self.models: List[str] = models
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure models dir exists

        self.voices: List[PiperVoice] = self.ensure_voices_exist_and_download(
            self.models
        )

        # Carregar mais vozes extra a partir dos próprios modelos.
        for extra_model in extra_models_paths:
            voice = PiperVoice.load(
                model_path=extra_model,
                config_path=Path(f"{extra_model}.json"),
                use_cuda=torch.cuda.is_available(),
            )

            self.voices.append(voice)

    def download_tugao_voice(self):
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/pt/pt_PT/tugão/medium/"
        filenames = ["pt_PT-tugão-medium.onnx", "pt_PT-tugão-medium.onnx.json"]

        destination_dir = self.MODELS_DIR  # Use class attribute
        destination_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists

        for filename in filenames:
            file_path = destination_dir / filename  # Use Path object for joining
            url = base_url + filename

            if not file_path.exists():  # Use Path object's exists method
                print(f"Downloading from: {url}")
                response = requests.get(url)
                response.raise_for_status()
                print(f"Saving to: {file_path}")
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"✓ Successfully downloaded {filename}")
            else:
                print(f"✓ {filename} already exists, skipping download")

    def ensure_voices_exist_and_download(self, models: List[str]) -> List[PiperVoice]:
        # Download manual do modelo de voz do tugao para ultrapassar problemas de encoding da funcao de download da libraria.
        self.download_tugao_voice()

        download_dir = self.MODELS_DIR  # Use class attribute
        # download_dir.mkdir(parents=True, exist_ok=True) # Already created in __init__ or download_tugao_voice

        voices_info = get_voices(download_dir, update_voices=False)

        loaded_voices = []

        # Loop through each model and download if missing/incomplete
        for model_name in models:
            try:
                print(f"Ensuring voice exists: {model_name}")
                ensure_voice_exists(
                    model_name, [download_dir], download_dir, voices_info
                )
                print(f"✓ Voice '{model_name}' is ready.")

                model_path, config_path = find_voice(model_name, [download_dir])
                voice = PiperVoice.load(
                    model_path=model_path,
                    config_path=config_path,
                    use_cuda=torch.cuda.is_available(),
                )

                loaded_voices.append(voice)

            except VoiceNotFoundError:
                print(f"✗ Voice '{model_name}' not found in voices.json.")
            except Exception as e:
                print(f"⚠️ Error with voice '{model_name}': {e}")

        return loaded_voices

    def _resample_audio(
        self, audio_data: np.ndarray, original_samplerate: int, target_samplerate: int
    ) -> np.ndarray:
        """
        Resamples audio data to the target sample rate.
        """
        if audio_data.ndim > 1 and audio_data.shape[1] == 1:
            audio_data = audio_data[:, 0]  # Convert to mono if needed

        resampled_audio = librosa.resample(
            y=audio_data, orig_sr=original_samplerate, target_sr=target_samplerate
        )
        return resampled_audio

    def generate_samples_piper(
        self,
        texts: List[str],
        max_samples: int,
        output_dir: str,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
    ):
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        original_sample_rate = 22050
        resample_rate = 16000

        # Initialize progress display
        # my_display = display("Starting sample generation...", display_id=True)

        # for i in tqdm(range(max_samples)):
        for i in range(max_samples):
            voice = random.choice(self.voices)
            text = random.choice(texts)

            # # Update progress display
            # update_display(
            #     f"Sample {i + 1}/{max_samples} for text: {text}",
            #     display_id=my_display.display_id,
            # )

            # Controls the duration of the generated speech (larger = slower/longer)
            current_length_scale = length_scale or round(
                random.triangular(0.6, 1.8, 1.0), 3
            )

            # Controls the amount of randomness/noise in generation (affects prosody)
            current_noise_scale = noise_scale or round(
                random.triangular(0.4, 1, 0.667), 3
            )

            # Controls pitch/energy variation (often for expressive TTS)
            current_noise_w = noise_w or round(random.triangular(0.5, 1.2, 0.8), 3)

            # logger.info(f"Generating sample {i + 1}/{max_samples} for text: {text}")
            synthesize_args = {
                "length_scale": current_length_scale,
                "noise_scale": current_noise_scale,
                "noise_w": current_noise_w,
            }

            # Generate audio to memory buffer first
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, "wb") as wav_file:
                voice.synthesize(text, wav_file, **synthesize_args)

            # Read the generated audio
            audio_buffer.seek(0)
            audio_data, sample_rate = sf.read(audio_buffer, dtype="float32")

            # Resample the audio
            resampled_audio = self._resample_audio(
                audio_data, original_sample_rate, resample_rate
            )

            # Save resampled audio to file
            # Sanitize text for filename
            safe_text = (
                "".join(c if c.isalnum() or c in (" ", "_") else "" for c in text[:30])
                .rstrip()
                .replace(" ", "_")
            )
            wav_path = (
                Path(output_dir)
                / f"{str(voice.config.espeak_voice)}_{safe_text}_{time.monotonic_ns()}.wav"
            )

            sf.write(str(wav_path), resampled_audio, resample_rate, subtype="PCM_16")
            # logger.info(f"Saved resampled audio ({resample_rate}Hz) to {wav_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple Piper ONNX Sample Generator")

    parser.add_argument(
        "--texts",
        nargs="+",
        help="Text strings to convert to speech (alternative to --text-file)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        default="./samples_generated",
        type=str,
        help="Output directory for generated samples",
    )

    args = parser.parse_args()

    # More info about voices in: https://piper.ttstool.com/
    args.models = [
        "pt_PT-tugão-medium",
        # "es_MX-claude-high",
        # "it_IT-paola-medium",
        # "pt_BR-cadu-medium",
        # "pt_BR-faber-medium",
        # "ro_RO-mihai-medium",
    ]

    # Use the MODELS_DIR from PiperGenerator for consistency
    models_base_path = PiperGenerator.MODELS_DIR
    extra_models = [
        models_base_path / "pt_PT-rita.onnx",
        # models_base_path / "pt_PT-tugão-medium.onnx", # This is downloaded by ensure_voices_exist_and_download
    ]

    # Validate inputs
    if not args.texts:
        parser.error("--texts must be provided")

    # Load texts
    else:
        texts = args.texts
        logger.info(f"Using {len(texts)} provided texts")

    # Create generator and generate samples
    generator = PiperGenerator(args.models, extra_models_paths=extra_models)
    generator.generate_samples_piper(
        texts,
        args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
