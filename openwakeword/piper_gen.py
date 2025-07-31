import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import logging
import os
import random
import time
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from scipy import signal
import soundfile as sf
import torch
from piper.download import (
    VoiceNotFoundError,
    ensure_voice_exists,
    find_voice,
    get_voices,
)
from piper.voice import PiperVoice
from tqdm import tqdm

import onnxruntime as ort

# Not sure if this helps, but in some tests it could only use the GPU if I had run this line before..
print(ort.get_available_providers())

# This will reduce most library logs
logging.basicConfig(level=logging.INFO)  # or logging.ERROR for even less output

# Keep your specific logger at INFO level if you want to see your own messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF INFO and WARNING messages


class PiperGenerator:
    MODELS_DIR = Path("models")  # Define models directory as a class attribute

    # More info about voices in: https://piper.ttstool.com/
    DEFAULT_MODELS = [
        "pt_PT-tugão-medium",
        # "en_GB-cori-high",
        "es_MX-claude-high",
        "it_IT-paola-medium",
        # "pt_BR-cadu-medium",
        # "pt_BR-faber-medium",
        # "ro_RO-mihai-medium",
        "pt_PT-rita",  # Add any local models here
    ]

    def __init__(
        self,
        models: Optional[List[str]] = None,
    ):
        self.models: List[str] = models if models is not None else self.DEFAULT_MODELS
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure models dir exists

        self.voices: List[PiperVoice] = self.ensure_voices_exist_and_download(
            self.models
        )

        print(f"✅ Loaded {len(self.voices)} voice(s)")
        for i, voice in enumerate(self.voices):
            print(f"  {i}: {voice.config.espeak_voice}")

    def download_tugao_voice(self):
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/pt/pt_PT/tugão/medium/"
        filenames = ["pt_PT-tugão-medium.onnx", "pt_PT-tugão-medium.onnx.json"]

        destination_dir = self.MODELS_DIR
        destination_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists

        for filename in filenames:
            file_path = destination_dir / filename  # Use Path object for joining
            url = base_url + filename

            if not file_path.exists():
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

        download_dir = self.MODELS_DIR

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
                print(
                    f"✗ Voice '{model_name}' not found in voices.json. Checking local models..."
                )
                # Check for local model files
                local_model_path = self.MODELS_DIR / f"{model_name}.onnx"
                local_config_path = self.MODELS_DIR / f"{model_name}.onnx.json"

                if local_model_path.exists() and local_config_path.exists():
                    print(f"✓ Found local model: {local_model_path}")
                    try:
                        voice = PiperVoice.load(
                            model_path=local_model_path,
                            config_path=local_config_path,
                            use_cuda=torch.cuda.is_available(),
                        )
                        loaded_voices.append(voice)
                        print(f"✓ Successfully loaded local model '{model_name}'")
                    except Exception as e:
                        print(f"⚠️ Error loading local model '{model_name}': {e}")
                else:
                    print(f"✗ Local model files not found for '{model_name}'")
            except Exception as e:
                print(f"⚠️ Error with voice '{model_name}': {e}")

        return loaded_voices

    def _resample_audio_fast(
        self, audio_data: np.ndarray, original_samplerate: int, target_samplerate: int
    ) -> np.ndarray:
        """
        Fast resampling using scipy.signal.resample for better performance.
        """
        if audio_data.ndim > 1 and audio_data.shape[1] == 1:
            audio_data = audio_data[:, 0]  # Convert to mono if needed

        # Calculate the number of samples needed for target sample rate
        target_length = int(len(audio_data) * target_samplerate / original_samplerate)

        # Use scipy's faster resampling
        resampled_audio = signal.resample(audio_data, target_length)
        return resampled_audio.astype(np.float32)

    def _generate_single_sample(
        self,
        voice_idx: int,
        text: str,
        length_scale: float,
        noise_scale: float,
        noise_w: float,
        sample_id: int,
        output_dir: str,
        original_sample_rate: int,
        resample_rate: int,
    ) -> tuple[bool, str]:
        """Generate a single audio sample (for parallel processing)."""
        try:
            voice = self.voices[voice_idx]

            synthesize_args = {
                "length_scale": length_scale,
                "noise_scale": noise_scale,
                "noise_w": noise_w,
            }

            # Generate audio to memory buffer
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, "wb") as wav_file:
                voice.synthesize(text, wav_file, **synthesize_args)

            # Read the generated audio
            audio_buffer.seek(0)
            audio_data, sample_rate = sf.read(audio_buffer, dtype="float32")

            # Fast resampling
            resampled_audio = self._resample_audio_fast(
                audio_data, original_sample_rate, resample_rate
            )

            # Save resampled audio to file
            safe_text = (
                "".join(c if c.isalnum() or c in (" ", "_") else "" for c in text[:30])
                .rstrip()
                .replace(" ", "_")
            )
            wav_path = (
                Path(output_dir)
                / f"{str(voice.config.espeak_voice)}_{safe_text}_{sample_id}_{time.monotonic_ns()}.wav"
            )

            sf.write(str(wav_path), resampled_audio, resample_rate, subtype="PCM_16")
            return True, f"Saved audio to {wav_path}"

        except Exception as e:
            return False, f"Error generating sample {sample_id}: {str(e)}"

    def generate_samples_piper(
        self,
        texts: List[str],
        max_samples: int,
        output_dir: str,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        max_workers: Optional[int] = None,
        use_fast_resampling: bool = True,
    ):
        """
        Optimized version of sample generation with parallel processing.

        Args:
            max_workers: Number of parallel workers. If None, uses min(4, number of voices)
            use_fast_resampling: Whether to use scipy.signal.resample (faster) or librosa (higher quality)
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        original_sample_rate = 22050
        resample_rate = 16000

        # Pre-generate all random choices for better performance
        logger.info("Pre-generating random parameters...")
        random_choices = []

        for i in range(max_samples):
            voice_idx = random.randint(0, len(self.voices) - 1)
            text = random.choice(texts)

            current_length_scale = length_scale or round(
                random.triangular(0.9, 1.5, 1.2), 3
            )
            current_noise_scale = noise_scale or round(
                random.triangular(0.5, 0.9, 0.667), 3
            )
            current_noise_w = noise_w or round(random.triangular(0.6, 1, 0.8), 3)

            random_choices.append(
                {
                    "voice_idx": voice_idx,
                    "text": text,
                    "length_scale": current_length_scale,
                    "noise_scale": current_noise_scale,
                    "noise_w": current_noise_w,
                    "sample_id": i,
                }
            )

        # Determine number of workers
        if max_workers is None:
            max_workers = min(4, len(self.voices), max_samples)

        logger.info(
            f"Starting generation of {max_samples} samples using {max_workers} workers..."
        )

        # Use ThreadPoolExecutor for parallel processing
        successful_generations = 0
        failed_generations = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {}
            for choice in random_choices:
                future = executor.submit(
                    self._generate_single_sample,
                    choice["voice_idx"],
                    choice["text"],
                    choice["length_scale"],
                    choice["noise_scale"],
                    choice["noise_w"],
                    choice["sample_id"],
                    output_dir,
                    original_sample_rate,
                    resample_rate,
                )
                future_to_sample[future] = choice["sample_id"]

            # Process completed tasks with progress bar
            with tqdm(total=max_samples, desc="Generating samples") as pbar:
                for future in as_completed(future_to_sample):
                    sample_id = future_to_sample[future]
                    try:
                        success, message = future.result()
                        if success:
                            successful_generations += 1
                            logger.debug(message)
                        else:
                            failed_generations += 1
                            logger.error(message)
                    except Exception as e:
                        failed_generations += 1
                        logger.error(f"Sample {sample_id} failed with exception: {e}")

                    pbar.update(1)

        logger.info(
            f"Generation complete! Successful: {successful_generations}, Failed: {failed_generations}"
        )


def main():
    parser = argparse.ArgumentParser(description="Simple Piper ONNX Sample Generator")

    parser.add_argument(
        "--texts",
        nargs="+",
        help="Text strings to convert to speech (alternative to --text-file)",
    )
    parser.add_argument(
        "--samples", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        default="./samples_generated",
        type=str,
        help="Output directory for generated samples",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.texts:
        args.texts = ["Hey Clãriss"]
    texts = args.texts
    logger.info(f"Using {len(texts)} provided texts")

    # More info about voices in: https://piper.ttstool.com/
    default_models = [
        # "pt_PT-tugão-medium",
        # "en_GB-cori-high",
        # "es_MX-claude-high",
        # "it_IT-paola-medium",
        "pt_BR-cadu-medium",
        "pt_BR-faber-medium",
        # "ro_RO-mihai-medium",
        "pt_PT-rita",  # Local model example
    ]

    # Create generator and generate samples
    generator = PiperGenerator(models=default_models)
    generator.generate_samples_piper(
        texts,
        args.samples,
        output_dir=args.output_dir,
        length_scale=1,
        noise_scale=0.7,
        noise_w=0.6,
    )

    # current_length_scale = length_scale or round(
    #     random.triangular(0.7, 1.5, 1.0), 3
    # )
    # current_noise_scale = noise_scale or round(
    #     random.triangular(0.5, 0.9, 0.667), 3
    # )
    # current_noise_w = noise_w or round(random.triangular(0.6, 1, 0.8), 3)


if __name__ == "__main__":
    main()
