import argparse
import logging
import os
import random
import time
import wave
from pathlib import Path
from typing import List, Optional

import requests
import torch
from piper.download import (
    VoiceNotFoundError,
    ensure_voice_exists,
    find_voice,
    get_voices,
)
from piper.voice import PiperVoice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PiperGenerator:
    def __init__(
        self,
        models: List[str],
        extra_models_paths: Optional[list[str | Path]] = None,
    ):
        self.models: List[str] = models

        self.voices: List[PiperVoice] = self.validate_and_load_voices(self.models)

        # Carregar mais vozes extra a partir dos próprios modelos.
        if extra_models_paths:
            self.voices.extend(self.validate_and_load_extra_models(extra_models_paths))

    def download_tugao_voice(self):
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/pt/pt_PT/tugão/medium/pt_PT-tugão-medium"
        files_to_download = [f"{base_url}.onnx", f"{base_url}.onnx.json"]

        # Destination directory
        destination_dir = "models"
        Path(destination_dir).mkdir(parents=True, exist_ok=True)

        for url in files_to_download:
            filename = url.split("/")[-1]
            file_path = os.path.join(destination_dir, filename)

            # Check if file already exists
            if os.path.exists(file_path):
                print(f"✓ {filename} already exists, skipping download")
                continue

            print(f"Downloading from: {url}")

            # Download and save the file
            response = requests.get(url)
            response.raise_for_status()  # Raise an error if download fails

            print(f"Saving to: {file_path}")
            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"✓ Successfully downloaded {filename}")

    def validate_and_load_voices(self, models: List[str]) -> List[PiperVoice]:
        # Download manual do modelo de voz do tugao para ultrapassar problemas de encodign da funcao de download da libraria.
        self.download_tugao_voice()

        download_dir = Path("models")
        download_dir.mkdir(parents=True, exist_ok=True)

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

    def validate_and_load_extra_models(
        self, extra_models_paths: List[str]
    ) -> List[PiperVoice]:
        voices = []

        for extra_model in extra_models_paths:
            try:
                # Check if model path exists
                if not os.path.exists(extra_model):
                    print(f"⚠️ Model path does not exist: {extra_model}")
                    continue

                voice = PiperVoice.load(
                    model_path=extra_model,
                    config_path=extra_model + ".json",
                    use_cuda=torch.cuda.is_available(),
                )
                voices.append(voice)
                print(f"✓ Loaded extra model: {extra_model}")

            except Exception as e:
                print(f"⚠️ Error loading extra model '{extra_model}': {e}")

        return voices

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

        for i in range(max_samples):
            voice = random.choice(self.voices)
            text = random.choice(texts)

            # Controls the duration of the generated speech (larger = slower/longer)
            length_scale = length_scale or round(random.triangular(0.5, 2, 1.0), 3)

            # Controls the amount of randomness/noise in generation (affects prosody)
            noise_scale = noise_scale or round(random.triangular(0.3, 1.2, 0.667), 3)

            # Controls pitch/energy variation (often for expressive TTS)
            noise_w = noise_w or round(random.triangular(0.3, 1.5, 0.8), 3)

            logger.info(f"Generating sample {i + 1}/{max_samples} for text: {text}")
            synthesize_args = {
                "length_scale": length_scale,
                "noise_scale": noise_scale,
                "noise_w": noise_w,
            }

            # Save audio to file
            wav_path = (
                Path(output_dir)
                / f"{str(voice.config.espeak_voice)}_{text}_{time.monotonic_ns()}.wav"
            )
            with wave.open(str(wav_path), "wb") as wav_file:
                voice.synthesize(text, wav_file, **synthesize_args)


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

    args.models = [
        "pt_PT-tugão-medium",
        "es_ES-carlfm-x_low",
        "es_ES-davefx-medium",
        "es_ES-sharvard-medium",
        "es_MX-ald-medium",
        "es_MX-claude-high",
        "it_IT-paola-medium",
        "pt_BR-cadu-medium",
        "pt_BR-faber-medium",
        "pt_BR-jeff-medium",
        "ro_RO-mihai-medium",
        # "sl_SI-artur-medium",
    ]

    extra_models = [
        "models/pt_PT-rita.onnx",
        "models/pt_PT-tugão-medium.onnx",
    ]

    # Validate inputs
    if not args.texts:
        parser.error("--texts must be provided")

    # Load texts
    else:
        texts = args.texts
        logger.info(f"Using {len(texts)} provided texts")

    # Create generator and generate samples
    generator = PiperGenerator(
        args.models, args.output_dir, extra_models_paths=extra_models
    )
    generator.generate_samples_piper(
        texts,
        args.num_samples,
    )


if __name__ == "__main__":
    main()
