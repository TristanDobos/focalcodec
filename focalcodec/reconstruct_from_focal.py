files_to_reconstruct = ["/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0024.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0029.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0034.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0039.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0044.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0049.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0054.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0025.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0030.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0035.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0040.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0045.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0050.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0055.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0026.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0031.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0036.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0041.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0046.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0051.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0056.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0027.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0032.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0037.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0042.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0047.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0052.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0028.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0033.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0038.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0043.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0048.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs/3853-163249-0053.wav"]

from build_codecs import build_cb3_codec, build_cb7_codec, load_matching_weights
import torch
import torchaudio
import json
from pathlib import Path
from codec import FocalCodec
import os
from datetime import datetime
from pathlib import Path
import sys


device = "cpu"

prefix_path = "/mnt/matylda6/xdobos00/NeMo-old/scripts/dataset_processing/data-100/"
json_file = "/mnt/matylda6/xdobos00/NeMo-old/scripts/dataset_processing/data-100/librispeech_16000_flat.json"
manifest_path = Path(json_file) 



def reconstruct_wavs(experiment_name):
    # Load your trained model here (this is just a placeholder)
    codec = FocalCodec.from_pretrained(f"lucadellalib/focalcodec_50hz")

    allowed_experiments = ["fo_e1_50hz", "fo_e1_25hz", "fo_e1_12_5hz", "fo_e2_cb13", "fo_e2_cb7", "fo_e2_cb3"]
    if experiment_name not in allowed_experiments:
        print(f"Error: experiment_name must be one of {allowed_experiments}")
        sys.exit(1)
    

    if experiment_name.startswith("fo_e2") or experiment_name == "fo_e1_50hz":
        base_path = "/mnt/scratch/tmp/xdobos00/focal_results/save/" + experiment_name + "/"

        print(f"Looking for checkpoints in {base_path}...")

        root = Path(base_path)


        latest_experiment = max(
            (p for p in root.iterdir() if p.is_dir() and p.name.startswith("CKPT")),
            key=lambda p: p.stat().st_ctime,
        )

        bsq_path = os.path.join(base_path, latest_experiment.name)
    
    if experiment_name == "fo_e2_cb7":
        codec = build_cb7_codec()

        pretrained = FocalCodec.from_pretrained("lucadellalib/focalcodec_50hz")

        missing_comp = load_matching_weights(codec.compressor, pretrained.compressor.state_dict())
        missing_decomp = load_matching_weights(codec.decompressor, pretrained.decompressor.state_dict())
        missing_enc = load_matching_weights(codec.encoder, pretrained.encoder.state_dict())
        missing_dec = load_matching_weights(codec.decoder, pretrained.decoder.state_dict())

        print("Skipped compressor keys:", missing_comp)
        print("Skipped decompressor keys:", missing_decomp)
    elif experiment_name == "fo_e2_cb3":
        codec = build_cb3_codec()

        pretrained = FocalCodec.from_pretrained("lucadellalib/focalcodec_50hz")

        missing_comp = load_matching_weights(codec.compressor, pretrained.compressor.state_dict())
        missing_decomp = load_matching_weights(codec.decompressor, pretrained.decompressor.state_dict())
        missing_enc = load_matching_weights(codec.encoder, pretrained.encoder.state_dict())
        missing_dec = load_matching_weights(codec.decoder, pretrained.decoder.state_dict())

        print("Skipped compressor keys:", missing_comp)
        print("Skipped decompressor keys:", missing_decomp)
    elif experiment_name == "fo_e1_50hz" or experiment_name == "fo_e2_cb13":
        codec = FocalCodec.from_pretrained("lucadellalib/focalcodec_50hz")
    elif experiment_name == "fo_e1_25hz":
        codec = FocalCodec.from_pretrained("lucadellalib/focalcodec_25hz")
    elif experiment_name == "fo_e1_12_5hz":
        codec = FocalCodec.from_pretrained("lucadellalib/focalcodec_12_5hz")


    if experiment_name in ["fo_e2_cb13", "fo_e2_cb7", "fo_e2_cb3"]:
        print("the bsq path is:", bsq_path)
        compressor_path = os.path.join(bsq_path, "compressor.ckpt")
        quantizer_path = os.path.join(bsq_path, "quantizer.ckpt")
        decompressor_path = os.path.join(bsq_path, "decompressor.ckpt")



        codec.compressor.load_state_dict(torch.load(compressor_path, map_location="cpu"), strict=False)
        codec.quantizer.load_state_dict(torch.load(quantizer_path, map_location="cpu"), strict=False)
        codec.decompressor.load_state_dict(torch.load(decompressor_path, map_location="cpu"), strict=False)


    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue

            item = json.loads(line)
            wav_path = prefix_path + item["audio_filepath"]
            print()
            sig, sample_rate = torchaudio.load(wav_path)

            wav_name = os.path.basename(wav_path)
            print(f"Processing {wav_name}...")

            # Resample for encoding
            input_sig = torchaudio.functional.resample(sig, sample_rate, codec.sample_rate_input).to(device)

            with torch.no_grad():
                toks = codec.sig_to_toks(input_sig)

                save_path = "/mnt/scratch/tmp/xdobos00/nemo_tokens_big/" + experiment_name + "/" + wav_name

                print(f"Saving {wav_name} into the path {save_path}")

                save_dir = f"/mnt/scratch/tmp/xdobos00/nemo_tokens_big/{experiment_name}"
                os.makedirs(save_dir, exist_ok=True)

                # print("toks are")
                # print(toks)
                
                # torch.save(
                #     {
                #         "tokens": toks,
                #     },
                #     save_path,
                # )
                print("Inference test complete. Saved reconstruction.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_focal.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    reconstruct_wavs(experiment_name)
