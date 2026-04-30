import os
import sys
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from nemo.collections.tts.models import AudioCodecModel
import json

files_to_reconstruct22k = ["/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0024.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0029.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0034.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0039.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0044.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0049.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0054.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0025.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0030.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0035.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0040.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0045.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0050.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0055.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0026.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0031.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0036.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0041.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0046.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0051.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0056.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0027.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0032.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0037.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0042.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0047.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0052.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0028.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0033.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0038.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0043.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0048.wav",
"/mnt/matylda6/xdobos00/bp-training/gt_wavs_2205k/3853-163249-0053.wav"]

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

device = "cuda" if torch.cuda.is_available() else "cpu"

prefix_path = "/mnt/matylda6/xdobos00/NeMo-old/scripts/dataset_processing/data-100/"

json_file = "/mnt/matylda6/xdobos00/NeMo-old/scripts/dataset_processing/data-100/librispeech_16000_flat.json"
manifest_path = Path(json_file) 

def load_nemo_codec(model_path: str):
    model_path = str(model_path)

    if model_path.endswith(".nemo"):
        codec = AudioCodecModel.restore_from(model_path)
    elif model_path.endswith(".ckpt"):
        codec = AudioCodecModel.load_from_checkpoint(model_path)
    else:
        raise ValueError("model_path must point to either a .nemo or .ckpt file")

    codec = codec.to(device)
    codec.eval()
    return codec


@torch.inference_mode()
def reconstruct_wavs(experiment_name: str):
    checkpoints_path = f"/mnt/scratch/tmp/xdobos00/Nemo/exp/{experiment_name}/checkpoints"
    if not os.path.exists(checkpoints_path):
        print(f"Error: Checkpoints directory {checkpoints_path} does not exist.")
        sys.exit(1)

    root = Path(checkpoints_path)

    latest_checkpoint = max(
        (p for p in root.iterdir() if p.is_file() and p.suffix == ".ckpt"),
        key=lambda p: p.stat().st_ctime,
    )

    print("Latest checkpoint:", latest_checkpoint)

    checkpoint_path = latest_checkpoint
    print(f"Using checkpoint path: {checkpoint_path}")

    codec = load_nemo_codec(checkpoint_path)

    # nano_results_path = f"/mnt/scratch/tmp/xdobos00/validation_reconstructions/{experiment_name}"

    # save_dir = Path(nano_results_path)
    # save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded model from: {checkpoint_path}")
    print(f"Running on device: {device}")
    print(f"Model input sample rate: {codec.sample_rate}")

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            wav_path = prefix_path + item["audio_filepath"]

            wav_name = os.path.basename(wav_path)
            print(f"Processing {wav_name}...")

            last_part = wav_path.split("/")[-1]


            save_path = Path("/mnt/scratch/tmp/xdobos00/nemo_tokens_big/"+experiment_name+"/"+last_part+ ".pt") 


            if save_path.exists():
                print(f"Skipping, file already exists: {save_path}")
            else:
                sig_np, sample_rate = sf.read(wav_path, dtype="float32")
                sig = torch.from_numpy(sig_np)

                if sig.ndim == 1:
                    sig = sig.unsqueeze(0)      # [1, T]
                else:
                    sig = sig.transpose(0, 1)   # [C, T]

                if sig.shape[0] > 1:
                    sig = sig.mean(dim=0, keepdim=True)

                audio = sig.to(torch.float32).to(device)
                audio_len = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)

                encoded, encoded_len = codec.encode(
                    audio=audio,
                    audio_len=audio_len,
                )


                last_part = wav_path.split("/")[-1]

                save_path = Path("/mnt/scratch/tmp/xdobos00/nemo_tokens_big/"+experiment_name+"/"+last_part+ ".pt") 

                output_dir = Path("/mnt/scratch/tmp/xdobos00/nemo_tokens_big") / experiment_name
                output_dir.mkdir(parents=True, exist_ok=True)


                if save_path.exists():
                    print(f"Skipping, file already exists: {save_path}")
                else:
                    torch.save(
                        {
                            "tokens": encoded.detach().cpu(),
                            "encoded_len": encoded_len.detach().cpu() if torch.is_tensor(encoded_len) else encoded_len,
                        },
                        save_path,
                    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_nemo_codec.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    reconstruct_wavs(experiment_name)



# >>> data
# {'tokens': tensor([[[2043, 2043, 2043,  ..., 2611, 2620, 2611],
#          [2223, 2223, 2223,  ..., 2646, 2646, 2646],
#          [2790, 2790, 2790,  ..., 1549, 1539, 1548],
#          [1099, 1099, 1099,  ..., 2181, 1542, 1542]]], dtype=torch.int32), 'encoded_len': tensor([367], dtype=torch.int32)}


# qsub -N na_e1_v3_25hz   -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e1_v3_25hz,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh
# qsub -N na_e1_v3_6_25hz   -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e1_v3_6_25hz,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh
# qsub -N na_e2_0_3kbps   -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e2_0_3kbps,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh
# qsub -N na_e2_v3_0_15kbps   -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e2_v3_0_15kbps,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh
# qsub -N na_e3_12_5hz        -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e3_12_5hz,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh
# qsub -N na_e1_v3_12_5hz   -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e1_v3_12_5hz,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh
# qsub -N na_e1_v3_50hz   -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e1_v3_50hz,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh
# qsub -N na_e2_0_15kbps    -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e2_0_15kbps,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh
# qsub -N na_e2_0_6kbps   -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e2_0_6kbps,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh
# qsub -N na_e2_v3_0_3kbps    -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e2_v3_0_3kbps,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh
# qsub -N na_e4_48_12_5hz   -q all.q -l  gpu=1,gpu_ram=16G,ram_free=16G,mem_free=16G,matylda6=1 -v EXP_NAME=na_e4_48_12_5hz,ALLOWED_GPUS=1 /mnt/matylda6/xdobos00/bp-training/inference_metrics.sh