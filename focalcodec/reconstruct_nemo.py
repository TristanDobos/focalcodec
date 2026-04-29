import os
import sys
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from nemo.collections.tts.models import AudioCodecModel

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

device = "cpu"

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

    nano_results_path = f"/mnt/scratch/tmp/xdobos00/validation_reconstructions/{experiment_name}"

    save_dir = Path(nano_results_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded model from: {checkpoint_path}")
    print(f"Running on device: {device}")
    print(f"Model input sample rate: {codec.sample_rate}")
    # print(f"Model output sample rate: {codec.output_sample_rate}")

    for wav_path in files_to_reconstruct:
        print()
        wav_name = os.path.basename(wav_path)
        print(f"Processing {wav_name}...")

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

        # encoded, encoded_len = codec.encode_audio(
        #     audio=audio,
        #     audio_len=audio_len,
        # )

        # tokens = codec.quantize(encoded=encoded, encoded_len=encoded_len)
        # decoded_repr, decoded_len = codec.dequantize(
        #     tokens=tokens,
        #     tokens_len=encoded_len,
        # )
        # rec_sig, rec_sig_len = codec.decode_audio(
        #     inputs=decoded_repr,
        #     input_len=decoded_len,
        # )

        # rec_sig = rec_sig[0, : rec_sig_len[0]].unsqueeze(0).cpu()



        # [B]
        # audio, audio_len = self.pad_audio(audio, audio_len)

        # [B, D, T_encoded]
        # encoded, encoded_len = self.audio_encoder(audio=audio, audio_len=audio_len)
        encoded, encoded_len = codec.encode_audio(
            audio=audio,
            audio_len=audio_len,
        )

        if codec.encoder_noise is not None:
            encoded = codec.encoder_noise(encoded)

        if codec.vector_quantizer:
            if codec.vector_quantizer_has_commit_loss:
                encoded, _, commit_loss = codec.vector_quantizer(inputs=encoded, input_len=encoded_len)
            else:
                encoded, _ = codec.vector_quantizer(inputs=encoded, input_len=encoded_len)
                commit_loss = 0.0
        else:
            commit_loss = 0.0

        last_part = wav_path.split("/")[-1]

        save_path = Path("/mnt/scratch/tmp/xdobos00/nemo_tokens/"+experiment_name+"/"+last_part+ ".pt") 


        output_dir = Path("/mnt/scratch/tmp/xdobos00/nemo_tokens") / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print("tokens are")
        print(encoded.detach().cpu())

        torch.save(
            {
                "tokens": encoded.detach().cpu(),
                "encoded_len": encoded_len.detach().cpu() if torch.is_tensor(encoded_len) else encoded_len,
            },
            save_path,
        )



        # [B, T]
        # audio_gen, _ = codec.audio_decoder(inputs=encoded, input_len=encoded_len)

        # sample_rate = 16000


        # save_path = f"{nano_results_path}/{wav_name}"
        # print(f"Saving {wav_name} into the path {save_path}")
        # # torchaudio.save(str(save_path), audio_gen, sample_rate)
        # sf.write(str(save_path), audio_gen.squeeze(0).cpu().numpy(), sample_rate)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_nemo_codec.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    reconstruct_wavs(experiment_name)


