#!/bin/bash
#$ -o /mnt/matylda6/xdobos00/runs/logs/codebook-util-$JOB_NAME.$JOB_ID.out            # standard output log file
#$ -e /mnt/matylda6/xdobos00/runs/logs/codebook-util-$JOB_NAME.$JOB_ID.err            # standard error log file



if [ -z "$ALLOWED_GPUS" ]; then
    echo "Error: ALLOWED_GPUS environment variable is not set."
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$(~/scripts/free-gpus.sh "$ALLOWED_GPUS")

if [[ "$EXP_NAME" == na* ]]; then
    source /mnt/matylda6/xdobos00/nemo_final/bin/activate
    python calculate_codebook_utilization_nano.py "$EXP_NAME"
else
    source /mnt/matylda6/xdobos00/miniconda/etc/profile.d/conda.sh
    conda activate focal-pv310
fi

