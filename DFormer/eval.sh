#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPUS=${GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CONFIG=${CONFIG:-local_configs.NYUDepthv2.DFormer_Base}
CHECKPOINT=${CHECKPOINT:-checkpoints/NYUD_CD/epoch-292_miou_55.05.pth}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export CUDA_VISIBLE_DEVICES
PYTHONPATH="$SCRIPT_DIR":${PYTHONPATH:-} \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/eval.py \
    --config="$CONFIG" \
    --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --amp \
    --continue_fpath="$CHECKPOINT"
