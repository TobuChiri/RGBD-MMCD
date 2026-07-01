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
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export CUDA_VISIBLE_DEVICES
PYTHONPATH="$SCRIPT_DIR":${PYTHONPATH:-} \
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config="$CONFIG" \
    --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    --no-use_seed
