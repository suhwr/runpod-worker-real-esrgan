#!/usr/bin/env bash

echo "Worker Initiated"

if [ -d "/runpod-volume" ]; then
    echo "Syncing network volume cache..."
    [ -d "/runpod-volume/.cache" ] && rm -rf /root/.cache && ln -s /runpod-volume/.cache /root/.cache
fi

echo "Starting Serverless Handler"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=0
export CUDA_MODULE_LOADING=LAZY

if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
elif [ -f /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4:$LD_PRELOAD
elif [ -f /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD
fi

cd /workspace/runpod-worker-real-esrgan
exec python3 -u handler.py
