#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ts-config config.properties --model-store model_store
else
    eval "$@"
fi
# prevent docker exit
tail -f /dev/null


