#!/usr/bin/env bash
set -e
cd "$(cd -P -- "$(dirname -- "$0")" && pwd -P)"

## Usage ##
#
#   bash export.sh -m path/to/ckpt.pth -c path/to/pointpillar.yaml
#
#   note: path is relative to this script
#

while getopts m:c: flag
do
    case "${flag}" in
        m) _CKPT=${OPTARG};;
        c) _CONFIG=${OPTARG};;
    esac
done

if [ ! -f "${_CONFIG}" ]
then
    echo "Config .yaml Does not exist at: ${_CONFIG}"
    exit
fi

if [ ! -f "${_CKPT}" ]
then
    echo "Ckpt .pth Does not exist at: ${_CKPT}"
    exit
fi

_CONFIG=$_CONFIG
export _CONFIG
echo "CONFIG: $_CONFIG";
_CKPT=$_CKPT
export _CKPT
echo "CKPT: $_CKPT";

export _UID=$(id -u) 

# Run backend
docker-compose run --rm open_pcdet_export
