#!/bin/sh
set -eu
cd "$(dirname "$0")"
case "$(uname -s)" in
  Darwin) cc -std=c99 -O2 -Wall -Wextra -Wno-deprecated-declarations -framework OpenCL opencl_bridge.c -o hyperl-opencl ;;
  Linux) cc -std=c99 -O2 -Wall -Wextra opencl_bridge.c -o hyperl-opencl $(pkg-config --cflags --libs OpenCL) ;;
  *) echo 'Build manually with your reviewed OpenCL headers/import library; no SDK is installed automatically.' >&2; exit 2 ;;
esac
./hyperl-opencl --probe
