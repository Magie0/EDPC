# EDPC: Accelerating Lossless Compression via Lightweight Probability Models and Decoupled Parallel Dataflow

Official implementation of the EDPC paper (https://arxiv.org/abs/2507.18969).

A standalone neural data compression implementation using PyTorch and arithmetic coding.

## Features

- Multi-path Byte Refinement Blocks (MBRB) for enhanced feature extraction
- Decoupled Pipeline Compression Architecture (DPCA) for accelerated processing  
- Information Flow Refinement (IFR) based on mutual information theory
- Lossless compression with competitive compression ratios

## Requirements

- PyTorch with CUDA support
- NumPy
- absl-py
- numba

## Quick Start

### Basic Usage

```bash
python PEARencodingdic.py --input_dir <input_file> --prefix <output_prefix> --gpu_id <gpu_id> --batch_size <batch_size>
```

### Example

```bash
python PEARencodingdic.py --input_dir data.bin --prefix compressed --gpu_id 0 --batch_size 8192
```

This will:
1. Compress `data.bin` into `compressed_*.compressed.combined`
2. Automatically decompress and verify the result in `decompressed_out`

### Batch Compression

For parallel compression of multiple files across multiple GPUs:

```bash
# Start all compression jobs
./run_compression.sh start

# Monitor progress
./run_compression.sh monitor  

# View results
./run_compression.sh results

# Start and monitor
./run_compression.sh all
```

## Parameters

- `--input_dir`: Path to input file to compress
- `--prefix`: Output file prefix
- `--gpu_id`: CUDA GPU ID to use
- `--batch_size`: Training batch size (default: 512, recommend: 8192)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--hidden_dim`: Model hidden dimension (default: 256)

## Output Files

- `{prefix}_*.compressed.combined`: Compressed file
- `decompressed_out`: Decompressed verification file

## Notes

- EDPC achieves up to 2.24× speedup in compression pipeline
- Multi-process encoding can achieve up to 21.73× speedup
- Optimal 2-branch MBRB balances compression effectiveness and system efficiency
- All compression is lossless with automatic integrity verification