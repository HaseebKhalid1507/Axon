# Axon

The arc reactor for inference.

Make capable model inference so cheap and efficient it runs anywhere, on anything.

## Status

Day zero. Learning the problem space.

## The bottleneck

LLM inference is memory-bandwidth-bound, not compute-bound. The GPU sits idle waiting for bytes. Every technique — quantization, sparsity, speculative decoding, new architectures — is a different way of moving fewer bytes.

The model that fits in 2 GB and runs fast on a CPU changes everything.

## License

TBD
