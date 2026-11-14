#!/bin/bash
# models.sh fetch|check: the CI test models (CONTRIBUTING.md), fetched into models/ or checked against their sha256.
#   models/qwen2.5-0.5b-q4_k_m.gguf                  the test model (491 MB)
#   models/hf/Qwen2.5-0.5B-Instruct-FP8-dynamic/     the FP8 checkpoint (9 files, 877 MiB): safetensors tests, fuzzing
set -euo pipefail
FP8=models/hf/Qwen2.5-0.5B-Instruct-FP8-dynamic
SUMS="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db  models/qwen2.5-0.5b-q4_k_m.gguf
58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b  $FP8/added_tokens.json
3fae208d3ae1a0faeea0767ccb30d3d744f413fd3d74bb9c9a04c36c6f716d94  $FP8/config.json
b87d5b053ff14b30842cb40fc4f7bf08b09047fa27ed65fb942ea6943e9a895b  $FP8/generation_config.json
8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5  $FP8/merges.txt
5c804f9499421e548f50f5db893035377b2c8b807ecdef1226710ab001f3b2b0  $FP8/model.safetensors
76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd  $FP8/special_tokens_map.json
7e88129d9769a0b14b1587a7d5e829fe93ac0e1511636471fdfc0811951418e6  $FP8/tokenizer_config.json
913950e4971737031da511cdd1b410daae4566f62eb845b3975bca5a102323d8  $FP8/tokenizer.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  $FP8/vocab.json"
case ${1:-} in
fetch)
  mkdir -p "$FP8"
  get() { curl -fsSL --retry 5 -o "$2" "$1"; }
  get https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    models/qwen2.5-0.5b-q4_k_m.gguf
  for f in added_tokens.json config.json generation_config.json merges.txt model.safetensors \
           special_tokens_map.json tokenizer_config.json tokenizer.json vocab.json; do
    get "https://huggingface.co/RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic/resolve/main/$f" "$FP8/$f"
  done ;;
check) echo "$SUMS" | sha256sum -c ;;
*) echo "usage: models.sh fetch|check" >&2; exit 2 ;;
esac
