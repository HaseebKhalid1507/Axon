# Security policy

## Supported versions

Axon has no releases yet. Only the latest `main` is supported; fixes land there.

## Reporting a vulnerability

Please **don't** open a public issue for a security problem. Report it privately through GitHub:
**Security → Report a vulnerability** on this repository
([direct link](https://github.com/HaseebKhalid1507/Axon/security/advisories/new)).

Include what you can of:

- the commit (`git rev-parse HEAD`) and how you built and ran Axon (`axon serve` flags, `--device`);
- the input that triggers it (a request, a GGUF file, a prompt), or the smallest one you could make;
- what happens (crash, hang, wrong memory access, a bypass) and what you expected.

This is a small project maintained in spare time: expect an acknowledgement within about a week. Once a fix is
on `main` we publish an advisory crediting you, unless you'd rather not be named.

## What's in scope

- **`axon serve`**: the HTTP server and its OpenAI, Responses and ollama APIs. That covers request parsing, size
  limits, the `--api-key` check, stored-response access, the queue and slots, and anything that lets a client
  crash the server, read another client's data or use more memory than the limits allow.
- **GGUF loading**: Axon reads model files with its own parser. A malformed or hostile `.gguf` file that causes
  memory corruption, an out-of-bounds read or an unbounded allocation (rather than a clean refusal) is a bug we
  want to hear about.
- **Tokenizer and chat templates**: crashes or hangs on hostile input text.
- **The CUDA path**: out-of-bounds GPU memory access reachable from a request or a model file.

## What's out of scope

- Model behaviour: what a model says, jailbreaks, prompt injection against the model itself.
- Running `axon serve` on a public address without `--api-key`. It binds to `127.0.0.1` by default; exposing it
  without a key is a configuration choice.
- Denial of service that needs more requests than the configured limits (`--max-queue`, slots, request size) allow.
- Bugs in the NVIDIA driver, CUDA or the operating system.

Wrong output and ordinary crashes on valid input are regular bugs: please open an issue with the
[bug template](.github/ISSUE_TEMPLATE/bug.yml).
