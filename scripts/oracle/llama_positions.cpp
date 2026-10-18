// llama_positions: full-vocab top-K log-probabilities at every position of token-id sequences, from libllama
// (llama.cpp 4413a19), for scripts/oracle/positions.py. No tokenizer, no sampling, no server: the ids go in
// as one batch per sequence (llama.cpp's prefill), logits for every position come out.
//
//   llama_positions MODEL SEQS.txt K LAST N_GPU_LAYERS CPU_MOE THREADS > out.txt
//   SEQS.txt: one sequence per line, comma-separated ids. Output, per sequence: "prompt ids (n): [..]", then
//   for each of the last LAST positions "position p:" and K lines "id logit logprob \"\"" (axon logits' format).
#include "llama.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s MODEL SEQS K LAST NGL CPU_MOE THREADS\n", argv[0]); return 2; }
    const int k = atoi(argv[3]), last = atoi(argv[4]), ngl = atoi(argv[5]), cpu_moe = atoi(argv[6]), th = atoi(argv[7]);
    std::vector<std::vector<llama_token>> seqs;
    std::ifstream in(argv[2]);
    for (std::string line; std::getline(in, line);) {
        std::vector<llama_token> s; std::stringstream ss(line); std::string x;
        while (std::getline(ss, x, ',')) if (!x.empty()) s.push_back(atoi(x.c_str()));
        if (!s.empty()) seqs.push_back(s);
    }
    size_t maxn = 0; for (auto & s : seqs) maxn = std::max(maxn, s.size());
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model_tensor_buft_override ov[2] = {{"\\.ffn_(up|down|gate)_exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    if (cpu_moe) mp.tensor_buft_overrides = ov;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ((maxn + 255) / 256 * 256);
    cp.n_batch = cp.n_ubatch = (uint32_t) maxn;
    cp.n_threads = cp.n_threads_batch = th;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (auto & s : seqs) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int n = (int) s.size();
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) {
            b.token[i] = s[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i >= n - last;
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        printf("prompt ids (%d): [", n);
        for (int i = 0; i < n; i++) printf(i ? ", %d" : "%d", s[i]);
        printf("]\n");
        for (int p = std::max(0, n - last); p < n; p++) {
            const float * l = llama_get_logits_ith(ctx, p);
            float mx = -INFINITY; for (int v = 0; v < nv; v++) mx = std::max(mx, l[v]);
            double z = 0; for (int v = 0; v < nv; v++) z += std::exp((double) l[v] - mx);
            const double lse = mx + std::log(z);
            std::vector<int> idx(nv); for (int v = 0; v < nv; v++) idx[v] = v;
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int c) { return l[a] > l[c]; });
            printf("position %d:\n", p);
            for (int j = 0; j < k; j++) printf("%7d  %.6f  %.6f  \"\"\n", idx[j], l[idx[j]], l[idx[j]] - lse);
        }
        llama_batch_free(b);
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
