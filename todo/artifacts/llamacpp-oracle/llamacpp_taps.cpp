// Dump the per-layer input stream of Kolibri-1 through llama.cpp's own extraction API.
//
// Uses the supported mechanism (llama_set_embeddings_layer_inp / llama_get_embeddings_layer_inp)
// rather than a private hook, so it is reproducible against upstream llama.cpp with only the
// one-line graph change that exposes the tensor.
//
// Output: layer<TAB>quantity<TAB>position<TAB>hidden_index<TAB>value
//   quantity is "residual_in" for layers 0..n_layer-1 and "l_out" for the last one, matching
//   the convention used elsewhere: the residual entering layer L is the output of block L-1.

#include "llama.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/tmp/kolibri1-llamacpp-auto.gguf";
    const char * out_path   = argc > 2 ? argv[2] : "/tmp/opencode/kolibri1-llamacpp-layerdump.tsv";
    const int    n_pos      = 6;   // the golden prompt is 6 tokens

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    fprintf(stderr, "n_layer=%d n_embd=%d\n", n_layer, n_embd);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 4096;
    cparams.n_batch    = 2048;
    cparams.n_ubatch   = 2048;
    cparams.n_threads  = 32;
    cparams.n_threads_batch = 32;
    cparams.embeddings = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    // Ask for the input stream of every layer, plus the final one.
    for (int il = 0; il <= n_layer; ++il) {
        llama_set_embeddings_layer_inp(ctx, (uint32_t) il, true);
    }

    // The golden prompt, as token ids, so the run is token-exact against the goldens.
    const llama_token ids[6] = { 5249, 25079, 74493, 15774, 121598, 116026 };

    llama_batch batch = llama_batch_get_one((llama_token *) ids, n_pos);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    FILE * f = fopen(out_path, "w");
    if (f == nullptr) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }

    for (int il = 0; il <= n_layer; ++il) {
        const float * vals = llama_get_embeddings_layer_inp(ctx, (uint32_t) il);
        if (vals == nullptr) {
            fprintf(stderr, "no data for layer %d\n", il);
            continue;
        }
        // The last entry is the final block output, i.e. l_out; the rest are layer inputs.
        const char * quantity = (il == n_layer) ? "l_out" : "residual_in";
        for (int p = 0; p < n_pos; ++p) {
            for (int h = 0; h < n_embd; ++h) {
                fprintf(f, "%d\t%s\t%d\t%d\t%.9g\n", il, quantity, p, h, (double) vals[p * n_embd + h]);
            }
        }
    }
    fclose(f);

    fprintf(stderr, "wrote %s\n", out_path);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
