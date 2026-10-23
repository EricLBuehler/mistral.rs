// Decode the golden prompt once, so the in-library Kolibri-1 tap hook writes the dumps.
//
// The hook lives in the graph compute, so it fires on any decode; what this tool controls is
// the SHAPE of that decode. The server splits a 6-token prompt into chunks, so the tensors it
// sees have 2 or 4 positions; one decode of 6 tokens gives the full prompt.
//
// Usage: llamacpp_decode_once <model.ggif> [n_tokens]

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "/tmp/kolibri1-llamacpp-auto.gguf";

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 4096;
    cparams.n_batch         = 2048;
    cparams.n_ubatch        = 2048;
    cparams.n_threads       = 32;
    cparams.n_threads_batch = 32;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    // The golden prompt, as token ids, so the run is token-exact against the goldens.
    const llama_token ids[6] = { 5249, 25079, 74493, 15774, 121598, 116026 };
    const int n_pos = 6;

    llama_batch batch = llama_batch_get_one((llama_token *) ids, n_pos);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    fprintf(stderr, "decoded %d tokens\n", n_pos);
    return 0;
}
