#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "llama.h"

#define LOG_TAG "TinyVertexJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model*   g_model = nullptr;
static llama_context* g_ctx   = nullptr;
static llama_sampler* g_smp   = nullptr;
static const llama_vocab* g_vocab = nullptr;

// Tokenize `text` into `out`, growing the buffer once if needed.
// Returns false on failure.
static bool tokenize_text(const std::string& text,
                          std::vector<llama_token>& out,
                          bool add_special) {
    const int32_t n_text = (int32_t)text.size();
    out.assign(text.size() + 8, 0);

    int32_t n = llama_tokenize(g_vocab, text.c_str(), n_text,
                               out.data(), (int32_t)out.size(),
                               add_special, false);
    if (n < 0) {
        // Not enough room for the tokens plus a possible BOS.
        out.resize(-n + 8);
        n = llama_tokenize(g_vocab, text.c_str(), n_text,
                           out.data(), (int32_t)out.size(),
                           add_special, false);
    }
    if (n < 0) {
        LOGE("tokenize failed: %d", n);
        return false;
    }
    out.resize((size_t)n);
    return true;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_vertex_tinyvertex_MainActivity_loadModel(
    JNIEnv* env, jobject /*thiz*/, jstring modelPath) {

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("Loading model: %s", path);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only

    g_model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!g_model) {
        LOGE("Failed to load model");
        return JNI_FALSE;
    }

    g_vocab = llama_model_get_vocab(g_model);
    if (!g_vocab) {
        LOGE("Failed to get vocab");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx          = 1024;
    ctx_params.n_batch        = 512;
    ctx_params.n_threads      = 4;
    ctx_params.n_threads_batch = 4;

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        g_vocab  = nullptr;
        return JNI_FALSE;
    }

    g_smp = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(g_smp, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(g_smp, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(g_smp, llama_sampler_init_dist(1337));

    LOGI("Model loaded successfully");
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_vertex_tinyvertex_MainActivity_generateReply(
    JNIEnv* env, jobject /*thiz*/, jstring prompt, jint maxTokens) {

    if (!g_ctx || !g_smp || !g_vocab) {
        return env->NewStringUTF("Model not loaded");
    }

    const char* prompt_c = env->GetStringUTFChars(prompt, nullptr);
    std::string full_prompt(prompt_c);
    env->ReleaseStringUTFChars(prompt, prompt_c);

    std::vector<llama_token> tokens;
    if (!tokenize_text(full_prompt, tokens, true) || tokens.empty()) {
        return env->NewStringUTF("Tokenization failed");
    }

    // Fresh generation for every turn.
    llama_memory_clear(llama_get_memory(g_ctx), false);
    llama_sampler_reset(g_smp);

    // Fit the whole prompt inside the context window.
    const int32_t n_ctx = (int32_t)llama_n_ctx(g_ctx);
    if ((int32_t)tokens.size() >= n_ctx - 8) {
        return env->NewStringUTF("Prompt too long for context window");
    }

    // Feed the prompt in chunks of at most n_batch tokens.
    const int32_t n_batch = 512;
    for (size_t start = 0; start < tokens.size(); start += (size_t)n_batch) {
        size_t end = start + (size_t)n_batch;
        if (end > tokens.size()) end = tokens.size();
        const bool is_last = (end == tokens.size());

        llama_batch batch = llama_batch_init((int32_t)(end - start), 0, 1);
        for (size_t i = start; i < end; i++) {
            const size_t k = i - start;
            batch.token[k]    = tokens[i];
            batch.pos[k]      = (llama_pos)i;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0]= 0;
            batch.logits[k]   = is_last ? 1 : 0;
        }
        batch.n_tokens = (int32_t)(end - start);

        const int rc = llama_decode(g_ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            LOGE("llama_decode failed: %d", rc);
            return env->NewStringUTF("Decode failed");
        }
    }

    // Generate tokens one at a time.
    std::string result;
    int32_t n_cur = (int32_t)tokens.size();

    for (int i = 0; i < maxTokens && n_cur < n_ctx; i++) {
        const llama_token next = llama_sampler_sample(g_smp, g_ctx, -1);
        if (llama_vocab_is_eog(g_vocab, next)) {
            break;
        }
        llama_sampler_accept(g_smp, next);

        char buf[256];
        const int len = llama_token_to_piece(g_vocab, next, buf, (int32_t)sizeof(buf), 0, false);
        if (len > 0) {
            result.append(buf, (size_t)len);
        }

        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0]     = next;
        batch.pos[0]       = n_cur;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        batch.n_tokens     = 1;

        const int rc = llama_decode(g_ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            LOGE("decode(next) failed: %d", rc);
            break;
        }
        n_cur++;
    }

    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_vertex_tinyvertex_MainActivity_unloadModel(
    JNIEnv* /*env*/, jobject /*thiz*/) {
    if (g_smp) {
        llama_sampler_free(g_smp);
        g_smp = nullptr;
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_vocab = nullptr;
    llama_backend_free();
}

}
