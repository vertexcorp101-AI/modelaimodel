#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "llama.h"

#define LOG_TAG "TinyVertexJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static llama_sampler* g_sampler = nullptr;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_vertex_tinyvertex_MainActivity_loadModel(
    JNIEnv* env, jobject thiz, jstring modelPath) {

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("Loading model: %s", path);

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only

    g_model = llama_load_model_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!g_model) {
        LOGE("Failed to load model");
        return JNI_FALSE;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 4;

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_free_model(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    g_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1337));

    LOGI("Model loaded successfully");
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_vertex_tinyvertex_MainActivity_generateReply(
    JNIEnv* env, jobject thiz, jstring prompt, jint maxTokens) {

    if (!g_ctx || !g_sampler) {
        return env->NewStringUTF("Model not loaded");
    }

    const char* prompt_c = env->GetStringUTFChars(prompt, nullptr);
    std::string full_prompt = std::string(prompt_c);
    env->ReleaseStringUTFChars(prompt, prompt_c);

    // Tokenize prompt
    std::vector<llama_token> tokens = llama_tokenize(g_ctx, full_prompt, true, true);
    if (tokens.empty()) {
        return env->NewStringUTF("Tokenization failed");
    }

    // Clear KV cache for fresh generation
    llama_kv_cache_clear(g_ctx);
    llama_sampler_reset(g_sampler);

    // Process prompt
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == tokens.size() - 1) ? 1 : 0;
    }
    batch.n_tokens = tokens.size();

    if (llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        return env->NewStringUTF("Decode failed");
    }
    llama_batch_free(batch);

    // Generate tokens
    std::string result;
    int n_cur = tokens.size();
    int n_vocab = llama_n_vocab(g_model);

    for (int i = 0; i < maxTokens && n_cur < 1024; i++) {
        llama_token next = llama_sampler_sample(g_sampler, g_ctx, -1);
        if (llama_token_is_eog(g_model, next)) {
            break;
        }

        char buf[256];
        int len = llama_token_to_piece(g_model, next, buf, sizeof(buf), 0, false);
        if (len > 0) {
            result.append(buf, len);
        }

        llama_sampler_accept(g_sampler, next);

        llama_batch next_batch = llama_batch_init(1, 0, 1);
        next_batch.token[0] = next;
        next_batch.pos[0] = n_cur;
        next_batch.n_seq_id[0] = 1;
        next_batch.seq_id[0][0] = 0;
        next_batch.logits[0] = 1;
        next_batch.n_tokens = 1;

        if (llama_decode(g_ctx, next_batch) != 0) {
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
        n_cur++;
    }

    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_vertex_tinyvertex_MainActivity_unloadModel(
    JNIEnv* env, jobject thiz) {
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
    llama_backend_free();
}

}
