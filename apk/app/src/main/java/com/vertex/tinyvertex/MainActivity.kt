package com.vertex.tinyvertex

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.method.ScrollingMovementMethod
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var chatDisplay: TextView
    private lateinit var inputText: EditText
    private lateinit var sendButton: Button
    private var modelLoaded = false
    private val executor = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())

    companion object {
        init {
            System.loadLibrary("tinyvertex_jni")
        }
    }

    private external fun loadModel(modelPath: String): Boolean
    private external fun generateReply(prompt: String, maxTokens: Int): String
    private external fun unloadModel()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        chatDisplay = findViewById(R.id.chat_display)
        inputText = findViewById(R.id.input_text)
        sendButton = findViewById(R.id.send_button)

        chatDisplay.movementMethod = ScrollingMovementMethod()
        sendButton.isEnabled = false

        appendMessage("System", "Copying model from assets...")

        executor.execute {
            val modelFile = copyModelFromAssets()
            handler.post {
                if (modelFile != null) {
                    appendMessage("System", "Loading model...")
                    sendButton.isEnabled = false
                    executor.execute {
                        modelLoaded = loadModel(modelFile.absolutePath)
                        handler.post {
                            if (modelLoaded) {
                                appendMessage("tiny-Vertex", "hey! i'm here. talk to me.")
                                sendButton.isEnabled = true
                            } else {
                                appendMessage("System", "Failed to load model. Not enough RAM?")
                            }
                        }
                    }
                } else {
                    appendMessage("System", "Model file not found in assets.")
                }
            }
        }

        sendButton.setOnClickListener {
            val userText = inputText.text.toString().trim()
            if (userText.isNotEmpty() && modelLoaded) {
                appendMessage("You", userText)
                inputText.text.clear()
                sendButton.isEnabled = false

                val prompt = buildPrompt(userText)
                executor.execute {
                    val reply = generateReply(prompt, 40)
                    handler.post {
                        appendMessage("tiny-Vertex", cleanReply(reply))
                        sendButton.isEnabled = true
                    }
                }
            }
        }
    }

    private fun copyModelFromAssets(): File? {
        val modelName = "tinyvertex-360m-Q4_K_M.gguf"
        val outFile = File(filesDir, modelName)
        if (outFile.exists() && outFile.length() > 100_000_000) {
            return outFile
        }

        return try {
            assets.open(modelName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            outFile
        } catch (e: IOException) {
            e.printStackTrace()
            null
        }
    }

    private fun buildPrompt(userText: String): String {
        return """<|im_start|>system
you are tiny-Vertex, the user's close friend, not an assistant. text like a friend texts: 1-2 short casual sentences, warm and comforting. never lecture, never talk like a formal assistant.
<|im_end|>
<|im_start|>user
$userText<|im_end|>
<|im_start|>assistant
"""
    }

    private fun cleanReply(reply: String): String {
        var text = reply
        val mark = text.indexOf("you are tiny-Vertex")
        if (mark >= 0) {
            text = text.substring(0, mark)
        }
        return text.trim().ifEmpty { "hmm, say that again?" }
    }

    private fun appendMessage(sender: String, message: String) {
        val current = chatDisplay.text.toString()
        chatDisplay.text = "$current\n$sender: $message"
        chatDisplay.post {
            chatDisplay.scrollTo(0, chatDisplay.layout?.lineCount?.times(chatDisplay.lineHeight) ?: 0)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        unloadModel()
    }
}
