package com.sentineledge

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

/**
 * On-device fraud classifier backed by ONNX Runtime.
 *
 * Loads a quantised MLP model from the app's assets directory and runs
 * inference on the 18-dimensional handcrafted feature vector produced by
 * [FeatureExtractor].  The output is a single float in `[0, 1]`
 * representing the probability that the call is fraudulent.
 *
 * Usage:
 * ```kotlin
 * val classifier = FraudClassifier(context)
 * classifier.load()                       // loads ONNX model from assets
 * val score = classifier.predict(features) // returns fraud probability
 * classifier.close()                      // release resources
 * ```
 *
 * @param context Application or Activity context used to access assets.
 */
class FraudClassifier(private val context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    /**
     * Load the ONNX model from the assets directory.
     *
     * @param modelPath Filename of the ONNX model inside the `assets/` folder.
     *                  Defaults to `"call_fraud_mlp.onnx"`.
     * @throws IllegalStateException if the model file cannot be read.
     */
    fun load(modelPath: String = "call_fraud_mlp.onnx") {
        val modelBytes = try {
            context.assets.open(modelPath).use { it.readBytes() }
        } catch (e: Exception) {
            throw IllegalStateException(
                "Failed to load ONNX model '$modelPath' from assets: ${e.message}", e
            )
        }
        session = env.createSession(modelBytes)
        Log.i(SentinelApp.TAG, "ONNX model loaded: $modelPath")
    }

    /**
     * Whether the model has been loaded and is ready for inference.
     */
    val isLoaded: Boolean
        get() = session != null

    /**
     * Run inference on a feature vector and return the fraud probability.
     *
     * @param features A [FloatArray] of 18 handcrafted features produced by
     *                 [FeatureExtractor.extract].
     * @return Fraud probability in `[0.0, 1.0]`.
     * @throws IllegalStateException if [load] has not been called.
     */
    fun predict(features: FloatArray): Float {
        val activeSession = session
            ?: throw IllegalStateException("Model not loaded. Call load() first.")

        val tensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(features),
            longArrayOf(1, features.size.toLong())
        )

        return try {
            val results = activeSession.run(mapOf("input" to tensor))
            val output = results[0].value

            // The model output shape depends on export format; handle both
            // [1, 1] (single-output MLP) and [1, 2] (two-class softmax).
            when (output) {
                is Array<*> -> {
                    @Suppress("UNCHECKED_CAST")
                    val matrix = output as Array<FloatArray>
                    if (matrix[0].size == 1) {
                        matrix[0][0]
                    } else {
                        // Two-class output: return probability of class 1 (fraud)
                        matrix[0][1]
                    }
                }
                is FloatArray -> output[0]
                else -> {
                    Log.w(SentinelApp.TAG, "Unexpected ONNX output type: ${output?.javaClass}")
                    0.0f
                }
            }
        } finally {
            tensor.close()
        }
    }

    /**
     * Release ONNX Runtime resources.
     *
     * The classifier should not be used after calling this method.
     */
    fun close() {
        session?.close()
        session = null
        Log.i(SentinelApp.TAG, "ONNX session closed")
    }
}
