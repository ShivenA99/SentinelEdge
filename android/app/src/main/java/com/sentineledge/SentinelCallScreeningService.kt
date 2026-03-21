package com.sentineledge

import android.telecom.Call
import android.telecom.CallScreeningService
import android.util.Log

/**
 * Screens incoming calls for potential fraud.
 *
 * Registered in the manifest with [android.permission.BIND_SCREENING_SERVICE] so the
 * system invokes [onScreenCall] for every incoming call when the user has granted
 * SentinelEdge the default call-screening role.
 *
 * Current behaviour: logs call details and allows all calls through.  Once the
 * on-device ONNX model is integrated ([FraudClassifier]), this service will run
 * real-time inference on transcribed audio and optionally silence or reject
 * high-risk calls.
 */
class SentinelCallScreeningService : CallScreeningService() {

    override fun onScreenCall(callDetails: Call.Details) {
        val handle = callDetails.handle
        val number = handle?.schemeSpecificPart ?: "unknown"
        val presentation = callDetails.callerDisplayNamePresentation

        Log.i(SentinelApp.TAG, "Incoming call from: $number (presentation: $presentation)")

        // For now, allow all calls (no ML inference yet)
        val response = CallResponse.Builder()
            .setDisallowCall(false)
            .setRejectCall(false)
            .setSilenceCall(false)
            .setSkipNotification(false)
            .build()

        respondToCall(callDetails, response)
        Log.i(SentinelApp.TAG, "Call allowed: $number")
    }
}
