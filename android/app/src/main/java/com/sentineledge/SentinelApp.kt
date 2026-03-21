package com.sentineledge

import android.app.Application
import android.util.Log

class SentinelApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "SentinelEdge application started")
    }

    companion object {
        const val TAG = "SentinelEdge"
    }
}
