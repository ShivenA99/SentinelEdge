package com.sentineledge

import android.content.Context
import android.graphics.PixelFormat
import android.os.Build
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.filled.Shield
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.setViewTreeLifecycleOwner
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.savedstate.SavedStateRegistryOwner
import androidx.savedstate.SavedStateRegistryController

/**
 * Fraud alert overlay drawn on top of the active call screen.
 *
 * Uses [WindowManager] with [WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY]
 * to display a Compose-based alert banner regardless of which app is in the
 * foreground.  Requires [Settings.ACTION_MANAGE_OVERLAY_PERMISSION].
 *
 * Three severity tiers:
 *  - **MEDIUM** (0.50–0.75): compact amber banner at the top
 *  - **HIGH** (0.75–0.90): larger amber overlay with reasons
 *  - **CRITICAL** (>0.90): full red overlay with pulsing border
 *
 * Usage from [SentinelCallScreeningService]:
 * ```kotlin
 * val overlay = AlertOverlay(applicationContext)
 * overlay.show(score = 0.87f, reasons = listOf("IRS impersonation", "Urgency"))
 * // user taps Block:
 * overlay.onBlockAction = { /* reject call */ }
 * overlay.dismiss()
 * ```
 */
class AlertOverlay(private val context: Context) {

    /**
     * Risk level derived from the fraud score.
     */
    enum class RiskLevel(val label: String, val threshold: Float) {
        CRITICAL("SCAM DETECTED", 0.90f),
        HIGH("HIGH RISK CALL", 0.75f),
        MEDIUM("Suspicious Activity", 0.50f),
        SAFE("Safe", 0.0f);

        companion object {
            fun from(score: Float): RiskLevel = when {
                score >= CRITICAL.threshold -> CRITICAL
                score >= HIGH.threshold     -> HIGH
                score >= MEDIUM.threshold   -> MEDIUM
                else                        -> SAFE
            }
        }
    }

    // -- State --
    private val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private var overlayView: ComposeView? = null
    private var isShowing = false

    // Mutable state observed by the Compose UI
    private var score by mutableStateOf(0f)
    private var riskLevel by mutableStateOf(RiskLevel.SAFE)
    private var reasons by mutableStateOf(listOf<String>())
    private var visible by mutableStateOf(false)

    /** Called when the user taps "Block Caller". */
    var onBlockAction: (() -> Unit)? = null

    /** Called when the user taps "Dismiss". */
    var onDismissAction: (() -> Unit)? = null

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Check whether the app has permission to draw overlays.
     */
    fun canDrawOverlay(): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.M || Settings.canDrawOverlays(context)

    /**
     * Show or update the fraud alert overlay.
     *
     * @param score  Fraud probability in `[0, 1]`.
     * @param reasons Human-readable explanations for the alert.
     */
    fun show(score: Float, reasons: List<String> = emptyList()) {
        if (!canDrawOverlay()) {
            Log.w(SentinelApp.TAG, "Cannot show overlay: SYSTEM_ALERT_WINDOW not granted")
            return
        }

        this.score = score
        this.riskLevel = RiskLevel.from(score)
        this.reasons = reasons

        if (riskLevel == RiskLevel.SAFE) {
            dismiss()
            return
        }

        if (!isShowing) {
            createOverlayView()
        }

        visible = true
        Log.i(SentinelApp.TAG, "Alert overlay shown: ${riskLevel.label} (${(score * 100).toInt()}%)")
    }

    /**
     * Dismiss the overlay and release window resources.
     */
    fun dismiss() {
        visible = false
        if (isShowing && overlayView != null) {
            try {
                windowManager.removeView(overlayView)
            } catch (e: Exception) {
                Log.w(SentinelApp.TAG, "Failed to remove overlay: ${e.message}")
            }
            overlayView = null
            isShowing = false
            Log.i(SentinelApp.TAG, "Alert overlay dismissed")
        }
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    private fun createOverlayView() {
        val lifecycleOwner = OverlayLifecycleOwner()
        lifecycleOwner.handleLifecycleEvent(Lifecycle.Event.ON_CREATE)
        lifecycleOwner.handleLifecycleEvent(Lifecycle.Event.ON_START)
        lifecycleOwner.handleLifecycleEvent(Lifecycle.Event.ON_RESUME)

        val view = ComposeView(context).apply {
            setViewTreeLifecycleOwner(lifecycleOwner)
            setViewTreeSavedStateRegistryOwner(lifecycleOwner)

            setContent {
                MaterialTheme {
                    AlertOverlayContent(
                        visible = visible,
                        riskLevel = riskLevel,
                        score = score,
                        reasons = reasons,
                        onBlock = {
                            onBlockAction?.invoke()
                            dismiss()
                        },
                        onDismiss = {
                            onDismissAction?.invoke()
                            dismiss()
                        },
                    )
                }
            }
        }

        val layoutType = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            layoutType,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
        }

        windowManager.addView(view, params)
        overlayView = view
        isShowing = true
    }
}

// ---------------------------------------------------------------------------
// Compose UI for the overlay
// ---------------------------------------------------------------------------

@Composable
private fun AlertOverlayContent(
    visible: Boolean,
    riskLevel: AlertOverlay.RiskLevel,
    score: Float,
    reasons: List<String>,
    onBlock: () -> Unit,
    onDismiss: () -> Unit,
) {
    AnimatedVisibility(
        visible = visible,
        enter = slideInVertically(initialOffsetY = { -it }) + fadeIn(),
        exit = slideOutVertically(targetOffsetY = { -it }) + fadeOut(),
    ) {
        when (riskLevel) {
            AlertOverlay.RiskLevel.CRITICAL -> CriticalAlert(score, reasons, onBlock, onDismiss)
            AlertOverlay.RiskLevel.HIGH     -> HighAlert(score, reasons, onBlock, onDismiss)
            AlertOverlay.RiskLevel.MEDIUM   -> MediumAlert(score, onDismiss)
            AlertOverlay.RiskLevel.SAFE     -> { /* nothing */ }
        }
    }
}

/**
 * Full red overlay with pulsing border for critical threats (>90%).
 */
@Composable
private fun CriticalAlert(
    score: Float,
    reasons: List<String>,
    onBlock: () -> Unit,
    onDismiss: () -> Unit,
) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.7f, targetValue = 1.0f,
        animationSpec = infiniteRepeatable(tween(600), RepeatMode.Reverse),
        label = "pulseAlpha",
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
            .alpha(pulseAlpha)
            .border(3.dp, Color(0xFFEF4444), RoundedCornerShape(16.dp))
            .background(Color(0xE6991B1B), RoundedCornerShape(16.dp))
            .padding(20.dp),
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            // Warning icon
            Icon(
                imageVector = Icons.Default.Warning,
                contentDescription = "Critical alert",
                tint = Color.White,
                modifier = Modifier.size(48.dp),
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "SCAM DETECTED",
                color = Color.White,
                fontSize = 24.sp,
                fontWeight = FontWeight.ExtraBold,
                textAlign = TextAlign.Center,
            )

            Text(
                text = "${(score * 100).toInt()}% Confidence",
                color = Color.White.copy(alpha = 0.9f),
                fontSize = 16.sp,
                textAlign = TextAlign.Center,
            )

            if (reasons.isNotEmpty()) {
                Spacer(modifier = Modifier.height(12.dp))
                reasons.take(3).forEach { reason ->
                    Text(
                        text = "• $reason",
                        color = Color.White.copy(alpha = 0.85f),
                        fontSize = 14.sp,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally),
            ) {
                Button(
                    onClick = onBlock,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEF4444)),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.weight(1f),
                ) {
                    Text("Block Caller", fontWeight = FontWeight.Bold, color = Color.White)
                }

                OutlinedButton(
                    onClick = onDismiss,
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.weight(1f),
                ) {
                    Text("Dismiss", color = Color.White.copy(alpha = 0.8f))
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Protected by SentinelEdge",
                color = Color.White.copy(alpha = 0.5f),
                fontSize = 11.sp,
                textAlign = TextAlign.Center,
            )
        }
    }
}

/**
 * Amber overlay for high-risk calls (75–90%).
 */
@Composable
private fun HighAlert(
    score: Float,
    reasons: List<String>,
    onBlock: () -> Unit,
    onDismiss: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
            .border(2.dp, Color(0xFFF59E0B), RoundedCornerShape(16.dp))
            .background(Color(0xE6783610), RoundedCornerShape(16.dp))
            .padding(16.dp),
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Warning,
                    contentDescription = "High risk",
                    tint = Color(0xFFF59E0B),
                    modifier = Modifier.size(32.dp),
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "HIGH RISK CALL",
                    color = Color(0xFFF59E0B),
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                )
            }

            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = "Suspicious patterns detected • ${(score * 100).toInt()}%",
                color = Color.White.copy(alpha = 0.8f),
                fontSize = 14.sp,
            )

            if (reasons.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                reasons.take(3).forEach { reason ->
                    Text(
                        text = "• $reason",
                        color = Color.White.copy(alpha = 0.75f),
                        fontSize = 13.sp,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally),
            ) {
                Button(
                    onClick = onBlock,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEF4444)),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.weight(1f),
                ) {
                    Text("Block Caller", color = Color.White)
                }

                OutlinedButton(
                    onClick = onDismiss,
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.weight(1f),
                ) {
                    Text("Dismiss", color = Color.White.copy(alpha = 0.8f))
                }
            }
        }
    }
}

/**
 * Compact amber banner for medium-risk calls (50–75%).
 */
@Composable
private fun MediumAlert(score: Float, onDismiss: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 4.dp)
            .background(Color(0xCC78360F), RoundedCornerShape(12.dp))
            .padding(12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Shield,
                    contentDescription = "Caution",
                    tint = Color(0xFFF59E0B),
                    modifier = Modifier.size(20.dp),
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Caution • ${(score * 100).toInt()}% suspicious",
                    color = Color.White,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium,
                )
            }

            OutlinedButton(
                onClick = onDismiss,
                shape = RoundedCornerShape(8.dp),
            ) {
                Text("OK", color = Color.White, fontSize = 12.sp)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lifecycle owner for the overlay ComposeView
// ---------------------------------------------------------------------------

/**
 * Minimal [LifecycleOwner] + [SavedStateRegistryOwner] so that a [ComposeView]
 * hosted inside a [WindowManager] overlay has a valid lifecycle.
 */
private class OverlayLifecycleOwner :
    LifecycleOwner, SavedStateRegistryOwner {

    private val lifecycleRegistry = LifecycleRegistry(this)
    private val savedStateController = SavedStateRegistryController.create(this)

    override val lifecycle: Lifecycle
        get() = lifecycleRegistry

    override val savedStateRegistry
        get() = savedStateController.savedStateRegistry

    init {
        savedStateController.performRestore(null)
    }

    fun handleLifecycleEvent(event: Lifecycle.Event) {
        lifecycleRegistry.handleLifecycleEvent(event)
    }
}
