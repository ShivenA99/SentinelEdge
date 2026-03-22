package com.sentineledge.android.ui.model

enum class AlertSeverity {
    Warning,
    Danger,
    Critical,
}

data class ScamAlert(
    val severity: AlertSeverity,
    val confidence: Int,
    val headline: String,
    val supportingText: String,
    val reasons: List<String>,
)

data class CallSummary(
    val caller: String,
    val time: String,
    val confidence: Int,
    val resultLabel: String,
    val note: String,
)

data class FeatureScore(
    val label: String,
    val strength: Float,
    val detail: String,
) {
    val percentLabel: String
        get() = "${(strength * 100).toInt()}%"
}

data class TranscriptLine(
    val speaker: String,
    val text: String,
)

object SampleSentinelData {
    val overlayAlerts = listOf(
        ScamAlert(
            severity = AlertSeverity.Warning,
            confidence = 68,
            headline = "Warning: likely pressure tactic",
            supportingText = "Readable at a glance with room for the user to continue if context changes.",
            reasons = listOf(
                "Urgent account closure language",
                "Requests immediate payment",
                "Caller avoids callback verification",
            ),
        ),
        ScamAlert(
            severity = AlertSeverity.Danger,
            confidence = 87,
            headline = "Scam risk rising fast",
            supportingText = "Escalates to stronger color and sharper language once multiple scam cues stack up.",
            reasons = listOf(
                "Threatens legal action",
                "Mentions gift cards or wire transfer",
                "Asks to stay on the line while paying",
            ),
        ),
        ScamAlert(
            severity = AlertSeverity.Critical,
            confidence = 96,
            headline = "Block this caller now",
            supportingText = "Full-screen critical state interrupts the moment when the model is highly confident.",
            reasons = listOf(
                "Impersonation of bank security team",
                "Credential collection detected",
                "High-confidence fraud score sustained",
            ),
        ),
    )

    val recentCalls = listOf(
        CallSummary(
            caller = "Unknown number",
            time = "Today - 2:14 PM",
            confidence = 93,
            resultLabel = "Blocked",
            note = "Bank impersonation and credential harvesting cues.",
        ),
        CallSummary(
            caller = "Mesa Family Clinic",
            time = "Today - 11:30 AM",
            confidence = 8,
            resultLabel = "Safe",
            note = "Routine appointment confirmation with no pressure language.",
        ),
        CallSummary(
            caller = "Unknown number",
            time = "Yesterday - 7:52 PM",
            confidence = 74,
            resultLabel = "Warned",
            note = "Gift card payment request and urgency spike.",
        ),
    )

    val featureBreakdown = listOf(
        FeatureScore(
            label = "Urgency language",
            strength = 0.89f,
            detail = "Repeated words like immediately, urgent, and right now increase risk quickly.",
        ),
        FeatureScore(
            label = "Authority impersonation",
            strength = 0.81f,
            detail = "Caller claims to represent a bank security team without verification paths.",
        ),
        FeatureScore(
            label = "Payment coercion",
            strength = 0.76f,
            detail = "Gift-card and transfer instructions match known scam patterns.",
        ),
        FeatureScore(
            label = "Benign context",
            strength = 0.14f,
            detail = "Very little scheduling or relationship language that would soften the score.",
        ),
    )

    val transcript = listOf(
        TranscriptLine(
            speaker = "Caller",
            text = "This is the fraud department. Your checking account is compromised and we need to secure it immediately.",
        ),
        TranscriptLine(
            speaker = "You",
            text = "Can I call the number on my card instead?",
        ),
        TranscriptLine(
            speaker = "Caller",
            text = "No, that will freeze the recovery process. Stay with me and purchase the verification cards now.",
        ),
        TranscriptLine(
            speaker = "You",
            text = "That sounds suspicious.",
        ),
    )
}
