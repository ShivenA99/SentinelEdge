package com.sentineledge.android.ui.screens

import android.content.res.Configuration
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.weight
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CallEnd
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.PriorityHigh
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.sentineledge.android.ui.components.NoteCard
import com.sentineledge.android.ui.components.RiskTag
import com.sentineledge.android.ui.components.SectionLabel
import com.sentineledge.android.ui.model.AlertSeverity
import com.sentineledge.android.ui.model.SampleSentinelData
import com.sentineledge.android.ui.model.ScamAlert
import com.sentineledge.android.ui.theme.CriticalTone
import com.sentineledge.android.ui.theme.DangerTone
import com.sentineledge.android.ui.theme.SentinelEdgeTheme
import com.sentineledge.android.ui.theme.WarningTone

@Composable
fun AlertOverlayScreen(modifier: Modifier = Modifier) {
    var selectedSeverity by rememberSaveable { mutableStateOf(AlertSeverity.Warning) }
    val alert = SampleSentinelData.overlayAlerts.first { it.severity == selectedSeverity }

    LazyColumn(
        modifier = modifier,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        item {
            SectionLabel(
                eyebrow = "Screen 1",
                title = "In-call alert overlay",
                supportingText = "Three escalation states for stressful moments, each tuned for fast scanning and thumb-friendly actions.",
            )
        }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                AlertSeverity.entries.forEach { severity ->
                    FilterChip(
                        selected = selectedSeverity == severity,
                        onClick = { selectedSeverity = severity },
                        label = { Text(severityLabel(severity)) },
                        leadingIcon = {
                            if (selectedSeverity == severity) {
                                Icon(Icons.Filled.CheckCircle, contentDescription = null, modifier = Modifier.size(18.dp))
                            }
                        },
                    )
                }
            }
        }
        item {
            OverlayPreviewCard(alert = alert)
        }
        item {
            NoteCard(
                title = "Interaction note",
                body = "Amber keeps the user informed, red pushes for a decision, and critical expands to full-screen once the confidence stays high long enough to avoid noisy false positives.",
            )
        }
    }
}

@Composable
private fun OverlayPreviewCard(alert: ScamAlert) {
    val tone = when (alert.severity) {
        AlertSeverity.Warning -> OverlayToneSet(
            container = WarningTone.copy(alpha = 0.22f),
            onContainer = Color(0xFF3D2A00),
            accent = Color(0xFF9C5C00),
        )
        AlertSeverity.Danger -> OverlayToneSet(
            container = DangerTone.copy(alpha = 0.18f),
            onContainer = Color(0xFF440C12),
            accent = DangerTone,
        )
        AlertSeverity.Critical -> OverlayToneSet(
            container = CriticalTone,
            onContainer = Color.White,
            accent = Color(0xFFFFE4EA),
        )
    }

    ElevatedCard(
        colors = CardDefaults.elevatedCardColors(containerColor = tone.container),
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    brush = Brush.verticalGradient(
                        colors = listOf(tone.container, tone.container.copy(alpha = 0.88f)),
                    ),
                ),
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Icon(
                        imageVector = when (alert.severity) {
                            AlertSeverity.Warning -> Icons.Filled.Warning
                            AlertSeverity.Danger -> Icons.Filled.PriorityHigh
                            AlertSeverity.Critical -> Icons.Filled.Security
                        },
                        contentDescription = null,
                        tint = tone.accent,
                    )
                    Text(
                        text = "${alert.confidence}% confidence",
                        color = tone.accent,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                    )
                }

                Text(
                    text = alert.headline,
                    style = if (alert.severity == AlertSeverity.Critical) {
                        MaterialTheme.typography.headlineMedium
                    } else {
                        MaterialTheme.typography.headlineSmall
                    },
                    color = tone.onContainer,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = alert.supportingText,
                    style = MaterialTheme.typography.bodyLarge,
                    color = tone.onContainer.copy(alpha = 0.85f),
                )

                LinearProgressIndicator(
                    progress = { alert.confidence / 100f },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp),
                    color = tone.accent,
                    trackColor = tone.accent.copy(alpha = 0.18f),
                )

                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    alert.reasons.take(3).forEach { reason ->
                        RiskTag(label = reason, color = tone.accent)
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Button(
                        onClick = {},
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(Icons.Filled.CallEnd, contentDescription = null)
                        Text(text = "Block", modifier = Modifier.padding(start = 8.dp))
                    }
                    OutlinedButton(
                        onClick = {},
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Not a scam")
                    }
                }
            }
        }
    }
}

private data class OverlayToneSet(
    val container: Color,
    val onContainer: Color,
    val accent: Color,
)

private fun severityLabel(severity: AlertSeverity): String = when (severity) {
    AlertSeverity.Warning -> "Amber warning"
    AlertSeverity.Danger -> "Red alert"
    AlertSeverity.Critical -> "Critical"
}

@Preview(showBackground = true, backgroundColor = 0xFFF6F8FB)
@Composable
private fun AlertOverlayPreviewLight() {
    SentinelEdgeTheme {
        AlertOverlayScreen()
    }
}

@Preview(
    showBackground = true,
    backgroundColor = 0xFF10131A,
    uiMode = Configuration.UI_MODE_NIGHT_YES,
)
@Composable
private fun AlertOverlayPreviewDark() {
    SentinelEdgeTheme {
        AlertOverlayScreen()
    }
}
