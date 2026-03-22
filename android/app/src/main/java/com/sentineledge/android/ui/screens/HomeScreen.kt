package com.sentineledge.android.ui.screens

import android.content.res.Configuration
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoGraph
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.HealthAndSafety
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.sentineledge.android.ui.components.NoteCard
import com.sentineledge.android.ui.components.RiskTag
import com.sentineledge.android.ui.components.SectionLabel
import com.sentineledge.android.ui.components.StatCard
import com.sentineledge.android.ui.model.SampleSentinelData
import com.sentineledge.android.ui.theme.DangerTone
import com.sentineledge.android.ui.theme.SafeTone
import com.sentineledge.android.ui.theme.SentinelEdgeTheme
import com.sentineledge.android.ui.theme.WarningTone

@Composable
fun HomeScreen(modifier: Modifier = Modifier) {
    var protectionEnabled by rememberSaveable { mutableStateOf(true) }

    LazyColumn(
        modifier = modifier,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        item {
            SectionLabel(
                eyebrow = "Screen 2",
                title = "Protection home",
                supportingText = "A real Android entry point with status, quick stats, and recent calls instead of the existing web demo simulator.",
            )
        }
        item {
            ElevatedCard(
                colors = CardDefaults.elevatedCardColors(
                    containerColor = if (protectionEnabled) {
                        MaterialTheme.colorScheme.primaryContainer
                    } else {
                        MaterialTheme.colorScheme.surfaceContainerHigh
                    },
                ),
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(20.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Icon(
                                imageVector = if (protectionEnabled) Icons.Filled.Shield else Icons.Filled.HealthAndSafety,
                                contentDescription = null,
                            )
                            Text(
                                text = if (protectionEnabled) "Protection active" else "Protection inactive",
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Bold,
                            )
                        }
                        Text(
                            text = if (protectionEnabled) {
                                "Foreground protection, post-call summaries, and alert overlays are ready."
                            } else {
                                "Calls are not being scored right now. The persistent notification is also paused."
                            },
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(
                        checked = protectionEnabled,
                        onCheckedChange = { protectionEnabled = it },
                    )
                }
            }
        }
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                StatCard(
                    label = "Calls analyzed",
                    value = "184",
                    modifier = Modifier.weight(1f),
                )
                StatCard(
                    label = "Scams blocked",
                    value = "23",
                    modifier = Modifier.weight(1f),
                    accent = DangerTone,
                )
            }
        }
        item {
            ElevatedCard(
                colors = CardDefaults.elevatedCardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                ),
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(18.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(Icons.Filled.AutoGraph, contentDescription = null)
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(
                            text = "Today's trend",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = "Most high-risk calls spiked within the first 30 seconds, so alerts stay front-loaded.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
        item {
            SectionLabel(
                eyebrow = "Component",
                title = "Recent calls",
                supportingText = "Scam indicators stay visible without overwhelming the list.",
            )
        }
        items(SampleSentinelData.recentCalls) { call ->
            ElevatedCard {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text(
                        text = call.caller,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = call.time,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    RiskTag(
                        label = "${call.resultLabel} - ${call.confidence}% risk",
                        color = when {
                            call.confidence >= 85 -> DangerTone
                            call.confidence >= 40 -> WarningTone
                            else -> SafeTone
                        },
                        icon = Icons.Filled.CheckCircle,
                    )
                    Text(
                        text = call.note,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
        item {
            NoteCard(
                title = "Interaction note",
                body = "The status switch mirrors the foreground protection service. Tapping a call is expected to open the Call Detail screen rather than a web-style modal or phone mock shell.",
            )
        }
    }
}

@Preview(showBackground = true, backgroundColor = 0xFFF6F8FB)
@Composable
private fun HomePreviewLight() {
    SentinelEdgeTheme {
        HomeScreen()
    }
}

@Preview(
    showBackground = true,
    backgroundColor = 0xFF10131A,
    uiMode = Configuration.UI_MODE_NIGHT_YES,
)
@Composable
private fun HomePreviewDark() {
    SentinelEdgeTheme {
        HomeScreen()
    }
}
