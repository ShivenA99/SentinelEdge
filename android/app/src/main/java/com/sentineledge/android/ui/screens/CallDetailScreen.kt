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
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Subtitles
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
import com.sentineledge.android.ui.components.FeatureMeter
import com.sentineledge.android.ui.components.NoteCard
import com.sentineledge.android.ui.components.RiskTag
import com.sentineledge.android.ui.components.SectionLabel
import com.sentineledge.android.ui.model.SampleSentinelData
import com.sentineledge.android.ui.theme.DangerTone
import com.sentineledge.android.ui.theme.SentinelEdgeTheme

@Composable
fun CallDetailScreen(modifier: Modifier = Modifier) {
    var transcriptOptIn by rememberSaveable { mutableStateOf(true) }

    LazyColumn(
        modifier = modifier,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        item {
            SectionLabel(
                eyebrow = "Screen 4",
                title = "Post-call detail view",
                supportingText = "Explains the decision with transcript, feature weights, and plain-language rationale when the user stores local history.",
            )
        }
        item {
            ElevatedCard(
                colors = CardDefaults.elevatedCardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                ),
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Text(
                        text = "Unknown number - 2:14 PM",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.SemiBold,
                    )
                    RiskTag(
                        label = "73% scam risk",
                        color = DangerTone,
                        icon = Icons.Filled.CheckCircle,
                    )
                    Text(
                        text = "Flagged because the caller claimed bank authority, discouraged callback verification, and moved quickly to payment instructions.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
        item {
            ElevatedCard {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(18.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(
                            text = "Store transcript locally",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = "Transcript never leaves the device and can be disabled anytime.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(
                        checked = transcriptOptIn,
                        onCheckedChange = { transcriptOptIn = it },
                    )
                }
            }
        }
        item {
            SectionLabel(
                eyebrow = "Component",
                title = "Transcript",
                supportingText = if (transcriptOptIn) {
                    "Speaker turns stay readable and scannable."
                } else {
                    "Hidden when the user does not opt in."
                },
            )
        }
        if (transcriptOptIn) {
            items(SampleSentinelData.transcript) { line ->
                ElevatedCard {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Icon(Icons.Filled.Subtitles, contentDescription = null)
                            Text(
                                text = line.speaker,
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.SemiBold,
                            )
                        }
                        Text(
                            text = line.text,
                            style = MaterialTheme.typography.bodyLarge,
                        )
                    }
                }
            }
        } else {
            item {
                NoteCard(
                    title = "Transcript unavailable",
                    body = "Users who do not opt in still get the scam decision, score, and feature explanation without any local transcript history.",
                )
            }
        }
        item {
            SectionLabel(
                eyebrow = "Component",
                title = "Feature breakdown",
                supportingText = "Model signals are translated into human-readable indicators instead of raw ML internals.",
            )
        }
        items(SampleSentinelData.featureBreakdown) { feature ->
            ElevatedCard {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.Top,
                ) {
                    Icon(Icons.Filled.GraphicEq, contentDescription = null, modifier = Modifier.padding(top = 4.dp))
                    FeatureMeter(feature = feature)
                }
            }
        }
        item {
            NoteCard(
                title = "Interaction note",
                body = "This screen becomes the explanation layer after a call. It should reduce fear, show why the model acted, and make corrections easy when the user marks a false positive.",
            )
        }
    }
}

@Preview(showBackground = true, backgroundColor = 0xFFF6F8FB)
@Composable
private fun CallDetailPreviewLight() {
    SentinelEdgeTheme {
        CallDetailScreen()
    }
}

@Preview(
    showBackground = true,
    backgroundColor = 0xFF10131A,
    uiMode = Configuration.UI_MODE_NIGHT_YES,
)
@Composable
private fun CallDetailPreviewDark() {
    SentinelEdgeTheme {
        CallDetailScreen()
    }
}
