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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.PrivacyTip
import androidx.compose.material.icons.filled.SettingsSuggest
import androidx.compose.material.icons.filled.SyncLock
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.sentineledge.android.ui.components.NoteCard
import com.sentineledge.android.ui.components.SectionLabel
import com.sentineledge.android.ui.theme.SentinelEdgeTheme

@Composable
fun SettingsScreen(modifier: Modifier = Modifier) {
    var protectionEnabled by rememberSaveable { mutableStateOf(true) }
    var federatedLearningEnabled by rememberSaveable { mutableStateOf(true) }
    var sensitivity by rememberSaveable { mutableFloatStateOf(0.35f) }

    LazyColumn(
        modifier = modifier,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        item {
            SectionLabel(
                eyebrow = "Screen 3",
                title = "Protection settings",
                supportingText = "Material 3 controls with clear privacy language and enough context to avoid accidental changes.",
            )
        }
        item {
            ElevatedCard {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(18.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text(
                                text = "Enable protection",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                text = "Turns the overlay, foreground notification, and live scoring pipeline on or off.",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Switch(
                            checked = protectionEnabled,
                            onCheckedChange = { protectionEnabled = it },
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text(
                                text = "Federated learning",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                text = "Opt in to share differentially private model updates only, never transcripts or raw audio.",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Switch(
                            checked = federatedLearningEnabled,
                            onCheckedChange = { federatedLearningEnabled = it },
                        )
                    }
                }
            }
        }
        item {
            ElevatedCard {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp),
                ) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(Icons.Filled.SettingsSuggest, contentDescription = null)
                        Text(
                            text = "Alert sensitivity",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                    }
                    Text(
                        text = "Aggressive catches scams earlier but may create more false positives. Conservative waits for more evidence.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Slider(
                        value = sensitivity,
                        onValueChange = { sensitivity = it },
                    )
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Aggressive", style = MaterialTheme.typography.labelMedium)
                        Text("Conservative", style = MaterialTheme.typography.labelMedium)
                    }
                }
            }
        }
        item {
            ElevatedCard {
                Column(modifier = Modifier.padding(vertical = 6.dp)) {
                    ListItem(
                        headlineContent = { Text("Privacy information") },
                        supportingContent = { Text("Explain on-device inference, transcript handling, and differential privacy.") },
                        leadingContent = { Icon(Icons.Filled.PrivacyTip, contentDescription = null) },
                        trailingContent = { Icon(Icons.Filled.ChevronRight, contentDescription = null) },
                    )
                    ListItem(
                        headlineContent = { Text("Model update status") },
                        supportingContent = { Text("Last synced 2 days ago with encrypted and signed update packages.") },
                        leadingContent = { Icon(Icons.Filled.SyncLock, contentDescription = null) },
                        trailingContent = { Icon(Icons.Filled.ChevronRight, contentDescription = null) },
                    )
                }
            }
        }
        item {
            NoteCard(
                title = "Interaction note",
                body = "These controls should live inside the Android app settings flow, not a web dashboard. Privacy copy stays plain-language and close to the toggles that matter.",
            )
        }
    }
}

@Preview(showBackground = true, backgroundColor = 0xFFF6F8FB)
@Composable
private fun SettingsPreviewLight() {
    SentinelEdgeTheme {
        SettingsScreen()
    }
}

@Preview(
    showBackground = true,
    backgroundColor = 0xFF10131A,
    uiMode = Configuration.UI_MODE_NIGHT_YES,
)
@Composable
private fun SettingsPreviewDark() {
    SentinelEdgeTheme {
        SettingsScreen()
    }
}
