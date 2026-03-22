package com.sentineledge.android.ui.screens

import android.content.res.Configuration
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.sentineledge.android.ui.components.NoteCard
import com.sentineledge.android.ui.components.NotificationCard
import com.sentineledge.android.ui.components.SectionLabel
import com.sentineledge.android.ui.theme.DangerTone
import com.sentineledge.android.ui.theme.SafeTone
import com.sentineledge.android.ui.theme.SentinelEdgeTheme
import com.sentineledge.android.ui.theme.WarningTone

@Composable
fun NotificationsScreen(modifier: Modifier = Modifier) {
    LazyColumn(
        modifier = modifier,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        item {
            SectionLabel(
                eyebrow = "Screen 5",
                title = "Notification surfaces",
                supportingText = "Foreground protection stays persistent while post-call summaries translate the result into a compact Android notification.",
            )
        }
        item {
            SectionLabel(
                eyebrow = "Component",
                title = "Persistent notification",
                supportingText = "Low-drama, always visible, and easy to trust.",
            )
        }
        item {
            NotificationCard(
                title = "SentinelEdge protection active",
                body = "Monitoring live calls on-device. Tap to review recent decisions and privacy settings.",
                timestamp = "Now",
                accent = MaterialTheme.colorScheme.primary,
            )
        }
        item {
            SectionLabel(
                eyebrow = "Component",
                title = "Post-call results",
                supportingText = "A short outcome string plus risk percentage keeps the notification easy to parse from the shade.",
            )
        }
        item {
            NotificationCard(
                title = "SentinelEdge analyzed your call",
                body = "Safe - 8% scam risk",
                timestamp = "11:31 AM",
                accent = SafeTone,
            )
        }
        item {
            NotificationCard(
                title = "SentinelEdge analyzed your call",
                body = "Warning - 73% scam risk",
                timestamp = "Yesterday - 7:53 PM",
                accent = WarningTone,
            )
        }
        item {
            NotificationCard(
                title = "SentinelEdge analyzed your call",
                body = "Blocked - 93% scam risk",
                timestamp = "Today - 2:16 PM",
                accent = DangerTone,
            )
        }
        item {
            NoteCard(
                title = "Interaction note",
                body = "The persistent notification is tied to active protection. Post-call notifications should deep-link into the exact call detail screen and offer a quick false-positive correction path.",
            )
        }
    }
}

@Preview(showBackground = true, backgroundColor = 0xFFF6F8FB)
@Composable
private fun NotificationsPreviewLight() {
    SentinelEdgeTheme {
        NotificationsScreen()
    }
}

@Preview(
    showBackground = true,
    backgroundColor = 0xFF10131A,
    uiMode = Configuration.UI_MODE_NIGHT_YES,
)
@Composable
private fun NotificationsPreviewDark() {
    SentinelEdgeTheme {
        NotificationsScreen()
    }
}
