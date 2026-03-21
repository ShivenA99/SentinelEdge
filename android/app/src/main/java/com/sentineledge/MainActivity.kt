package com.sentineledge

import android.app.role.RoleManager
import android.content.Context
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * Main activity for SentinelEdge.
 *
 * Displays the app name, current protection status, a button to request
 * the default call-screening role, and a log of recently screened calls.
 */
class MainActivity : ComponentActivity() {

    private var isDefaultScreener = mutableStateOf(false)
    private val screenedCalls = mutableStateListOf<ScreenedCallEntry>()

    private val roleRequestLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        isDefaultScreener.value = checkIsDefaultScreener()
        Log.i(SentinelApp.TAG, "Role request result: ${result.resultCode}")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        isDefaultScreener.value = checkIsDefaultScreener()

        // Add a few placeholder entries so the UI isn't empty on first launch
        if (screenedCalls.isEmpty()) {
            screenedCalls.add(
                ScreenedCallEntry("(No calls screened yet)", "-- ", "")
            )
        }

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    SentinelEdgeScreen(
                        isDefaultScreener = isDefaultScreener.value,
                        screenedCalls = screenedCalls,
                        onRequestRole = { requestCallScreeningRole() }
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        isDefaultScreener.value = checkIsDefaultScreener()
    }

    // ------------------------------------------------------------------
    // Call-screening role helpers
    // ------------------------------------------------------------------

    private fun checkIsDefaultScreener(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return false
        val roleManager = getSystemService(Context.ROLE_SERVICE) as? RoleManager
        return roleManager?.isRoleHeld(RoleManager.ROLE_CALL_SCREENING) ?: false
    }

    private fun requestCallScreeningRole() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            Log.w(SentinelApp.TAG, "Call screening requires Android 10+")
            return
        }
        val roleManager = getSystemService(Context.ROLE_SERVICE) as RoleManager
        if (!roleManager.isRoleHeld(RoleManager.ROLE_CALL_SCREENING)) {
            val intent = roleManager.createRequestRoleIntent(RoleManager.ROLE_CALL_SCREENING)
            roleRequestLauncher.launch(intent)
        }
    }
}

// ---------------------------------------------------------------------------
// Data class for call log entries
// ---------------------------------------------------------------------------

data class ScreenedCallEntry(
    val number: String,
    val timestamp: String,
    val result: String,
)

// ---------------------------------------------------------------------------
// Compose UI
// ---------------------------------------------------------------------------

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SentinelEdgeScreen(
    isDefaultScreener: Boolean,
    screenedCalls: List<ScreenedCallEntry>,
    onRequestRole: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("SentinelEdge") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    titleContentColor = MaterialTheme.colorScheme.onPrimary,
                ),
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            // -- Protection status card --
            ProtectionStatusCard(isActive = isDefaultScreener)

            Spacer(modifier = Modifier.height(16.dp))

            // -- Set as default button --
            if (!isDefaultScreener) {
                Button(onClick = onRequestRole) {
                    Text("Set as Default Call Screener")
                }
                Spacer(modifier = Modifier.height(16.dp))
            }

            // -- Recent calls header --
            Text(
                text = "Recent Screened Calls",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.fillMaxWidth(),
            )

            Spacer(modifier = Modifier.height(8.dp))

            // -- Call log --
            LazyColumn(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(screenedCalls) { entry ->
                    CallLogCard(entry)
                }
            }
        }
    }
}

@Composable
fun ProtectionStatusCard(isActive: Boolean) {
    val statusColor = if (isActive) Color(0xFF4CAF50) else Color(0xFFFFA726)
    val statusText = if (isActive) "Protection Active" else "Protection Inactive"
    val descriptionText = if (isActive) {
        "SentinelEdge is screening your incoming calls."
    } else {
        "Tap the button below to enable call screening."
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = statusColor.copy(alpha = 0.12f)),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = statusText,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = statusColor,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = descriptionText,
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
fun CallLogCard(entry: ScreenedCallEntry) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = entry.number,
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.Medium,
                )
                Text(
                    text = entry.timestamp,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (entry.result.isNotBlank()) {
                Text(
                    text = entry.result,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
        }
    }
}
