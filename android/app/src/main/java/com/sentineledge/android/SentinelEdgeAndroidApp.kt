package com.sentineledge.android

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.NotificationsActive
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.tooling.preview.Preview
import com.sentineledge.android.ui.screens.AlertOverlayScreen
import com.sentineledge.android.ui.screens.CallDetailScreen
import com.sentineledge.android.ui.screens.HomeScreen
import com.sentineledge.android.ui.screens.NotificationsScreen
import com.sentineledge.android.ui.screens.SettingsScreen
import com.sentineledge.android.ui.theme.SentinelEdgeTheme

private enum class AppDestination(
    val label: String,
    val icon: ImageVector,
) {
    Home("Home", Icons.Filled.Home),
    Alerts("Alerts", Icons.Filled.WarningAmber),
    CallDetail("Call Detail", Icons.Filled.Description),
    Notifications("Notifications", Icons.Filled.NotificationsActive),
    Settings("Settings", Icons.Filled.Settings),
}

@Composable
fun SentinelEdgeAndroidApp() {
    SentinelEdgeTheme {
        var destination by rememberSaveable { mutableStateOf(AppDestination.Home) }

        Surface(modifier = Modifier.fillMaxSize()) {
            AppScaffold(
                destination = destination,
                onDestinationChange = { destination = it },
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AppScaffold(
    destination: AppDestination,
    onDestinationChange: (AppDestination) -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(destination.label) },
                colors = TopAppBarDefaults.topAppBarColors(),
            )
        },
        bottomBar = {
            NavigationBar {
                AppDestination.entries.forEach { item ->
                    NavigationBarItem(
                        selected = destination == item,
                        onClick = { onDestinationChange(item) },
                        icon = { Icon(item.icon, contentDescription = item.label) },
                        label = { Text(item.label) },
                    )
                }
            }
        },
    ) { innerPadding ->
        ScreenContent(destination = destination, innerPadding = innerPadding)
    }
}

@Composable
private fun ScreenContent(
    destination: AppDestination,
    innerPadding: PaddingValues,
) {
    val screenModifier = Modifier
        .fillMaxSize()
        .padding(innerPadding)

    when (destination) {
        AppDestination.Home -> HomeScreen(screenModifier)
        AppDestination.Alerts -> AlertOverlayScreen(screenModifier)
        AppDestination.CallDetail -> CallDetailScreen(screenModifier)
        AppDestination.Notifications -> NotificationsScreen(screenModifier)
        AppDestination.Settings -> SettingsScreen(screenModifier)
    }
}

@Preview(showBackground = true, backgroundColor = 0xFFF6F8FB)
@Composable
private fun AppPreviewLight() {
    SentinelEdgeAndroidApp()
}
