package com.sentineledge.android.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

private val LightColors = lightColorScheme(
    primary = BrandPrimary,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFA5EEF7),
    onPrimaryContainer = Color(0xFF001F24),
    secondary = BrandSecondary,
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFD0E5E8),
    onSecondaryContainer = Color(0xFF081F23),
    background = Color(0xFFF5F7FA),
    onBackground = Color(0xFF161D1E),
    surface = Color(0xFFFBFCFF),
    onSurface = Color(0xFF171D1E),
    surfaceContainerHigh = Color(0xFFE7EDF0),
    surfaceContainerHighest = Color(0xFFDDE3E6),
)

private val DarkColors = darkColorScheme(
    primary = BrandPrimaryDark,
    onPrimary = Color(0xFF00363C),
    primaryContainer = Color(0xFF004F58),
    onPrimaryContainer = Color(0xFFA5EEF7),
    secondary = BrandSecondaryDark,
    onSecondary = Color(0xFF203336),
    secondaryContainer = Color(0xFF364A4D),
    onSecondaryContainer = Color(0xFFD0E5E8),
    background = Color(0xFF0F1415),
    onBackground = Color(0xFFDEE4E6),
    surface = Color(0xFF101617),
    onSurface = Color(0xFFDEE4E6),
    surfaceContainerHigh = Color(0xFF1C2527),
    surfaceContainerHighest = Color(0xFF263033),
)

@Composable
fun SentinelEdgeTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit,
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = SentinelTypography,
        content = content,
    )
}
