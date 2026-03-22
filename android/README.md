# SentinelEdge Android UI

This directory contains a standalone Android UI implementation for issue `#28`. It is intentionally separate from the web demo under `demo/frontend` and does not reuse the web phone simulator layout, styling, or component structure.

## What is included

- Jetpack Compose + Material 3 app shell
- Home screen
- Alert overlay screen with amber, red, and critical variants
- Settings screen
- Call detail screen
- Notification preview screen
- Light and dark theme previews in code

## Open the project

1. Open the `android/` directory in Android Studio.
2. Let Android Studio sync the Gradle project.
3. Run the `app` configuration on an emulator or device.

## Notes

- The module is UI-focused and uses sample data for now.
- It is designed to be the Android-native surface for SentinelEdge, separate from the existing web demo.
- Gradle wrapper files were not generated in this environment because Gradle is not installed here. Android Studio can still import the project, or you can generate the wrapper locally with `gradle wrapper`.
