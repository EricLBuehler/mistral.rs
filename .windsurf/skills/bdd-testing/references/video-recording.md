# Video Recording Reference

## Playwright `recordVideo` API

Enable video recording when creating a `BrowserContext`:

```typescript
const context = await browser.newContext({
  recordVideo: {
    dir: 'tests/reports/videos/',           // Output directory
    size: { width: 1280, height: 720 },     // Video resolution
  },
});
const page = await context.newPage();
// ... test actions ...

// IMPORTANT: Video is only finalized after context.close()
const videoPath = await page.video()?.path();
await context.close();
```

### Key Facts

- Videos are saved as **WebM** files
- Video is only available **after** `page.close()` or `context.close()`
- `page.video()?.path()` returns the path to the video file
- `page.video()?.saveAs(path)` copies the video to a custom location
- Video size defaults to viewport size scaled to fit 800x800
- Set explicit `size` to control output resolution

### Environment Control

```bash
# Enable video for UI tests
VIDEO=true pnpm test:bdd:ui

# Disable headless mode to see browser during recording
HEADLESS=false pnpm test:bdd:ui
```

## Playwright Trace Viewer

For deeper debugging, enable tracing:

```typescript
await context.tracing.start({ screenshots: true, snapshots: true });
// ... test actions ...
await context.tracing.stop({ path: 'tests/reports/traces/trace.zip' });
```

View traces with: `npx playwright show-trace tests/reports/traces/trace.zip`

## Antigravity `browser_subagent` Recording

Antigravity can record browser interactions as WebP videos via the `browser_subagent` tool:

```
browser_subagent(
  TaskName: "Test: [Scenario Name]",
  Task: "[Detailed steps to perform in the browser]",
  RecordingName: "[snake_case_name]"   // → saved as .webp in artifacts
)
```

### When to Use

| Method | Format | When to Use |
|--------|--------|-------------|
| Playwright `recordVideo` | WebM | Automated test runs, CI/CD, batch execution |
| Antigravity `browser_subagent` | WebP | Interactive debugging, demo recordings, visual verification |

### Examples

```
// Record a login test
browser_subagent(
  TaskName: "BDD Test: Login",
  Task: "Go to localhost:3000/login. Fill [data-testid=email-input] with test@example.com. Fill [data-testid=password-input] with password. Click [data-testid=login-button]. Wait for URL to contain /dashboard. Return success if dashboard loads.",
  RecordingName: "login_test"
)

// Record a navigation test
browser_subagent(
  TaskName: "BDD Test: Sidebar Navigation",
  Task: "Go to localhost:3000/dashboard. Click each sidebar link. Verify each page loads. Return list of pages visited.",
  RecordingName: "sidebar_navigation"
)
```
