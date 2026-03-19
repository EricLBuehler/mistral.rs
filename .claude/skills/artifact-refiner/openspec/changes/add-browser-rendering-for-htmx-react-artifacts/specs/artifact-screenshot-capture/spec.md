## ADDED Requirements

### Requirement: Browser preview stage SHALL capture screenshot evidence
The workflow SHALL capture at least one PNG screenshot for each successful browser preview and SHALL persist it in `dist/previews/<artifact-id>/`.

#### Scenario: Screenshot captured for successful render
- **WHEN** browser preview renders without fatal errors
- **THEN** a PNG screenshot file is written to the preview output directory

#### Scenario: Screenshot failure is reported
- **WHEN** screenshot capture fails after page render attempt
- **THEN** the workflow records the failure in preview diagnostics and marks screenshot evidence as missing
