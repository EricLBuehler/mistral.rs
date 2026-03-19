## ADDED Requirements

### Requirement: Preview validation results SHALL be persisted for convergence checks
The workflow SHALL persist preview validation metadata, including render status and browser diagnostics, and SHALL include references to preview evidence in artifact manifest outputs.

#### Scenario: Preview diagnostics persisted
- **WHEN** a browser preview stage completes
- **THEN** a structured preview report is written with render outcome and browser diagnostics

#### Scenario: Manifest includes preview references
- **WHEN** preview artifacts are generated
- **THEN** the artifact manifest contains file references to preview report and screenshot outputs
