## ADDED Requirements

### Requirement: HTMX previews SHALL use an explicit runtime source policy
The workflow SHALL resolve HTMX runtime scripts using a declared source policy and SHALL default to a local/offline runtime source unless network mode is explicitly enabled.

#### Scenario: Offline-first runtime resolution
- **WHEN** an HTMX artifact is rendered without network-enabled constraints
- **THEN** the workflow loads HTMX runtime from a local path and renders the preview

#### Scenario: Explicit network-enabled runtime resolution
- **WHEN** an HTMX artifact declares network-enabled preview mode
- **THEN** the workflow may fetch HTMX runtime from configured remote sources and records the source used
