## ADDED Requirements

### Requirement: UI and A2UI artifacts SHALL produce browser preview outputs
The refinement workflow SHALL execute a deterministic browser preview stage for `ui` and `a2ui` artifact types and SHALL persist preview outputs under `dist/previews/`.

#### Scenario: Preview generated for UI artifact
- **WHEN** a refinement run includes `artifact_type: ui`
- **THEN** the workflow generates a browser preview artifact set in `dist/previews/`

#### Scenario: Preview generated for A2UI artifact
- **WHEN** a refinement run includes `artifact_type: a2ui`
- **THEN** the workflow generates a browser preview artifact set in `dist/previews/`
