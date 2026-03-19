## ADDED Requirements

### Requirement: React TSX artifacts SHALL be compiled before browser preview
The workflow SHALL compile React TSX preview sources into browser-loadable JavaScript before attempting browser rendering, and SHALL fail the preview stage if compilation does not succeed.

#### Scenario: Successful TSX compilation
- **WHEN** a preview input contains a `.tsx` entry file with resolvable imports
- **THEN** the workflow emits compiled preview assets and proceeds to browser rendering

#### Scenario: Compilation failure halts preview stage
- **WHEN** a `.tsx` preview input fails compilation
- **THEN** the workflow marks preview as failed and records compilation diagnostics
