# KBD Integration: bdd-testing

KBD invokes `bdd-testing` as the **Behavioral Verification layer** for each
change. BDD feature files serve as the living specification that proves a
change is functionally complete, not just structurally present.

**Global skill location**: `.agent/skills/bdd-testing/SKILL.md`
**Entry**: Generates Cucumber.js `.feature` + step definition files
**Runner**: `pnpm test:bdd` (or stack-appropriate equivalent from `project.json`)

---

## When KBD Invokes bdd-testing

| KBD Phase | BDD Role | Trigger |
|-----------|----------|---------|
| **Plan** (per-change) | Generate Gherkin scenarios as acceptance criteria | When creating an OpenSpec change or `change.md` |
| **Execute** (per-change QA) | Write step definitions + run Cucumber | After change is marked IN_PROGRESS |
| **Execute** (verification gate) | All BDD scenarios passing = change complete | Before marking DONE in progress.json |
| **Reflect** | BDD coverage report informs goal completion % | When generating reflection.md |

---

## Placement Within a KBD Change

```
Change spec created (OpenSpec /opsx:new or change.md)
  → Write Gherkin scenarios (acceptance criteria)   ← bdd-testing (feature file)
  → Executing tool implements the feature
  → Write step definitions                          ← bdd-testing (steps file)
  → Run: pnpm test:bdd:tag "@<change-id>"
  → All scenarios PASS → change DONE in progress.json
  → artifact-refiner QA gate
  → /opsx:archive or native archive
```

---

## Feature File Convention for KBD Changes

Name feature files after the change ID for traceability:

```
tests/features/
  ui/
    change-007-complete-team-invitations.feature
    change-008-clients-page.feature
  api/
    change-007-invitation-api.feature
```

Tag each feature with both its layer and its change ID:

```gherkin
@ui @video @change-007
Feature: Complete Team Invitations
  As a company owner
  I need to invite team members via email
  So that they can join and access the platform

  Scenario: Owner sends invitation email
    Given I am logged in as an owner
    When I navigate to the Consultants page
    And I enter "newconsultant@example.com" in the invite input
    And I click the "Send Invite" button
    Then I should see "Invitation sent" confirmation
    And the invitation should appear in the pending list
```

---

## What KBD Reads Back

After `pnpm test:bdd`, KBD checks:
- Exit code 0 = scenarios PASS → proceed to archive
- Exit code non-zero = scenarios FAIL → change stays IN_PROGRESS, add to blockers

KBD also reads the Cucumber HTML report at `tests/reports/*.html` to surface
scenario-level failure detail in `progress.json` notes.

---

## Antigravity Browser Recording

For `@ui` scenarios, Antigravity can record browser flows as proof of work:

```
browser_subagent(
  TaskName: "BDD Verify: <change-id>",
  Task: "<scenario description from feature file>",
  RecordingName: "<change_id>_verification"
)
```

KBD uses these recordings as evidence in the `reflection.md` "Delivered Changes"
section, embedded as video links.

---

## When NOT to Use

- For infrastructure-only changes (migrations, config) with no UI or API surface:
  skip BDD, use direct integration test commands
- For changes < 30 min complexity: the overhead of Gherkin may outweigh the value;
  use a quick Playwright script instead
- `bdd-testing` is most valuable for user-facing features where acceptance criteria
  can be stated from the user's perspective
