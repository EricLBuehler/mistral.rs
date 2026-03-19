# State Management Architecture

The Iterative Evolver uses a **State Provider** abstraction to decouple the evolution loop from storage backends. This enables the skill to work with dedicated state tools, agent memory systems, plain files, or custom backends — all through a single interface.

---

## Core Concepts

### Evolution Name

Every evolution cycle has a user-defined `evolution_name` — a human-friendly identifier used as the primary retrieval key across sessions.

```
/evolve "uar-api-improvement"
/evolve-status "q1-sales-strategy"
```

The `evolution_name`:
- Is provided by the user or auto-generated from the goal description
- Is unique within the active state provider
- Survives session boundaries — load state by name in any future session
- Is distinct from `evolution_id` (an internal UUID for deduplication)

### State Lifecycle

```
┌──────────┐      ┌─────────────┐      ┌───────────┐
│   INIT   │─────▶│ CHECKPOINT  │─────▶│ FINALIZE  │
│ (start)  │      │(mid-session)│      │(end-state)│
└──────────┘      └─────────────┘      └───────────┘
      ▲                                       │
      │           Next Cycle                  │
      └───────────────────────────────────────┘
```

#### init(evolution_name)
- **New evolution**: Creates empty state with the given name
- **Existing evolution**: Loads the current state for that name
- **Finalized evolution**: Loads the finalized end-state as the new start-state, increments cycle counter

#### checkpoint(evolution_name, event)
- Snapshots the current state after each phase
- Enables crash recovery — if a session dies mid-cycle, the last checkpoint is available
- Tagged with `event_type`, `phase`, `timestamp`

#### finalize(evolution_name)
- Marks the current state as the **end-state** of this iteration
- Archives to iteration history (e.g., `history/iteration-3.json`)
- The finalized state becomes the **start-state** when `init` is called again with the same name

---

## State Provider Interface

All providers implement the same logical operations:

```yaml
StateProvider:
  init:
    input: { evolution_name: string, domain: string, goals: array }
    output: { state: EvolutionState, is_new: boolean }
  
  get:
    input: { evolution_name: string }
    output: { state: EvolutionState | null }
  
  checkpoint:
    input: { evolution_name: string, event: HookEvent }
    output: { checkpoint_id: string }
  
  finalize:
    input: { evolution_name: string }
    output: { archived_path: string }
  
  list:
    input: {}
    output: { evolutions: [{ name, domain, status, updated_at }] }
  
  save:
    input: { evolution_name: string, state: EvolutionState }
    output: { success: boolean }
```

See `references/schemas/state-provider.schema.json` for the provider configuration schema.

---

## Provider Resolution Protocol

The skill resolves a state provider at startup using a 6-tier waterfall:

### Tier 1: Environment Variable
```bash
$EVOLVER_PROVIDER_CONFIG → path to provider config JSON
```

### Tier 2: Project-local Config
```
.evolver-provider.json   (in current working directory)
```

### Tier 3: Global Config
```
~/.evolver/provider.json
```

### Tier 4: MCP State Tool
Probe for an MCP server that provides state management tools (e.g., a SurrealDB-backed state server). Check if a tool named `evolution_state_*` or matching the `mcp_tool` provider pattern exists.

### Tier 5: Agent Memory
Probe for the `memory` MCP server (e.g., `surreal_memory`). If available, use it to store/retrieve evolution state as entities with the evolution name.

### Tier 6: Filesystem Fallback (always available)
```
.evolver/                     # Project-local state directory
  registry.json               # Maps evolution_name → state path
  evolutions/
    {evolution_name}/
      state.json              # Current evolution state
      checkpoints/            # Mid-session snapshots
        {timestamp}.json
      history/                # Finalized iteration archives
        iteration-{N}.json
```

The resolution script is `scripts/state-resolve-provider.sh`.

---

## Provider Configuration

Provider config follows `references/schemas/state-provider.schema.json`:

```json
{
  "provider_type": "filesystem",
  "config": {
    "state_directory": ".evolver",
    "scope": "project"
  }
}
```

### Filesystem Provider
```json
{
  "provider_type": "filesystem",
  "config": {
    "state_directory": ".evolver",
    "scope": "project | global"
  }
}
```

### Agent Memory Provider
```json
{
  "provider_type": "agent_memory",
  "config": {
    "entity_prefix": "evolution",
    "memory_server": "surreal_memory"
  }
}
```

### MCP Tool Provider
```json
{
  "provider_type": "mcp_tool",
  "config": {
    "server_name": "evolution-state-server",
    "init_tool": "evolution_init",
    "get_tool": "evolution_get",
    "save_tool": "evolution_save",
    "list_tool": "evolution_list"
  }
}
```

### Custom Provider
```json
{
  "provider_type": "custom",
  "config": {
    "script": "/path/to/custom-provider.sh",
    "args": ["--format", "json"]
  }
}
```

---

## Cross-Session State Recovery

When a new session starts:

1. Run provider resolution
2. Call `list()` to show available evolutions
3. User provides `evolution_name` (or skill auto-detects from context)
4. Call `init(name)` which:
   - If active state exists → resume from last checkpoint
   - If finalized state exists → load as new start-state
   - If nothing exists → create fresh state
5. Proceed with PMPO loop

This pattern ensures evolution state survives:
- Context window limits
- Session resets
- Model swaps
- Multi-agent handoffs
- Machine restarts
