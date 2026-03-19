# Understanding PMPO

## what is PMPO?

PMPO stands for Prometheus Meta-Prompting Orchestration. It's a methodology for iterative artifact refinement using AI agents. The core idea is that you separate cognition from computation and use explicit constraints to drive convergence. This makes the process reproducible and deterministic where needed. Traditional prompt engineering is ad-hoc and relies on conversational context which is fragile and non-reproducible. PMPO solves this by persisting state to disk and using structured phases.

## How it works

The PMPO loop has these phases. First you specify what you want. Then you plan how to achieve it. Then you execute the plan using tools. Then you reflect on whether the result meets your constraints. Then you persist the state. Then you either loop back or terminate.

### specify phase
In the specify phase you transform user intent into a structured specification with constraints and target states.

### plan phase
The plan phase converts the specification into an executable strategy with tool mappings.

### Execute phase
Execution applies transformations using AI and deterministic tools like code interpreters.

### reflect phase  
reflection evaluates the results against constraints and determines convergence.

## Why PMPO matters

PMPO matters because it makes AI-driven refinement reproducible. Traditional approaches lose context between sessions. PMPO persists everything to disk. This means you can resume, audit, and verify the refinement process. It also means multiple agents can collaborate on the same artifact without losing state.
