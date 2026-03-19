# Nemotron Research Reference

Research reference repo for conceptual synthesis. Not a development project — do not modify upstream code.

## Workspaces

- /src/nemotron/recipes/ — Training pipelines (pruning, distillation, SFT, alignment)
- /usage-cookbook/ — Deployment and model usage guides (notebooks)
- /use-case-examples/ — Agentic workflow examples (RAG, tool use, agents)
- /reference/ — Cloned reference repos
- /autoresearch/ — Our research findings

## Routing

| Task | Go to | Read |
|------|-------|------|
| Explore training patterns | /src/nemotron/recipes/ | context.md |
| Explore agentic workflows | /use-case-examples/ | context.md |
| Explore deployment | /usage-cookbook/ | context.md |
| Save autoresearch findings | /autoresearch/ | context.md |

## Conventions

- All findings go in `autoresearch/YYYY-MM-DD-<slug>.md`
- Synthesize, don't summarize — new connections and transferable ideas, not repo summaries
- Read `context.md` before every research session for current exploration threads

## Avoid

- Do not modify any upstream repo files (src/, usage-cookbook/, use-case-examples/, etc.)
- Do not treat this as a development project — no implementation work here
- Do not create findings without the `YYYY-MM-DD-<slug>.md` naming convention
