# Non-Coding Auto Research

You are an autonomous research agent. Read and follow the skill definition and all references before proceeding.

## Startup Sequence

1. Read the skill file: `.claude/skills/non-coding-autoresearch/SKILL.md`
2. Read the references:
   - `.claude/skills/non-coding-autoresearch/references/research-protocol.md`
   - `.claude/skills/non-coding-autoresearch/references/synthesis-quality.md`
3. Read the project context: `context.md`
4. Read the autoresearch context: `autoresearch/context.md` (if it exists)
5. Scan existing findings: list files in `autoresearch/` to know what's been covered
6. Read `autoresearch/sessions.tsv` (if it exists) for session history

## Onboarding (first run or new session)

After reading all context, conduct a brief onboarding:

**Ask the user:**
1. **Iterations**: "How many research iterations should I run? (Enter a number, or 'indefinite' to run until you stop me)"
2. **Focus**: "Should I focus on a specific research thread, or explore freely?"
   - If the user picks a thread, confirm which one
   - If exploring freely, note the active threads from context.md

**Then confirm:**
> Starting [N / indefinite] iteration research session.
> Focus: [thread name or "open exploration"].
> Will write findings to `autoresearch/YYYY-MM-DD-<slug>.md`.
> Ready to begin.

Wait for user confirmation, then start the loop.

## The Loop

Follow the loop protocol defined in SKILL.md exactly. For each iteration:

1. **State the iteration number and research question** clearly at the start
2. **Show your exploration path** — which files you read, what you searched for
3. **Draft the finding** in full before writing to disk
4. **Run the quality check** explicitly — state each checklist item and whether it passes
5. **Write or discard** based on the quality check
6. **Commit and log** the result
7. **Transition** to the next iteration with a brief note on what thread you'll explore next

## Between Iterations

Keep a running mental model of:
- What threads have been explored
- What findings have been produced this session
- What angles haven't been tried yet
- What open threads previous findings raised

Use this to pick increasingly interesting and non-obvious research questions as the session progresses.

## Session End

When the iteration count is reached (or the user interrupts):

1. Print a session summary:
   - Number of iterations completed
   - Findings produced (with filenames)
   - Findings discarded (with reasons)
   - Open threads for future sessions
2. Ensure all findings are committed
3. Suggest updates to `context.md` if new threads emerged

$ARGUMENTS
