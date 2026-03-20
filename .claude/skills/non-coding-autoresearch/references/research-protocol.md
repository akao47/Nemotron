# Research Protocol

Detailed protocol for the autonomous research loop. The SKILL.md defines WHAT to do; this file defines HOW to do it well.

## Pre-Session Checklist

Before starting any research session:

- [ ] Read `context.md` — know the active research threads
- [ ] Read all existing findings in `autoresearch/` — know what's covered
- [ ] Read `autoresearch/sessions.tsv` — know the session history
- [ ] Check for a research brief in `autoresearch/briefs/` — if one exists from `/non-coding-autoresearch:plan`, use it as starting context
- [ ] Verify the current date for filename conventions

## Iteration Protocol (Detailed)

### Step 1: READ

Every iteration starts with re-reading. This prevents drift and repetition.

**Minimum reads per iteration:**
- `context.md` (skim — focus on thread you're pursuing)
- Titles of existing findings (avoid covering same ground)
- At least 3 source files relevant to your chosen thread

**What to look for when reading source files:**
- Patterns that repeat across files — these are design principles
- Things that seem surprising or counter-intuitive — these are insights
- Connections to other parts of the codebase — these are synthesis opportunities
- Comments, TODOs, and design decisions — these reveal intent
- Differences from standard approaches — these are the interesting bits

### Step 2: PICK

State your research question explicitly. It should be:
- **Specific**: "How does the Nemotron SFT recipe handle multi-turn conversation formatting?" not "What does SFT do?"
- **Answerable from the source material**: Don't ask questions that require information you can't access
- **Non-obvious**: If the answer is in the README, it's not a research question
- **Connected**: Prefer questions that bridge two parts of the codebase or connect to external patterns

**Thread selection strategy:**
- **Early in session**: Follow threads from context.md or the research brief
- **Mid-session**: Follow open threads raised by previous findings
- **Late in session**: Try cross-cutting questions that connect multiple threads
- **When stuck**: Pick two random source files and ask "what connects these?"

### Step 3: EXPLORE

This is the most time-intensive step. Be thorough.

**Local exploration:**
- Read the files directly relevant to your question (minimum 3-5)
- Grep for related patterns, function names, or concepts
- Look at directory structure for organizational patterns
- Check config files for hidden assumptions
- Read tests or examples for usage patterns

**Web search (when needed):**
- Search for official documentation on techniques or tools mentioned in the code
- Look up papers referenced in comments or configs
- Check official repos for related implementations
- Find official best practices for patterns you observe

**Cross-reference:**
- Compare approaches across different recipes/examples
- Look for the same concept implemented differently in different places
- Check how reference repos handle the same problem

### Step 4: SYNTHESIZE

Draft the finding before writing to disk. The draft should:

1. **Lead with the insight** — the first paragraph should state the new connection or claim
2. **Show the evidence** — specific files, line numbers, code patterns that support it
3. **Explain transferability** — why this matters beyond this specific codebase
4. **Open a thread** — what question does this answer raise?

**Writing the draft:**
- Write as if explaining to a technically competent reader who hasn't read the source
- Use specific references (file paths, function names, config values)
- Avoid filler phrases: "interestingly," "it's worth noting," "upon further examination"
- Make the title a claim, not a topic: "Nemotron's Loss Masking Creates an Implicit Curriculum" not "About Loss Masking in Nemotron"

### Step 5: QUALITY CHECK

Run through every item. Be honest. A discarded finding is better than a weak one.

1. **Novelty**: Does this say something that isn't in the source material?
2. **Connection**: Does it link two or more ideas that aren't linked in the source?
3. **Transferability**: Could someone apply this insight in a different context?
4. **Surprise**: Would someone who read the source learn something new from this?
5. **Specificity**: Does it make concrete claims with evidence, not vague observations?

**Scoring**: The finding must pass at least 4 of 5 checks. If it passes 3, attempt a revision. If it passes 2 or fewer, discard.

### Step 6: WRITE or DISCARD

**If writing:**
- Create `autoresearch/YYYY-MM-DD-<slug>.md` with today's date
- The slug should be descriptive: `pruning-as-architecture-search` not `finding-3`
- Follow the finding template from SKILL.md

**If discarding:**
- Note why in sessions.tsv
- Consider: Was the question bad? Was the exploration insufficient? Is there a better angle?
- Use the failure to inform your next iteration

### Step 7: COMMIT

```
git add autoresearch/YYYY-MM-DD-<slug>.md
git commit -m "research: <slug>"
```

Commit message format: `research: <slug-description>`

Examples:
- `research: pruning-as-architecture-search`
- `research: sft-loss-masking-implicit-curriculum`
- `research: voice-rag-multi-model-composition`

### Step 8: LOG

Append to `autoresearch/sessions.tsv`:
```
YYYY-MM-DD	N	slug	keep|discard	one-line summary
```

### Step 9: CONTINUE

Before starting the next iteration:
- Note any open threads the current finding raised
- Check if the research brief suggests a next direction
- Decide whether to go deeper on the current thread or branch out

## Recovery Procedures

### Stuck: No ideas
1. Re-read context.md for threads you haven't explored
2. Pick two unrelated source directories and look for connections
3. Re-read your own previous findings — what open threads did they raise?
4. Search the codebase for a keyword you haven't explored yet
5. Look at the reference repos for patterns not present in the main codebase

### Stuck: Finding keeps failing quality check
1. You're probably not going deep enough — read more source files
2. Try narrowing the question — a specific claim about one file is better than a vague claim about a directory
3. Try the opposite angle — if you keep finding what you expect, look for what contradicts your expectation
4. Search official docs for context you're missing — maybe the pattern makes more sense with external context

### Stuck: Not sure if a finding is synthesis or summary
Ask yourself: **Could someone generate this finding by running a recursive grep and reading the README?**
- If yes → it's a summary. Discard.
- If no → it's at least partially synthesis. Check the other quality criteria.

## Session Pacing

- **Short sessions (3-5 iterations)**: Focus on one thread, go deep. Each finding should build on the previous one.
- **Medium sessions (5-10 iterations)**: Start with one thread, let open threads guide you to adjacent territory.
- **Long sessions (10+ or indefinite)**: Plan to shift threads every 3-4 iterations. Avoid diminishing returns on a single topic. Use cross-cutting questions to find connections between threads.
