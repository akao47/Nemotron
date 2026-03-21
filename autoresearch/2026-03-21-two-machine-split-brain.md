# The Split-Brain Architecture: Why Two Cheap Machines Beats One Expensive One

## The Insight

The obvious path is: buy one powerful machine that does everything. A Mac Mini M4 Pro 48GB ($1,600) or a PC with 64GB RAM + RTX 4090 ($2,500+) could run the micro, nano, and 14B model simultaneously. Why split across two machines?

Because the two use cases have **opposite uptime requirements**, and optimizing for both on one machine means compromising both.

The sidecar (micro + nano + embeddings) needs to be **always-on, silent, low-power, and invisible**. It runs 24/7, captures every Claude Code session, updates embeddings when vault files change, and is ready to inject context in <100ms. It's infrastructure — like your router or NAS. You never think about it until it stops working.

The link-to-learning pipeline needs to be **powerful, on-demand, and interruptible**. It spins up a 14B model, processes a URL for 5-15 minutes, generates a guide, and shuts down. It runs maybe 2-3 times a week. It doesn't need to be on at 3am.

A Mac Mini M4 at 3-5 watts idle is the perfect always-on sidecar. A PC with 64GB RAM is the perfect on-demand workhorse. Together, they cost less than one premium machine and serve each role better.

The architectural insight goes deeper: **separation of concerns isn't just a software pattern — it applies to hardware**. The Mac Mini is the "state layer" (persistent memory, always-available context). The PC is the "compute layer" (heavyweight processing, batch generation). They share state through the Obsidian vault (synced via Syncthing) and coordinate through Ollama's REST API over the local network.

## Evidence

**Power consumption proves the always-on case:**

| Machine | Idle power | Under load | Annual cost (24/7, $0.12/kWh) |
|---------|-----------|------------|------------------------------|
| Mac Mini M4 16GB | 3-5W | 40-45W | **$3-5/year** |
| PC 64GB (no GPU) | 40-80W | 150-250W | **$42-84/year** |
| PC 64GB + RTX 4090 | 80-120W | 400-600W | **$84-126/year** |
| Mac Mini M4 Pro 48GB | 5-8W | 70-140W | **$5-8/year** |

Running the Mac Mini 24/7 costs less than $5/year in electricity. Running the PC 24/7 would cost $42-84/year — and it's wasted energy because the 14B model is used maybe 10 hours/month. Jeff Geerling's testing confirmed the M4 Mini idles at 3 watts — comparable to a Raspberry Pi, but with 100x the computing capability.

**The communication layer is already solved:**

1. **Vault sync (Syncthing)**: Free, peer-to-peer, real-time file synchronization between Mac and PC. Obsidian vaults are just folders of markdown files — Syncthing handles them perfectly. The Obsidian Syncthing Integration plugin adds conflict resolution UI. Setup: install on both machines, share the vault folder, done. No cloud, no subscription, no data leaves your network.

2. **Model API (Ollama)**: Set `OLLAMA_HOST=0.0.0.0` on the PC. The Mac Mini calls `http://pc-ip:11434/api/generate` to invoke the 14B model remotely. Ollama's API is OpenAI-compatible, so any tool that works with ChatGPT's API works with your PC's Ollama instance. No code needed — just a URL change.

3. **Orchestration**: The micro on the Mac Mini acts as the router. When it classifies a request as "link-to-learning" (needs 14B), it sends the scraped content + vault context to the PC's Ollama API. The 14B generates the guide. The result comes back as text, gets saved to the vault, Syncthing propagates it, and the Mac Mini's embedding index updates automatically.

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│     MAC MINI M4 16GB        │     │      PC 64GB RAM            │
│     (always-on, 3-5W)       │     │      (on-demand)            │
│                             │     │                             │
│  ┌─────────────────────┐    │     │  ┌─────────────────────┐   │
│  │ SmolLM2-135M (micro)│    │     │  │ 14B model (Ollama)  │   │
│  │ • routing            │    │     │  │ • link-to-learning  │   │
│  │ • classification     │────────────│ • deep comprehension│   │
│  │ • daily LoRA retrain │    │ API │  │ • guide generation  │   │
│  └─────────────────────┘    │     │  └─────────────────────┘   │
│                             │     │                             │
│  ┌─────────────────────┐    │     │                             │
│  │ Phi-4-mini (nano)   │    │     │                             │
│  │ • synthesis          │    │     │                             │
│  │ • session summaries  │    │     │                             │
│  │ • plan review        │    │     │                             │
│  └─────────────────────┘    │     │                             │
│                             │     │                             │
│  ┌─────────────────────┐    │     │                             │
│  │ nomic-embed-text    │    │     │                             │
│  │ • embeddings         │    │     │                             │
│  └─────────────────────┘    │     │                             │
│                             │     │                             │
│  ┌─────────────────────┐    │     │  ┌─────────────────────┐   │
│  │   OBSIDIAN VAULT    │◄──Syncthing──►  OBSIDIAN VAULT    │   │
│  │   (primary copy)    │    │     │  │   (synced copy)     │   │
│  └─────────────────────┘    │     │  └─────────────────────┘   │
└─────────────────────────────┘     └─────────────────────────────┘
```

**The upgrade path is clean:**

| When | What changes | Cost |
|------|-------------|------|
| Now | Mac Mini + PC (CPU only) | $0 (have Mac) + $500-800 (PC) |
| Later: want faster link-to-learning | Add RTX 3090 to PC | +$700 used |
| Later: want to run 70B models | Add RTX 4090 or second 3090 | +$800-1,600 |
| Later: want Nemotron Ultra locally | Replace PC with multi-GPU rig | $3,000-5,000 |
| Never changes | Mac Mini stays as-is, always-on sidecar | $0 |

The Mac Mini never gets replaced or upgraded. It's the stable anchor. The PC is the upgradeable component. This separation means you never risk your always-on sidecar when upgrading or experimenting with the compute layer.

## Why This Matters

The two-machine architecture embodies a principle from distributed systems: **separate your state from your compute**. The vault (state) lives on the Mac Mini and syncs everywhere. The models (compute) run on whatever hardware is appropriate — tiny on the Mac, heavy on the PC, cloud on RunPod.

This has three practical benefits the user cares about:

1. **The sidecar never goes down.** The Mac Mini runs 24/7 at 3-5W. It doesn't restart for GPU driver updates. It doesn't thermal throttle from a 14B model. It doesn't run out of RAM because you loaded a big model. Claude Code's hooks always have a responsive sidecar — the micro answers in <50ms, every time.

2. **The PC is guilt-free.** You don't leave a 250W PC running 24/7 for a model you use twice a week. You boot it when you have a new tool to learn, run the pipeline, and shut it down. Or leave it sleeping and wake it via Wake-on-LAN from the Mac Mini when the micro detects a link-to-learning request.

3. **Total hardware cost is lower.** Mac Mini M4 16GB: $600 (already owned). A basic PC with 64GB DDR5 RAM, a modern i5/Ryzen 5, and no GPU: $500-800. Total: $1,100-1,400 for both. Compare to a single Mac Mini M4 Pro 48GB ($1,600) or Mac Studio M4 Max 64GB ($3,000) — which still can't run 14B alongside the full sidecar stack without memory pressure, and burns more power 24/7.

## Open Thread

Wake-on-LAN automation could make the two-machine split feel seamless. The micro on the Mac Mini detects a link-to-learning request, sends a WoL magic packet to the PC, waits for it to boot (~30 seconds), sends the request to the PC's Ollama API, gets the result, and the PC goes back to sleep. The user never has to manually wake the PC — they just drop a link and the guide appears in their vault 10-20 minutes later.

The deeper question: as Apple Silicon gets more RAM per dollar (M5, M6 generation), does the two-machine architecture eventually collapse into one? Possibly — if a $600 Mac Mini ships with 48GB in 2-3 years, you could run the full stack (micro + nano + 14B) on one machine. But even then, the always-on power advantage of a dedicated sidecar machine may still justify the split. And the PC's upgradeability (add a GPU, swap RAM) offers flexibility that a sealed Mac never will.
