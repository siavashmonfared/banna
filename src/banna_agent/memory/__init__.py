"""Memory subsystem: semantic + episodic + procedural memory.

Memory is a sidecar to the driver — the agent loop never reads memory
directly. Policies and tools consume it. This is what makes branching
(week 2 best-first) and ablation (on/off) clean."""
