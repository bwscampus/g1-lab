# The robot's knowledge base

What the robot knows in a `g1 start` session, kept as an AI agent keeps it (`g1/knowledge.py`).
Everything here except this README and `SOUL.example.md` belongs to one robot and is git-ignored.

| file | what it is | who writes it |
|---|---|---|
| `SOUL.md` | who the robot is: personality, how it talks, where it is, what it cares about | people only (an editor); made from `SOUL.example.md` on the first `g1 start` |
| `MEMORY.md` | what it has learned: one dated bullet per fact | the robot (the model's `remember` tool, and the distill at the end of each session); people may edit it |
| `sessions/DATE.md` | that day's log: one line per exchange (heard or typed, and what the robot did) | the session |

How they are used:

- Every turn, the model's system prompt is the session's rules (in `g1/session.py`, `MECHANICS`: the
  tools and the safety rules, which no file here can remove), then `SOUL.md`, then `MEMORY.md` (its
  newest bullets, up to `memory_max_chars`). The files are read afresh each turn.
- When the session ends (after the robot is at neutral), the model reads the session's log and
  updates `MEMORY.md`: it adds what is worth keeping and removes what is wrong or outdated. Esc skips it.
- `/memory` and `/soul` in `g1 start` show them. Only the session reads them: tasks never do.
