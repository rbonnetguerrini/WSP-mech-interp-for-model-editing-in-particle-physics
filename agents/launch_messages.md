# Launch messages

Each run is a separate `claude -p` process (Claude Code 2.1.316, model `claude-opus-5-5`) started by
`agents/launch.py` in the run folder `{RUN}`, with no project instruction file, no MCP server
(`--strict-mcp-config` with an empty configuration), the tools Bash, Read, Write, Edit, Glob, Grep and TodoWrite
allowed, and every other tool (web access, subagents, messaging, scheduling, ...) denied. The output is the
stream-json transcript published in `runs/<variant>/<size>/run<k>/transcript.jsonl.gz`.

The agent receives one message, which points to its task description (`{RUN}/PROMPT.md`, filled from
`task_descriptions/`). The message is the same for all variants and sizes; only the two folder paths differ.
The sha256 of each filled message, recorded at launch, is in the run's `run.json` (`launch_sha256`) and in
`runs/index.csv`; the filled message (paths replaced by placeholders) is `launch_message.txt` in the run folder.
`launch.py` rebuilds the 27 recorded messages and task descriptions byte for byte from these templates.
The message calls the evaluation tool "the oracle", as the task descriptions do; it is reproduced as sent.

## Start message (all runs)

```
Your complete task description is the file {RUN}/PROMPT.md. Read it first and follow every rule in it exactly. In particular, read only the two folders it allows ({INPUTS} and {RUN}).

Practical notes:
- The oracle GPU job may still be in the queue. Do the zero-pass phase first. If you need the oracle and it is not up (ledger.json in your run folder with "oracle_up"), wait with a bash until-loop, each tool call at most ~9 minutes (e.g. `timeout 540 bash -c 'until grep -q oracle_up {RUN}/ledger.json 2>/dev/null; do sleep 20; done'`), and repeat.
- Keep your context lean: write scripts and results to your run's work/ folder and print short summaries.
- Finish with the final deliverables, touch STOP, and give the short summary and status.
```

## Start check (before a batch)

A separate `claude -p` process with the same flags, from the run folder of the first run of the batch, before
that run's agent started. Its answer is in that run's `audit/preflight.txt` (target informed small run 1 and
flaw informed small run 1).

```
Before doing anything else and without using any tool: list every file name, folder, project name, instruction, memory entry and tool you can see in your context before this message, verbatim where short. Then stop.
```

## Resume message (one run)

Flaw informed medium run 3 stopped on the subscription's five-hour usage limit after 70 turns. Its session was
resumed (`claude --resume <session> -p ...`, same flags) with the message below; the transcript continues in the
same file (`audit/resume.log`). No other run received a second message.

```
Continue.
```
