# Agent-run records

The records of the 27 weight-space programming runs of the paper (Sec. 3.3, App. C and D): three variants x three
model sizes x three runs, all with Claude Opus 5.5 through Claude Code 2.1.316, started between 14 and 17 November 2026.

```
runs/<variant>/<size>/run<k>/      variant: flaw_informed | target_informed | physics_origin_informed
                                   size: small | medium | large (OmniLearned-s/m/l); k = 1, 2, 3 by start time
runs/index.csv                     one row per run: start/end time, queries, cost, selected edit, checksums
runs/excluded.md                   runs that were started but are not among the 27
```

## Files of one run

| file | content |
|---|---|
| `run.json` | variant, size, model and agent version; sha256 of the task description (`task_description.sha256`, the frozen text in `agents/task_descriptions/`), of the filled task description as sent (`prompt_sha256`) and of the launch message (`launch_sha256`), all recorded at launch; start and end time; queries used; cost in USD and the result lines of the transcript; token totals; the selected edit; name, size and sha256 of every handed-in edit file; dropped files; audit counts; events |
| `task_description.md` | the task description as the agent received it (`PROMPT.md`), paths replaced by placeholders |
| `launch_message.txt` | the message that started the agent (`agents/launch_messages.md`), paths replaced |
| `evaluation_tool_config.json` | the run's configuration of `agents/evaluation_tool.py` (older key names: `flaw_task` = flaw informed, `ref_task` = target informed, `rule` = physics origin informed) |
| `transcript.jsonl.gz` | the full Claude Code transcript (stream-json): every message, tool call and tool output |
| `notes/zero_pass_notes.md` | the agent's plan and predictions before its first query |
| `notes/final_notes.md` | the agent's final note: the edits, the mechanism it believes it found, failed ideas, predictions against outcomes |
| `ledger.json` | the evaluation tool's log: every served query with its hypothesis, prediction, kind and time |
| `requests/<id>.json` | every request the agent wrote |
| `responses/<id>.json` | every reply of the evaluation tool; the binary arrays it returned (`<id>.npz`: logits, activations, target predictions) are not published and are listed in `run.json` (`dropped`) |
| `scores/edit_<size>.remaining_error.json` | score of a handed-in edit on the test jets with the removal-rule scorer: remaining error against the target predictions, the limits, responses to added particles and physical changes, changed parameters (`.recheck` = a repeated scoring; `zero_pass.*` = the edit handed in before the first query, where scored) |
| `scores/edit_<size>.remaining_response.json` | score of the same edit with the remaining-response scorer |
| `audit/access_audit.json` | output of `agents/audit/access_audit.py`: every tool call that names a path outside the run's two folders, every listing or search, machine-state commands and the interpreter-hook rows |
| `audit/interpreter_hook.jsonl` | the log of `agents/audit/sitecustomize.py` (only where it logged an event) |
| `audit/evaluation_tool.log` | the evaluation tool's job output |
| `audit/start.log`, `audit/resume.log`, `audit/preflight.txt`, `audit/preflight.jsonl` | start of the agent once the tool was up; the resumed session (one run); the start check (two runs) |

`edit_small`, `edit_medium` and `edit_large` are the up to three final edits of different sizes the agent handed in
(files `final_small.npz`, `final_mid.npz`, `final_large.npz` in the run folder), `zero_pass` the edit handed in before
the first query. The edit files are on Zenodo as `edits/agents/<variant>/<size>/run<k>/{edit_small,edit_medium,
edit_large,edit_zero_pass}.npz` (and `extra_candidate_*.npz` for the further candidates of physics origin informed
large run 2); `run.json` gives both the Zenodo sha256 and the sha256 of the file as handed in (they differ only for
two zero-pass files that were trimmed on Zenodo). The selected edit of a run is the one with the lowest remaining error among those that pass the
accuracy, class, NLL and physical-response limits (App. C).

The JSON keys inside the records are those of the code that produced them: `E_rule` / `E_ref` = remaining error,
`R_vis` = remaining response, `rho` = response to physical changes (`P_reco` / `P_ref`: fragmentation after the
removal rule), `dof` = changed parameters, `gate_rule_no_g2` = all limits except the generator; the task descriptions
call the evaluation tool "the oracle".

## Placeholders

Paths, user and host names, accounts and job ids are replaced by placeholders (`agents/scrub_records.py`, the same
placeholders as `agents/make_public.py`, which lists their meaning): `{RUN}` and `{INPUTS}` are the run's own folders,
`{PYTHON}` the agent's interpreter, `{RUNS_DIR}/<variant>-<size>-run<k>` the folders of the other runs, `{USER}`,
`{NODE}`, `{JOB}` and so on. Nothing else in the records is changed.
