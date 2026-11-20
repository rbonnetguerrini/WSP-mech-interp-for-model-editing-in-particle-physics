# Runs started but not among the 27

| run | started | reason |
|---|---|---|
| physics origin informed, small, stopped 1 | 2026-11-15 07:08 | Stopped after about 15 minutes by the subscription's five-hour usage limit (shared by all runs of the batch), before any final edit; 4 queries used. Not scored. Replaced by a fresh run (small run 2 or 3). |
| physics origin informed, small, stopped 2 | 2026-11-15 07:10 | Same usage-limit stop; 3 queries used. Not scored. Replaced by a fresh run. |
| physics origin informed, medium, stopped 1 | 2026-11-15 07:09 | Same usage-limit stop; 0 queries used. Not scored. Replaced by a fresh run (medium run 2 or 3). |
| physics origin informed, medium, stopped 2 | 2026-11-15 07:09 | Same usage-limit stop; 2 queries used. Not scored. Replaced by a fresh run. |
| physics origin informed, large, stopped 1 | 2026-11-15 07:09 | Same usage-limit stop; 0 queries used. Not scored. Replaced by a fresh run (large run 2 or 3). |
| physics origin informed, large, stopped 2 | 2026-11-15 07:10 | Same usage-limit stop; 0 queries used. Not scored. Replaced by a fresh run. |
| evaluation-tool tests, physics origin informed (small, medium, large) | before the first batch | Scripted requests to check the evaluation tool on each model size; no agent. |
| evaluation-tool test, target informed | before the first batch | Scripted requests (including the kind `reference`); no agent. |
| evaluation-tool test, flaw informed | before the first batch | Scripted requests (checks that no remaining error is returned); no agent. |

The six stopped runs used the same frozen task description as the runs that replaced them. They were stopped by
an infrastructure limit, not by the access audit; no run was discarded for an access outside its folder. One
flaw informed run (medium run 3) hit the same limit later and was resumed in the same session instead (see its
`run.json`).
