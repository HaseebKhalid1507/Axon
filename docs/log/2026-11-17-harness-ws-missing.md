# harness_accept.py: a deleted workspace fails the task; `run` stops its servers on every exit

- Date: 2026-11-17. Branch: fix/harness-ws-missing, off main 110a17b. Scripts only, no engine code.

## What happened
Job 1117-042150-6ccd (#274's opencode harness, Qwen2.5-7B, okarin's copy of this script with two extra arms) crashed
after 41 results:
- In axon-off's round-1 `exception` task, opencode's model decided to "recreate the working directory". It ran
  `rm -rf <its workspace>` through opencode's bash tool, then tried to `git clone <path-to-your-repo>` into it
  (`r1-exception-t0.out`).
- So the workspace was deleted by the agent. The harness did create it, and no cleanup raced it.
- The sandbox binds the arm's directory (`axon-off-opencode/`) read-write, so the agent could remove the workspace
  directory itself, not just its contents.
- `run_check`'s `copytree` then raised FileNotFoundError, which ended the run.
- Nothing stopped the servers on that path. axon serve kept 8.7 GB of VRAM until axon-run swept the job's scope.

## What changed (`scripts/harness_accept.py`)
- **`run_check`:** a missing workspace fails the task with detail `workspace missing`. A workspace that is a symlink
  or a file, or that can't be copied (a named pipe, an unreadable file), fails it as `workspace broken: ...`. A
  symlinked workspace is no longer followed: the old code would copy wherever it pointed.
- **`cmd_run`:** now wraps the old body (`run_all`) in try/finally, which stops every engine and proxy still up (the
  `LIVE` list) on any way out.
  - SIGTERM and SIGHUP raise SystemExit, so the finally runs, and the turn in flight kills its harness's process
    group.
  - SIGKILL is still left to axon-run's sweep.
- **`harness_version`:** split out of `cmd_run` so the selfcheck can stub it. Same behaviour.
- **The module docstring** says what the read-write bind exposes.

## How it was verified
`selfcheck_exits`, in `selfcheck`:
- `run_check` on 4 broken workspaces: missing, a symlink to a directory outside, a file, and a named pipe inside.
- `run` against a fake engine (a small HTTP server standing in for `axon serve`), with the harness swapped for a
  shell command. Three cases:
  - the agent deletes its workspace: a failed result and rc 0;
  - an exception at 6ccd's crash site: it propagates;
  - SIGTERM mid-turn: SystemExit(143), and the harness is gone.

  In each case the engine's process and the proxy's port must be gone afterwards.

Each fix reverted on its own fails its own case, run locally:
- without the guard: missing / symlink / file;
- without the copy catch: the named pipe (it crashes);
- without the finally: exception / SIGTERM, with the servers left up;
- without the turn kill: SIGTERM, with the harness left running;
- without the signal handler: the selfcheck itself dies on SIGTERM and leaves its fake engine behind.

The full selfcheck passes on jade at 4f500a4 (job 1117-080803-c2c4, sandboxed). okarin's patched copy passes too
(1117-081128-c7a4).

## Not changed
The sandbox still binds the whole arm directory, so an agent could also reach the arm's other workspaces, round 1's
included while round 2 runs. None of the 587 task transcripts on bella (`r*-t*.out`, 131 of them round 2) names
another task's workspace; 277 name their own. Binding only the task's workspace and HOME would close it. It's left out here, so that the runs in flight
(#274) keep the conditions their earlier rows had.
