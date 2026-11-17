# harness_accept.py: deleted or planted workspaces, and `run` stops its servers on every exit

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

## Review fixes (shady 8/10 SHIP, comment 6949228150)
- **A second signal during cleanup** (SHOULD 1). `signal_exit` stayed installed while the finally ran. A second
  SIGTERM in `Proxy.stop`'s up-to-0.5 s wait escaped, so the engine stayed up and the handlers weren't restored.
  Shady's probe `double.py` showed this at gaps of 0.05, 0.2 and 0.4 s. The finally now ignores SIGTERM, SIGHUP and
  SIGINT until it's done, stops the engine first (it holds the VRAM), then restores the handlers. This is what
  axon-run's runner does. With the fix, `double.py` shows the engine stopped and the handlers restored at every gap
  (0 to 3 s).
- **Paths an agent planted** (SHOULD 2). The arm's directory is the sandbox's rw bind, so an agent can put a symlink at
  a later task's workspace. `rmtree(ignore_errors=True)` silently refused it, `build_base` followed it, and
  `git_init` ran `git add -A` in the target, which is the trusted side writing outside the sandbox. A plain file there
  crashed the run (`FileExistsError`). The turn's transcript, `<arm>/r<n>-<task>-t<i>.out`, had the same hole:
  `open(..., "w")` wrote through a planted symlink. Shady's review didn't list that one. `clear_path` now removes
  whatever is at those paths, and at `<ws>.check`, without following it.
- **nohup** (a regression in 4f500a4, found while fixing SHOULD 1). `cmd_run` turned SIGHUP into an exit even when
  the run was started ignoring it, so a `nohup`'d run would die on logout. An inherited SIG_IGN is now kept.
- **Nit:** the selfcheck matches the whole `sleep` argument (`sleep\0<tag>\0`).

New selfcheck cases:
- **`run: 2nd signal during cleanup`:** the engine's and the proxy's `stop()` each first send TERM, HUP and INT to
  the process, so the case doesn't depend on timing.
- **`run: SIGHUP under nohup`.**
- **`run: symlink planted at a ws`:** a symlink at round 2's workspace and transcript, pointing outside.
- **`run: file planted at a ws`.**

A server `cmd_run` leaves in `LIVE` now fails its case, and is stopped before the next case.

Each fix reverted on its own (locally) fails exactly its own case:
- without the SIG_IGN, or with the proxy stopped first: the second-signal case;
- without the nohup guard: nohup (SystemExit 129);
- with the old `rmtree` reset: both planted cases (the repo and `.git` written into the outside directory; the file
  case crashes);
- without clearing the transcript path: the symlink case (the outside file overwritten).

The full selfcheck passes on jade at e280d1c (job 1117-083539-be4d, sandboxed).

## Not changed
The sandbox still binds the whole arm directory. An agent could also reach the arm's other workspaces, round 1's
included while round 2 runs. None of the 587 task transcripts on bella (`r*-t*.out`, 131 of them round 2) names
another task's workspace; 277 name their own. It could also delete the harness config in HOME, which fails the arm's
later tasks (SHOULD 3). #277 tracks binding only the task's workspace and HOME, a config detector, and the selfcheck
in CI. The lead's ruling: after #274's runs finish, in its own PR, so those runs keep the conditions their earlier
rows had.
