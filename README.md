# gate-logs

Logs of Axon's GPU gate jobs, published by `axon-gatelogs` (axon-devstack) when a gate set finishes, so every
MEASURED/PASS claim in a PR links to its evidence. This branch holds no code and is never merged.

One directory per gate set: `<YYYY-MM>/<label>/`
- `MANIFEST.md` / `manifest.json`: job id, name, verdict/rc, host, GPU, driver, the sha each job tested, timings
- `<job id>.log`: the job's full log from bella's queue (`axon-run`), scrubbed of secrets before upload
- `hw-<host>.txt`: nvidia-smi / CPU record of each host (pods: captured at pod setup; bella/avante/jade: at upload)

The jobs ran on private machines (bella, avante, jade) and RunPod pods through axon-devstack's queue; see the PR
comment for how each gate was invoked. The test binaries were built from the PR's commit with the builders' rustc recipe.
