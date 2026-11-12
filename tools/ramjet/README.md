# ramjet

ramjet builds your worktree on [Modal](https://modal.com) and runs a command, such as pytest, on a runner with the GPUs you name, so you can test on GPUs your machine does not have:

```
python -m tools.ramjet --gpu H100 -- pytest test/test_nn.py -k linear
```

The runner, the sandbox your command runs in, takes `modal shell`'s options. `--gpu` takes GPUs as Modal names them, in any case, with `:N` for N of them: `L4`, `A100-80GB`, `H100`, `B200:8`; without it the runner has no GPU. `--cpu` sets its physical cores, 1 to 64 in powers of two, and `--memory` its memory in MiB, 8192 to 344064; by default it gets 8 cores per GPU, or 8 without one, and 8192 MiB per core. Modal has no CPU types, so the CPU model and its instruction set can't be chosen. The builder is always the same: 64 cores and no GPU. `--help` lists every choice. `--build-cuda` picks the CUDA version to build with: any that CI builds an image for and that has an official arch list, 12.6, 13.2 and 13.4 today. The default is the version of PyTorch's default wheel. `--python-version` picks the Python version, the same in the builder and the runner, which boots from the builder's image: any PyTorch ships wheels for, 3.11 to 3.15 with the free-threaded `t` builds. The default is the one in CI's image, 3.11 today. `--log-level` sets the least severe messages ramjet shows: `debug`, `info` (the default), `warn` or `error`. Everything after `--` is the command, run as given in the directory you run ramjet from: `pytest test/test_nn.py -k linear`, `nvidia-smi`, or `bash -c '...'` for pipes and `&&`.

## Running tests

`test` runs the tests under the paths you name on as many containers as Modal gives you, each with the GPUs you name:

```
python -m tools.ramjet --gpu L4 test test/dynamo
```

ramjet collects the tests and cuts them into pieces of about 150 seconds, as `test --list` shows them (below), and each piece runs in a container of its own: the containers that collected take the first pieces, and Modal starts more for the rest. The container runs it through `run_test.py`, as CI does: as many test processes at once as CI runs on such a machine, a failed test again in the same process and then in new ones, and without the tests on CI's slow and disabled lists. Arguments after `--` go to every test file as well as to the collection; `-m` doesn't work there, since `run_test.py` sets its own. Meanwhile ramjet shows how far the pieces are ([Progress](#progress)). As a piece finishes, each of its tests that failed is logged with the first line of its error; a test that failed and then passed when run again is flaky, which is a warning. A container gives up on a piece after 30 minutes, and the tests it didn't get to count as not run. Once the pieces are back, ramjet warns if any kept more cores busy or held more memory than a container asks for, with the piece that used the most: Modal bills what is used, and gives more than was asked for only while the machine has it to spare. `--log-level debug` shows what each piece used, its GPU's memory included, how many containers the run used, and where its time went: without a container, collecting and running. stdout gets a line of JSON for each test that failed or didn't run, such as `{"nodeid": "test/dynamo/test_misc.py::MiscTests::test_cast", "outcome": "failed"}`, and the exit code is 1 if there is one, or if a file couldn't be collected.

The run ends with its summary:

```
ramjet 21:15:02  info $0.36 for 5,439 tests in 3 min:
                      4,602 passed • 2 flaky • 38 xfailed • 778 skipped
                      3 FAILED • 1 ERROR • 1 CRASHED • 1 TIMED OUT • 1 XPASSED • 12 NOT RUN
                 Logs /tmp/ramjet-<user>/pytorch-77265b67/test-20261111-211249
                Modal https://modal.com/id/<app id>
```

Its rows are what the run's containers billed, how many tests it ran and how long it took; how many tests ended in each state that needs no look; in red, how many ended in each state that fails the run; the directory that holds each piece's output and its tests' reports (`torch/testing/_internal/torchci/README.md`); and the run's app on Modal's dashboard, which has its containers and their logs. A state that no test ended in is left out, and so is a row without states. Ctrl-C stops the run, logs the summary of the pieces that are back, and exits 130. To work on a failure, run it alone on the warm runner with `--`.

## Listing tests

`test --list` prints the tests under the paths you name that the GPU you name can run, each with the piece it would run in, one line of JSON each, without running them:

```
python -m tools.ramjet --gpu L4 test --list test/dynamo -- -k guard
```

The test files are collected in parallel: Modal spreads them over as many containers as it gives you, and a container collects as many at once as it has cores. Meanwhile ramjet shows how many files have come back, or how long Modal has given it no container ([Progress](#progress)): Modal can run out of a GPU for minutes. Each file is launched the way `run_test.py` launches it, so the list is what would run from that file on that GPU. A test can say which machines it runs on with the decorators in `torch/testing/_internal/tci`, such as `@tci.only(tci.cuda)` or `@tci.needs(tci.gpus >= 2)`. The container leaves out the tests whose constraints it doesn't meet, and ramjet logs how many, by constraint; `--log-level debug` names each one. A path is a test file, a directory of them, or a node id such as `test/test_nn.py::TestNN`, among the test files `run_test.py` runs; for others, use `-- pytest`. Arguments after `--` go to pytest's collection. A container gets 4 cores and 16384 MiB per GPU, or as much without one, unless `--cpu` and `--memory` say otherwise, and bills as a Modal Function, at a third of a sandbox's rates for cores and memory. The tests are then cut into pieces that should each take a container about 150 seconds, with as many test processes at once as `run_test.py` runs there: three, or two on GPUs older than Ampere such as T4. A file that takes longer fills that many processes of a piece: `run_test.py` gives each test to one of them by the hash of its node id, as it does in CI, so ramjet cuts the file where the fullest of them is full. Shorter files share a piece, side by side. A file that CI runs with nothing beside it, such as `test_nn.py`, is one process on its own, and the tests marked `@serialTest` run after their file's others, alone. For a piece, ramjet writes the file CI's sharding reads (`.additional_ci_files/test-times.json`) in the container, with the time that makes `run_test.py` start each file as the processes planned. Each line of the list has its test's `piece`, and the log says how many pieces there are and how much container time they add up to. A piece is sized by CI's time for each of its tests over the last 7 days, on machines with the same GPUs as the container or, for a test those don't run, on any machine; a test CI has no time for counts as the median of its file, and the log says how many were guessed. The times come from CI's database through the ci-metrics skill's wrapper (`.agents/skills/ci-metrics`), which needs a `gh` login that HUD accepts. Without one ramjet warns and sizes the pieces as if all tests took the same time. If a file can't be collected, ramjet shows the end of its output and exits 1. `--log-level debug` shows how long each file took, which container collected it, how many tests CI timed on the same GPUs and what is in each piece.

## Progress

On a terminal, `test` keeps a block under its log and draws it anew in place. Its first row says how far the run is, such as `81% • 55,655/68,228 • 2 FAILED • 16/26 A10 • 2 min left • $1.90`: the share of the tests that have ended; how many have, of all there are; how many failed in the pieces that are back, once one has; how many containers the run has of how many it can use now; about how long is left; and what the containers have billed so far. While files are collected the row says so and counts them instead, with no share: `collecting 80/152 files • 1/18 L4 • 3 min left • $0.06`. A row that starts with a percent is of tests. The run can use a container for each piece that isn't back or, while it collects, for as many files as a container has cores. The time left is the work left, at the pace of the pieces that are back, shared by the containers there are now; none is shown once the pieces left are all well past their plan. In a terminal too narrow for the row, the share goes first, then the cost, the time left and the containers. While tests run, each test process that runs and still has tests left has a row as wide as the terminal, such as `A10 3  2:11 [ 398/1167] test_ops.py::TestCommonCUDA::test_errors_gradient_cuda`: the container it runs in, how long it has been on its file, how many of its tests have ended, and the test it runs, by its file, suite and case, cut at its end where the terminal is too narrow. The container is the runner as `--gpu` names it, or `CPU` without one, and the container's number in the order the containers came: rows with the same number share a container. Before its first test, the row has the file alone. The process longest on its file comes first. Up to 8 are listed, fewer in a short terminal, and the rest are counted. For this, each container says how far its test processes are, every two seconds, on a Modal Queue that lives as long as the run; if it can't, its piece runs all the same. While Modal has given no container, the row says for how long, and that nothing is billed: `no L4 yet • 6:52 • nothing billed`. After three minutes a row under it adds that Ctrl-C is free and, for a GPU PyTorch's CI tests on, the others it tests on. The window's title says how far the run is, and the terminal's bell rings at the first failure and at the end.

When stderr is not a terminal there is no block. The same facts are a line of the log when a container arrives, at each tenth of the files or of the tests, and after five minutes without a line; never two within 15 seconds.

## Setup

```
pip install modal
modal token new
```

## How it works

1. Your working tree, edits and untracked files included, becomes a commit. HEAD, the index and the stash are left alone.
2. If the build reads none of the files that changed since the last build, there is no build. Otherwise a 64-core builder boots from the last build image, fetches a git bundle of the new commits, rebuilds incrementally and saves itself as the next build image. The first build clones pytorch/pytorch into CI's CUDA image and takes 1-1.5 hours when the compile cache has none of its compiles. Builders share that cache: ccache looks up each compile in the build image's own cache, then in the `ramjet-ccache` Modal Volume, which every builder in your Modal workspace reads and adds to. Each build deletes the Volume's entries written more than 14 days ago.
3. A runner of the size you ask for boots from the build image, or the warm one from your last run is reused. It gets a tar of the files that changed and runs your command.

## Costs

The builder bills about $13 an hour, so a first build costs $13-20 and an incremental one much less. A runner bills its GPUs plus its cores and memory: 8 cores and 64 GiB per GPU unless `--cpu` and `--memory` say otherwise. After a run it stays warm for 15 minutes, or until idling has cost $2.50, whichever comes first (`IDLE_MAX_SECONDS` and `IDLE_BUDGET` in `run.py`). `test`'s containers bill at a Function's rates, about $1.12 an hour for an L4 with 4 cores and 16 GiB, while they collect or run; in a run, one that finds no piece to take stops after 15 seconds. Before it runs the pieces, `test` logs about what they will cost by CI's times; a test that fails is run up to 9 times, so a change that breaks many tests costs more. A full build adds about 1.2 GB to the compile cache's Volume; Modal bills Volume storage past 1 TiB, at $0.09/GiB a month.

## Files

ramjet keeps its state in `/tmp/ramjet-<user>/<worktree folder>-<hash of its path>`, or under `$TMPDIR` if you set it: the build images per build config (`lineage.json`), the last build's log (`build.log`), a warm runner per runner size, and a `test-<time>` directory for each run of `test`. Build images expire after 5 days; ramjet rebuilds images older than 4 days so a lineage outlives them. A machine that clears `/tmp` of files not written for some days would take the lineage with it, so each run refreshes the timestamps of the lineage files it uses.

## Limits

- Builds use the official arch list for their CUDA version. Before building, ramjet stops if that list, or your `TORCH_CUDA_ARCH_LIST`, can't run on `--gpu`: 12.6's stops at sm_90, so it can't run on Blackwell GPUs such as B200, B300 or RTX PRO 6000. `--build-cuda` and your `BUILD_*`, `USE_*`, `CMAKE_BUILD_TYPE`, `DEBUG`, `REL_WITH_DEB_INFO` and `TORCH_CUDA_ARCH_LIST` variables change the build; each combination gets its own lineage.
- CI's CUDA images come with one Python. ramjet installs another `--python-version` the way CI's Dockerfile installs Python (`install_python.sh` into a venv first on `PATH`), as its own lineage. CI runs CUDA tests only on 3.11 and 3.12, so other versions with CUDA are untested there.
- A lineage stays on the version of CI's image it started on, so pulling a `.ci/docker` change doesn't restart it. To move to the current version, delete `lineage.json`.
- The command gets no terminal and no input, so interactive programs such as a bare `bash` or `python` don't work.
- Your whole Modal workspace shares the compile cache: a build links whatever objects other builders put in it.
