"""The class ramjet test runs on Modal. Each container boots from the build image,
brings its /pytorch to your working tree once, and then serves calls. It is imported
in the containers, so it uses only the standard library and modal."""

import contextlib
import functools
import gzip
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import modal


ROOT = "/pytorch"
# Your changes since the build image's commit, where test.py mounts them
CHANGES = "/tmp/ramjet/changes.tar"
DELETED = "/tmp/ramjet/deleted"  # paths, NUL-separated
TAIL = 2000  # characters of a failed collect's output to return
# pytest's plugin that deselects the tests whose tci constraints a machine doesn't meet
TCI = "torch.testing._internal.tci.plugin"
# pytest's plugin that names the tests marked @serialTest
SERIAL = "tools.ramjet.serial"
# A container gives up on a file's tests, which it lists in seconds, after
# COLLECT_SECONDS, and on a piece, which is planned to take a few minutes, after
# GIVE_UP_SECONDS; Modal ends any call a little later, at CALL_SECONDS
COLLECT_SECONDS = 5 * 60
GIVE_UP_SECONDS = 30 * 60
CALL_SECONDS = GIVE_UP_SECONDS + 5 * 60
TELL_SECONDS = 2  # between two words of how far a piece is, for test.py's progress
# What run_test.py prints as it starts a test process, and as that one ends: the
# name of the process's test file and which of the file's processes it is, from 1
PROCESS = re.compile(rb"(Running|Finished) (\S+) (\d+)/\d+ \.\.\. \[")
# Where torchci's plugin names the test a process runs, for run_test.py: its
# report_inflight in pytest's cache, under conftest's STEPCURRENT_CACHE_DIR and the
# test file's name, which may have a directory
IN_FLIGHT = ".pytest_cache/v/cache/stepcurrent/**/report_inflight"
# CI's list of disabled tests, fetched as run_test.py fetches it in CI, to where
# --import-disabled-tests reads it
DISABLED = "from tools.stats import import_test_stats as stats; "
DISABLED += "stats.get_disabled_tests('test')"
# What run_test.py lets a machine do, as this prints it in /pytorch/test: how many
# test processes it runs at once, the seconds past which it runs a file as several
# of them, and the files it runs with nothing beside them
ALLOWS = "import json, run_test as rules; "
ALLOWS += "alone = [name for name in rules.TESTS if rules.must_serial(name)]; "
ALLOWS += "allows = {'procs': rules.NUM_PROCS, 'split': rules.THRESHOLD}; "
ALLOWS += "print(json.dumps({**allows, 'alone': alone}))"
TIMES = ".additional_ci_files/test-times.json"  # where run_test.py reads files' times
# What prints the MiB in use on each GPU, every so many milliseconds
GPU_MEMORY = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
GPU_MEMORY += " --loop-ms={}"


class Collected(TypedDict):
    ids: list[str]  # as pytest prints them: test/test_nn.py::TestNN::test_linear
    left_out: dict[str, str]  # the tests tci leaves out here: the constraint unmet
    serial: list[str]  # the file's tests marked @serialTest, whatever leaves them out
    error: str  # the end of the output if collecting failed, else empty
    seconds: float  # how long collecting took
    container: str  # Modal's id for the container that did it
    # The container as tci reads it: its GPUs' name, count... With what run_test.py
    # lets it do: procs, how many test processes at once, and split, the seconds
    # past which a file runs as several of them
    machine: dict[str, Any]
    alone: bool  # whether run_test.py runs the file with nothing beside it


class Ran(TypedDict):
    piece: int  # the number test.py gave it
    code: int | None  # run_test.py's exit code; None if GIVE_UP_SECONDS ended it
    reports: dict[str, str]  # torchci's reports by file name: a JSON line per test run
    output: bytes  # what run_test.py printed, gzipped
    seconds: float  # how long the piece took
    container: str  # Modal's id for the container that ran it
    cores: float  # how many its processes kept busy, on average
    memory: int  # the most MiB in use in the container while it ran
    gpu_memory: int  # the most MiB in use on a GPU


Test = tuple[str, str, str]  # as torchci identifies it: file, suite and case name
# A test process that run_test.py starts for a piece: its test file's name, which of
# the file's processes it is, from 0, and whether it is the one started after the
# others, for the tests marked @serialTest
Process = tuple[str, int, bool]
# How far a piece is: how many tests of each of its processes have ended, and the
# processes that run, each with the test it runs or ran last, if it has had one
Said = tuple[dict[Process, int], dict[Process, Test | None]]


def identity(nodeid: str) -> Test:
    """How torchci identifies the test with a node id: by its file, innermost class
    and name."""
    path, bracket, parameters = nodeid.partition("[")  # which may hold a :: too
    file, *classes, case = path.split("::")
    return file, classes[-1] if classes else "", f"{case}{bracket}{parameters}"


@dataclass
class Watch:
    """How far a piece is, read from what run_test.py and its test processes write
    as they run."""

    reports: Path  # torchci's, for the piece: each gets a line as a test's run ends
    output: Path  # what run_test.py prints: see PROCESS
    where: dict[Test, Process]  # the process that runs each of the piece's tests
    read: dict[Path, int] = field(default_factory=dict)  # bytes of each, so far
    ended: set[Test] = field(default_factory=set)
    # How many times each of a file's processes has started, and those that run
    starts: Counter[tuple[str, int]] = field(default_factory=Counter)
    running: dict[Process, Test | None] = field(default_factory=dict)

    def fresh(self, path: Path) -> list[bytes]:
        """The lines a file has gained since the last look."""
        with open(path, "rb") as file:
            file.seek(self.read.get(path, 0))
            *lines, rest = file.read().split(b"\n")  # the last: being written
            self.read[path] = file.tell() - len(rest)
        return lines

    def look(self) -> Said:
        """What the piece's test processes have done, and do now: see Said."""
        for path in self.reports.glob("*.jsonl"):
            for line in self.fresh(path):
                with contextlib.suppress(ValueError, KeyError):  # torn, or no run
                    run = json.loads(line)
                    if run["type"] == "run":
                        self.ended.add((run["file"], run["suite"], run["case_name"]))
        for line in self.fresh(self.output):
            if printed := PROCESS.match(line):
                which = printed[2].decode(errors="replace"), int(printed[3]) - 1
                # run_test.py starts a process once for its tests, and once more
                # for those of them marked @serialTest
                if printed[1] == b"Running":
                    self.running[(*which, self.starts[which] > 0)] = None
                    self.starts[which] += 1
                else:
                    self.running.pop((*which, self.starts[which] > 1), None)
        for path in Path(ROOT).glob(IN_FLIGHT):
            # Gone, being written, or of a test that is not this piece's
            with contextlib.suppress(OSError, ValueError, KeyError):
                flight = json.loads(path.read_text())
                if flight and flight["path"].startswith(f"{self.reports}/"):
                    named = flight["test"]
                    test = named["file"], named["suite"], named["case_name"]
                    if (process := self.where[test]) in self.running:
                        self.running[process] = test
        done = Counter(self.where[test] for test in self.ended & self.where.keys())
        return dict(done), dict(self.running)


def environ() -> dict[str, str]:
    """The environment of what a container runs in /pytorch."""
    # Modal's PYTHONPATH holds its own packages and the tools/ it mounts to import
    # this module, which would shadow /pytorch/tools; the image sets none
    return {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}


def facts(*python: str) -> dict[str, Any]:
    """What a Python command prints last in /pytorch/test, which is JSON."""
    output = {"capture_output": True, "text": True}
    command = [sys.executable, *python]
    done = subprocess.run(command, cwd=f"{ROOT}/test", env=environ(), **output)
    if done.returncode != 0:
        raise RuntimeError(f"can't read the container:\n{done.stderr[-TAIL:]}")
    return json.loads(done.stdout.splitlines()[-1])


def machine() -> dict[str, Any]:
    """This container, as tci reads it."""
    return facts("-m", "torch.testing._internal.tci.machine")


def allows() -> dict[str, Any]:
    """What run_test.py lets this container do: see ALLOWS."""
    return facts("-c", ALLOWS)


def in_use() -> int:
    """The MiB of memory in use in this container, each page once, as its cgroup
    counts them. Its processes' own counts add up to several times that: each
    names the pages it shares with others."""
    kib = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, rest = line.partition(":")
        kib[name] = int(rest.split()[0])
    return (kib["MemTotal"] - kib["MemAvailable"]) // 1024


@contextlib.contextmanager
def gauged() -> Iterator[list[int]]:
    """For the block, the MiB in use on each GPU are read every TELL_SECONDS: after
    it, the list holds them. None without nvidia-smi, as where there is no GPU."""
    read: list[int] = []
    every = GPU_MEMORY.format(int(1000 * TELL_SECONDS)).split()
    pipes = {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL, "text": True}
    try:
        gauge = subprocess.Popen(every, **pipes)
    except OSError:
        yield read
        return
    try:
        yield read
    finally:
        gauge.kill()
        said, _ = gauge.communicate()
        read += [int(mib) for mib in said.split() if mib.isdigit()]


class Worker:
    machine: Future[dict[str, Any]]  # being read since sync, as allows is
    allows: Future[dict[str, Any]]
    fetched = False  # CI's list of disabled tests, which a container fetches once

    @modal.enter()
    def sync(self) -> None:
        """Once per container, what run.sync_files does to a runner: rm for the files
        deleted, then a tar of those changed."""
        for path in filter(None, Path(DELETED).read_text().split("\0")):
            Path(ROOT, path).unlink(missing_ok=True)
        if Path(CHANGES).stat().st_size:  # empty when no file changed
            subprocess.run(["tar", "-xf", CHANGES], cwd=ROOT, check=True)
        # Read while the first files are collected: each takes as long as importing
        # torch
        pool = ThreadPoolExecutor(2)
        self.machine, self.allows = pool.submit(machine), pool.submit(allows)

    @modal.method()
    def collect(self, names: list[str], pytest_args: list[str]) -> list[Collected]:
        """collect_file's answer for each test file named. The files are collected at
        once: a call names one per core."""
        with ThreadPoolExecutor(len(names)) as pool:
            one = functools.partial(self.collect_file, pytest_args=pytest_args)
            return list(pool.map(one, names))

    def collect_file(self, name: str, pytest_args: list[str]) -> Collected:
        """The tests run_test.py would run here from one of its test files, without
        those that tci leaves out on this machine."""
        # Launched as run_test.py launches it, so the file's __main__ guard decides
        # here too: pytest alone collects files that guard never lets run
        launch = [sys.executable, "-bb", f"{name}.py", "--use-pytest", "--collect-only"]
        where = {"cwd": f"{ROOT}/test", "env": environ(), "timeout": COLLECT_SECONDS}
        output = {"capture_output": True, "text": True, "errors": "replace"}
        start, scratch = time.monotonic(), tempfile.NamedTemporaryFile
        with scratch() as deselected, scratch() as marked:
            plugins = ["-p", TCI, f"--tci-deselected={deselected.name}"]
            plugins += ["-p", SERIAL, f"--ramjet-serial={marked.name}"]
            # Last, so a -v among your arguments can't turn the ids into a tree
            command = [*launch, *plugins, *pytest_args, "--verbosity=-1"]
            try:
                done = subprocess.run(command, **where, **output)
            except subprocess.TimeoutExpired:
                slow = f"still collecting after {COLLECT_SECONDS} s"
                done = subprocess.CompletedProcess(command, 1, "", slow)
            left_out = deselected.read()  # nothing, if the file started no pytest
            serial = marked.read().decode()
        failed = done.returncode != 0
        prefix = f"test/{name}.py::"  # the other lines are the file's own output
        lines = [] if failed else done.stdout.splitlines()
        allows = self.allows.result()
        limits = {key: allows[key] for key in ("procs", "split")}
        return {
            "ids": [line for line in lines if line.startswith(prefix)],
            "left_out": json.loads(left_out) if left_out and not failed else {},
            "serial": [] if failed else list(filter(None, serial.split("\n"))),
            "error": (done.stderr + done.stdout)[-TAIL:] if failed else "",
            "seconds": time.monotonic() - start,
            "container": os.environ.get("MODAL_TASK_ID", ""),
            "machine": {**self.machine.result(), **limits},
            "alone": name in allows["alone"],
        }

    @modal.method()
    def run(
        self,
        piece: int,
        tests: dict[str, list[str]],
        times: dict[str, float],
        where: dict[str, Process],
        queue: str,
        args: list[str],
    ) -> Ran:
        """Run a piece: some tests, by node id, of some of run_test.py's test files.
        run_test.py runs them as CI does: a few processes at once, a failure again
        in a new process, and without the tests on CI's slow and disabled lists. It
        is told that the files take the times given, which is what makes it run a
        file as several processes: where names the one that runs each test. As the
        piece starts, and then every TELL_SECONDS in which that changed, say how far
        its processes are on the queue with that id, for test.py's progress."""
        python, env, start = sys.executable, environ(), time.monotonic()
        container = os.environ.get("MODAL_TASK_ID", "")
        told = modal.Queue.from_id(queue)
        last: Said = ({}, {})
        # Without waiting: the piece runs whether or not the queue takes it
        with contextlib.suppress(Exception):
            told.put((piece, container, *last), block=False)
        known = Path(ROOT, TIMES)
        known.parent.mkdir(exist_ok=True)
        known.write_text(json.dumps({"default": {"default": times}}))
        with tempfile.TemporaryDirectory() as scratch:
            ids, reports = Path(scratch, "ids"), Path(scratch, "reports")
            ids.write_text("\n".join(i for run in tests.values() for i in run))
            launch = [python, "test/run_test.py", "--keep-going", "--pipe-logs"]
            launch += [f"--save-torchci-reports={reports}", "-i", *tests, "--"]
            # What run_test.py hands each of its test files, which all read the ids
            launch += [f"--nodeid-file={ids}", "--import-slow-tests"]
            launch += ["--import-disabled-tests", *args]
            with open(Path(scratch, "output"), "w+b") as printed, gauged() as on_gpus:
                output = {"stdout": printed, "stderr": subprocess.STDOUT}
                if not self.fetched:
                    fetch = [python, "-c", DISABLED]
                    done = subprocess.run(fetch, cwd=ROOT, env=env, **output)
                    self.fetched = done.returncode == 0
                # A session of its own, so that giving up ends its test processes too
                session = {"start_new_session": True, **output}
                before, began = os.times(), time.monotonic()
                process = subprocess.Popen(launch, cwd=ROOT, env=env, **session)
                names = {identity(nodeid): which for nodeid, which in where.items()}
                watch, memory = Watch(reports, Path(printed.name), names), 0
                code: int | None = None
                while time.monotonic() - start < GIVE_UP_SECONDS:
                    try:
                        code = process.wait(timeout=TELL_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        memory = max(memory, in_use())
                        with contextlib.suppress(Exception):  # as above
                            if (seen := watch.look()) != last:
                                told.put((piece, container, *seen), block=False)
                                last = seen
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                after, took = os.times(), time.monotonic() - began
                printed.seek(0)
                said = gzip.compress(printed.read())
            busy = after.children_user + after.children_system
            busy -= before.children_user + before.children_system
            found = sorted(reports.glob("*.jsonl"))
            return {
                "piece": piece,
                "code": code,
                "reports": {path.name: path.read_text() for path in found},
                "output": said,
                "seconds": time.monotonic() - start,
                "container": container,
                "cores": busy / took,
                "memory": memory,
                "gpu_memory": max(on_gpus, default=0),
            }
