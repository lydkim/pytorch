"""ramjet test: run the tests under the paths you name on the GPUs you name, or list
them. The work is calls to a Modal Function, which Modal spreads over as many
containers as it gives you, each taking one call at a time. The first calls each name
as many test files as a container has cores: the container collects their tests and
leaves out those whose tci constraints (torch/testing/_internal/tci) its machine
doesn't meet. The rest are cut into pieces that should each take a container about
PIECE_SECONDS, by CI's time for each test and with as many test processes at once
as run_test.py runs there. Each piece is one more call, which a container runs
through run_test.py: the containers that collected take the first pieces."""

import contextlib
import gzip
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import modal

from tools.ramjet import progress, sync, times, worker
from tools.ramjet.common import count, repo_root
from tools.ramjet.lineage import Build
from tools.ramjet.logger import bold, CONTINUED, dim, log, magenta, red, RING
from tools.ramjet.run import Resources
from tools.testing.discover_tests import TESTS


# A container's resources unless --cpu and --memory say otherwise: CORES_PER_GPU per
# GPU, or as many without one, with MIB_PER_CORE of memory each
CORES_PER_GPU = 4
MIB_PER_CORE = 4096
# How long a container waits for a call before it stops. When listing, the least
# Modal allows. When running, long enough for the containers that collected to get
# the pieces, which are cut once CI has given the tests' times
SCALEDOWN_SECONDS = 2
HANDOVER_SECONDS = 15
PIECE_SECONDS = 150
QUEUE_ITEMS = 5000  # what a Modal Queue holds at most: all there is, in one call
# A container's seconds per second of CI, and its seconds to start a test file: fitted
# on L4 containers running test/dynamo
SLOWDOWN = 2.28
START_SECONDS = 5.4
# CI's mean for a test, for the tests of a file it has timed none of
GUESS_SECONDS = 0.3
# A test's outcomes, as torchci's reports name them, with ramjet's own flaky: those
# that need no look, and those that fail a run, as does a test that did not run. The
# summary counts each, in this order
FINE = ("passed", "flaky", "xfailed", "skipped")
FAILED = ("failed", "error", "crashed", "timed_out", "xpassed")

Tests = list[tuple[str, float]]  # node ids, each with the seconds it should take


@dataclass
class Piece:
    """What one container runs: tests of some files, each file started once."""

    seconds: float = 0.0  # what it should take the container
    tests: dict[str, list[str]] = field(default_factory=dict)  # node ids, by their file
    # The seconds run_test.py is told each file takes: past those it splits a file
    # at, it runs the file as several processes
    times: dict[str, float] = field(default_factory=dict)
    # The test process that runs each test, by node id
    where: dict[str, worker.Process] = field(default_factory=dict)


@dataclass
class Found:
    """What the containers made of the files asked for."""

    # The tests asked for that the containers can run, by file
    kept: dict[str, list[str]] = field(default_factory=dict)
    # Each file's tests marked @serialTest, kept or not
    serial: dict[str, set[str]] = field(default_factory=dict)
    alone: set[str] = field(default_factory=set)  # the files run_test.py runs alone
    # The containers, counted by what they read of themselves
    machines: Counter[str] = field(default_factory=Counter)
    failed: int = 0  # files that couldn't be collected


@dataclass
class Run:
    """Consecutive tests of a file, as run_test.py runs them in a piece."""

    name: str
    ids: list[str]
    beside: list[float]  # the seconds of each process it runs beside others
    alone: float  # the seconds of those it runs with nothing beside them

    @property
    def seconds(self) -> float:
        """What it takes a container that runs nothing else."""
        return max(self.beside, default=0.0) + self.alone


def names(paths: list[str]) -> dict[str, list[str]]:
    """run_test.py's name for each test file the paths cover, with what the paths
    ask of it: the file, for all its tests, or node ids in it. A path is a test
    file, a directory of them or a node id, from the directory you are in."""
    test_dir = repo_root() / "test"
    asked: dict[str, list[str]] = {}
    for path in paths:
        file, _, node = path.partition("::")
        target = Path(file).resolve()
        found: list[str] = []
        if target.is_dir() and not node and target.is_relative_to(test_dir):
            under = "" if target == test_dir else f"{target.relative_to(test_dir)}/"
            # TESTS also names C++ binaries and tests that have no file of their own
            found = [name for name in TESTS if name.startswith(under)]
            found = [name for name in found if (test_dir / f"{name}.py").is_file()]
        elif target.is_file() and target.is_relative_to(test_dir):
            name = target.relative_to(test_dir).with_suffix("").as_posix()
            found = [name] if name in TESTS else []
        if not found:
            log.error("%s is not a test file or directory that run_test.py runs", path)
            log.error("with pytest instead: python -m tools.ramjet -- pytest %s", path)
            sys.exit(1)
        for name in found:
            file_id = f"test/{name}.py"
            asked.setdefault(name, []).append(f"{file_id}::{node}" if node else file_id)
    return asked


@contextlib.contextmanager
def fleet(
    image: Build, commit: str, resources: Resources, ramjet_dir: Path, idle: int
) -> Iterator[tuple[modal.Cls, str]]:
    """For the block, the class whose containers boot from the build image and take
    your changes since its commit, and the address of its app on Modal's dashboard.
    A container takes one call at a time and stops once it has had none for idle
    seconds."""
    changes = sync.diff(image.commit, commit)
    changed = [path for path, status in changes.items() if status != "D"]
    deleted = [path for path, status in changes.items() if status == "D"]
    with (
        tempfile.NamedTemporaryFile(dir=ramjet_dir, suffix=".tar") as tar,
        tempfile.NamedTemporaryFile("w", dir=ramjet_dir) as gone,
    ):
        if changed:  # given no paths, git archives the whole tree
            sync.archive(commit, changed, Path(tar.name))
        gone.write("\0".join(deleted))
        gone.flush()
        files = (
            modal.Image.from_id(image.image_id)
            # The layer Modal's builder adds: on the build image alone, which is a
            # sandbox's snapshot, a Function's container can't start
            .run_commands("true")
            # Mounted as each container starts, so runs share the image
            .add_local_file(tar.name, worker.CHANGES)
            .add_local_file(gone.name, worker.DELETED)
        )
        app = modal.App("ramjet-test")
        # A piece runs for longer than Modal lets a call by default
        limits = {"scaledown_window": idle, "timeout": worker.CALL_SECONDS}
        workers = app.cls(image=files, **limits, **resources._asdict())(worker.Worker)
        stopped: KeyboardInterrupt | None = None
        with app.run():  # its containers stop when ramjet does, Ctrl-C included
            try:
                yield workers, app.get_dashboard_url()
            except KeyboardInterrupt as stop:
                stopped = stop
                raise
        if stopped:  # Modal ends its app on Ctrl-C, and then swallows it
            raise stopped


def collect(
    workers: modal.Cls,
    resources: Resources,
    names: list[str],
    pytest_args: list[str],
    meter: progress.Meter,
) -> Iterator[worker.Collected | Exception]:
    """Each test file's tests, in the order of names and as each call comes back. A
    call that Modal lost comes back as its exception, once for each of its files.
    Meanwhile, show how far the files are."""
    method = workers().collect
    # Collecting a file is one process, so a call names one file per core
    cores = resources.cpu
    calls = [names[i : i + cores] for i in range(0, len(names), cores)]
    back: list[worker.Collected | Exception] = []

    def ask() -> None:
        meter.told(method.get_current_stats().num_total_runners, "collecting")

    with progress.shown(progress.Collecting(meter, len(names), cores, back), ask):
        kwargs = {"pytest_args": pytest_args}
        results = method.map(calls, kwargs=kwargs, return_exceptions=True)
        for call, result in zip(calls, results, strict=True):
            files = [result] * len(call) if isinstance(result, Exception) else result
            back.extend(files)
            yield from files


def timed(kept: dict[str, list[str]], machine: dict[str, Any]) -> dict[str, Tests]:
    """Each test with the seconds it should take in a container: SLOWDOWN times CI's
    time for it on machines with the same GPUs, else on any machine, else the median
    of its file's tests."""
    files = {f"test/{name}.py": len(ids) for name, ids in kept.items()}
    device, gpus = machine["device"], machine["gpus"]
    ci = times.fetch(files, device, gpus)
    same = guessed = 0
    estimates = {}
    for name, ids in kept.items():
        seconds = []  # CI's for each test, or None for one it has never run
        for nodeid in ids:
            here, anywhere = ci.get(nodeid, (None, None))
            seconds.append(anywhere if here is None else here)
            same += here is not None
        known = [s for s in seconds if s is not None]
        typical = statistics.median(known) if known else GUESS_SECONDS
        guessed += len(ids) - len(known)
        scaled = [SLOWDOWN * (typical if s is None else s) for s in seconds]
        estimates[name] = list(zip(ids, scaled))
    others = sum(files.values()) - same - guessed
    on = f"{gpus} x {device}" if gpus else "no GPU"
    sources = count(same, "test"), on, others, guessed
    log.debug("CI's times: %s with %s, %d on other machines, %d guessed", *sources)
    if guessed:
        log.info("guessed the time of %s that CI has none for", count(guessed, "test"))
    return estimates


def lane(nodeid: str, processes: int) -> int:
    """Which of the processes that run_test.py runs a file as gets a test, from 0:
    test/pytest_shard_custom.py goes by the hash of the node id."""
    digest = hashlib.sha256(nodeid.encode()).digest()
    return int.from_bytes(digest, "little") % processes


def paced(name: str, tests: Tests, serial: Collection[str], processes: int) -> Run:
    """Tests of a file that run_test.py starts as so many processes beside others:
    none, for a file it runs alone. The tests marked @serialTest run after those,
    alone: serial names all of the file's, and each process that the hash gives
    one of them is started once more for its own."""
    beside = [START_SECONDS] * processes
    again = {lane(nodeid, processes) for nodeid in serial} if processes else {0}
    alone = START_SECONDS * len(again)
    for nodeid, seconds in tests:
        if processes and nodeid not in serial:
            beside[lane(nodeid, processes)] += seconds
        else:
            alone += seconds
    return Run(name, [nodeid for nodeid, _ in tests], beside, alone)


def placed(run: Run, serial: Collection[str]) -> dict[str, worker.Process]:
    """The test process that runs each test of a run, as paced has them: one of
    those beside others or, for a test marked @serialTest, the same one started
    again. A file that run_test.py runs alone is one process."""
    width = len(run.beside)
    if not width:
        return dict.fromkeys(run.ids, (run.name, 0, False))
    return {i: (run.name, lane(i, width), i in serial) for i in run.ids}


def cut(name: str, tests: Tests, serial: Collection[str], width: int) -> Iterator[Run]:
    """A file as runs of consecutive tests. A run ends where one more test would
    keep a container that runs it as width processes past PIECE_SECONDS, and then
    gets the fewest processes that keep it within them."""
    fresh = paced(name, [], serial, width)
    chunks: list[Tests] = [[]]
    loads = [*fresh.beside, fresh.alone]
    for nodeid, seconds in tests:
        at = lane(nodeid, width) if width and nodeid not in serial else -1
        loads[at] += seconds
        if chunks[-1] and max(loads[:-1], default=0.0) + loads[-1] > PIECE_SECONDS:
            chunks.append([])
            loads = [*fresh.beside, fresh.alone]
            loads[at] += seconds
        chunks[-1].append((nodeid, seconds))
    for chunk in chunks:
        ways = [paced(name, chunk, serial, n) for n in range(min(width, 1), width + 1)]
        # None, for a test that is longer than a piece: it gets one process
        yield next((run for run in ways if run.seconds <= PIECE_SECONDS), ways[0])


def length(runs: list[Run], procs: int) -> float:
    """What runs of different files take a container. run_test.py goes by the files'
    names: it starts the processes that run beside others in that order, procs at
    once, and runs the rest one after another."""
    ends = [0.0] * procs
    for run in sorted(runs, key=lambda run: run.name):
        for seconds in run.beside:
            ends[ends.index(min(ends))] += seconds
    return max(ends) + sum(run.alone for run in runs)


def shard(
    timed: dict[str, Tests],
    alone: Collection[str],
    serial: dict[str, set[str]],
    procs: int,
    split: float,
) -> list[Piece]:
    """Pieces that should each take a container about PIECE_SECONDS, the longest
    first. The container runs procs test processes at once, so a file that takes
    longer is cut into runs of consecutive tests that fill that many, and shorter
    files share a piece: one run of a file per piece, so that a piece starts each
    of its files once. The files in alone run as one process, with nothing beside
    it. run_test.py is told that a file takes split seconds for each process but
    the last, which makes it run the file as that many."""
    runs: list[Run] = []
    for name, tests in timed.items():
        width = 0 if name in alone else procs
        runs += cut(name, tests, serial.get(name, ()), width)
    pieces: list[list[Run]] = []
    for run in sorted(runs, key=lambda run: -(sum(run.beside) + run.alone)):
        fits = (
            piece
            for piece in pieces
            if all(other.name != run.name for other in piece)
            and length([*piece, run], procs) <= PIECE_SECONDS
        )
        piece = next(fits, None)  # the first with room, of the most work so far
        if piece is None:
            piece = []
            pieces.append(piece)
        piece.append(run)
    planned = []
    for piece in pieces:
        ids = {run.name: run.ids for run in piece}
        told = {run.name: split * (max(len(run.beside), 1) - 0.5) for run in piece}
        where: dict[str, worker.Process] = {}
        for run in piece:
            where.update(placed(run, serial.get(run.name, ())))
        planned.append(Piece(length(piece, procs), ids, told, where))
    return sorted(planned, key=lambda piece: -piece.seconds)


def gather(
    workers: modal.Cls,
    resources: Resources,
    names: dict[str, list[str]],
    pytest_args: list[str],
    meter: progress.Meter,
) -> Found:
    """What these containers make of the files asked for: the tests of each that
    they can run, and how run_test.py runs them there."""
    on = resources.gpu or "a CPU container"
    log.info("collecting %s on %s", count(len(names), "file"), on)
    results = collect(workers, resources, list(names), pytest_args, meter)
    found = Found()
    left_out: Counter[str] = Counter()  # tests, by the tci constraint that fails
    # strict, so that collect runs to its end, where it stops showing its progress
    for (name, asked), result in zip(names.items(), results, strict=True):
        if isinstance(result, Exception):  # Modal lost the call
            error = repr(result)
        else:
            took = result["seconds"], result["container"]
            log.debug("test/%s.py took %.1f s on container %s", name, *took)
            error = result["error"].rstrip()
            found.machines[json.dumps(result["machine"])] += 1
            found.serial[name] = set(result["serial"])
            if result["alone"]:
                found.alone.add(name)
            # A node id also asks for the tests under it and for its parametrizations
            under = tuple(f"{a}{mark}" for a in asked for mark in ("::", "["))
            # A test that can run here has None; one left out, its constraint
            verdicts = {**dict.fromkeys(result["ids"]), **result["left_out"]}
            for nodeid, unmet in verdicts.items():
                if nodeid not in asked and not nodeid.startswith(under):
                    continue
                if unmet:
                    log.debug("left out %s: %s", nodeid, unmet)
                    left_out[unmet] += 1
                else:
                    found.kept.setdefault(name, []).append(nodeid)
        if error:
            log.error("collecting test/%s.py failed:\n%s", name, error)
            found.failed += 1
    files = count(len(names) - found.failed, "file")
    tests = count(sum(map(len, found.kept.values())), "test")
    log.info("found %s in %s", tests, files)
    if left_out:
        tests = count(left_out.total(), "test")
        log.info("left out %s that %s can't run:", tests, on)
        for unmet, n in left_out.most_common():
            log.info("  %d with %s", n, unmet)
    return found


def plan(found: Found) -> list[Piece]:
    """The pieces to run the tests in, sized for the machine most containers are."""
    if len(found.machines) > 1:  # Modal may serve an H100 with an H200
        log.warning("containers differ: %s", ", ".join(found.machines))
    machine = json.loads(found.machines.most_common(1)[0][0])
    alone = [name for name in found.kept if name in found.alone]
    facts = machine["procs"], len(alone)
    log.debug("test processes at once: %d, and %d of the files alone", *facts)
    estimates = timed(found.kept, machine)
    pieces = shard(estimates, alone, found.serial, machine["procs"], machine["split"])
    for number, piece in enumerate(pieces, 1):
        tests = count(sum(map(len, piece.tests.values())), "test")
        sizes = tests, count(len(piece.tests), "file")
        log.debug("piece %d, about %.0f s: %s of %s", number, piece.seconds, *sizes)
    minutes = math.ceil(sum(piece.seconds for piece in pieces) / 60)
    sizes = count(len(pieces), "piece"), count(minutes, "container-minute")
    log.info("%s: %s, the longest about %.0f s", *sizes, pieces[0].seconds)
    return pieces


def run(
    workers: modal.Cls, pytest_args: list[str], state: progress.Running
) -> Iterator[worker.Ran | Exception]:
    """What the container of each of the state's pieces returns, as pieces finish. A
    call that Modal lost comes back as its exception. Meanwhile, show how far the
    pieces are."""
    method = workers().run
    # Where each container says how far its piece's processes are, every few seconds
    with modal.Queue.ephemeral() as said:

        def ask() -> None:
            state.meter.told(method.get_current_stats().num_total_runners, "running")
            for word in said.get_many(QUEUE_ITEMS, block=False):
                state.report(*word)

        with progress.shown(state, ask):
            queue = said.object_id
            pieces = enumerate(state.pieces, 1)
            calls = [(n, p.tests, p.times, p.where, queue) for n, p in pieces]
            options = {"kwargs": {"args": pytest_args}, "return_exceptions": True}
            for result in method.starmap(calls, order_outputs=False, **options):
                state.back.append(result)
                yield result


def outcomes(reports: dict[str, str], sent: list[str]) -> dict[str, tuple[str, str]]:
    """Each test sent to a piece, with its outcome and what its last failing run
    said, from torchci's reports: the outcome of its last run, flaky if that one
    passed after one that failed, and not_run without any."""
    keys = {worker.identity(nodeid): nodeid for nodeid in sent}
    runs: dict[str, list[tuple[int, int, str, str]]] = {}
    for report in reports.values():
        for line in report.split("\n"):  # splitlines also breaks inside strings
            try:
                run = json.loads(line)
            except ValueError:  # empty, or torn as its process died
                continue
            key = run.get("file"), run.get("suite"), run.get("case_name")
            if run["type"] == "run" and key in keys:
                when = run["started_at"], run["rerun_number"]
                attempt = *when, run["outcome"], run["outcome_summary"]
                runs.setdefault(keys[key], []).append(attempt)
    verdicts = {}
    for nodeid in sent:
        attempts = sorted(runs.get(nodeid, []))
        failures = [said for *_, outcome, said in attempts if outcome in FAILED]
        outcome = attempts[-1][2] if attempts else "not_run"
        if outcome == "passed" and failures:
            outcome = "flaky"
        said = failures[-1] if failures else attempts[-1][3] if attempts else ""
        verdicts[nodeid] = outcome, said
    return verdicts


def beyond(used: dict[int, float], asked: float, unit: str, option: str) -> None:
    """Warn of the pieces that used more than a container asks for, which Modal
    bills, and gives only while the machine has it to spare."""
    over = {number: most for number, most in used.items() if most > asked}
    if over:
        pieces, worst = count(len(over), "piece"), max(over, key=over.__getitem__)
        log.warning("%s used more than a container's %g %s", pieces, asked, unit)
        most = worst, over[worst], unit.partition(" ")[0], option
        log.warning("piece %d the most, %.1f %s: see %s", *most, extra=CONTINUED)


def execute(
    workers: modal.Cls,
    pieces: list[Piece],
    pytest_args: list[str],
    directory: Path,
    meter: progress.Meter,
    resources: Resources,
    address: str,
) -> int:
    """Run the pieces, logging each failure as its piece finishes, with every piece's
    output and reports kept in the directory. Print each test that failed or didn't
    run, a JSON object per line, and return how many. The summary ends the run: what
    it cost, how many tests it ran and for how long, how many ended in each state,
    the directory, and the address of the run's app on Modal. resources are what a
    container asks for. Ctrl-C ends the run with the summary of the pieces handled
    until then."""
    on = meter.gpu or "a CPU container"
    cost = meter.rate * Decimal(sum(piece.seconds for piece in pieces)) / 3600
    log.info("$%.2f estimate running %s on %s", cost, count(len(pieces), "piece"), on)
    (directory / "reports").mkdir(parents=True)
    state = progress.Running(meter, pieces)
    containers = set()
    cores: dict[int, float] = {}  # what each piece used, by its number
    memory: dict[int, float] = {}  # in GiB
    verdicts: dict[str, tuple[str, str]] = {}
    stopped: KeyboardInterrupt | None = None
    try:
        for ran in run(workers, pytest_args, state):
            if isinstance(ran, Exception):  # which doesn't say whose piece it was
                log.error("Modal lost a piece: %r", ran)
                continue
            number, code = ran["piece"], ran["code"]
            ended = number, ran["seconds"], ran["container"], code
            log.debug("piece %d took %.0f s on container %s: exit code %s", *ended)
            cores[number], memory[number] = ran["cores"], ran["memory"] / 1024
            used = number, ran["cores"], ran["memory"], ran["gpu_memory"]
            log.debug("piece %d used %.1f cores, %d MiB, and %d MiB of a GPU", *used)
            containers.add(ran["container"])
            output = directory / f"piece-{number}.log"
            output.write_bytes(gzip.decompress(ran["output"]))
            for name, report in ran["reports"].items():
                (directory / "reports" / name).write_text(report)
            sent = [i for ids in pieces[number - 1].tests.values() for i in ids]
            found = outcomes(ran["reports"], sent)
            for nodeid, (outcome, said) in found.items():
                if outcome in FAILED:
                    # The run's first failure rings the bell: you may be elsewhere
                    ring = None if state.failed else RING
                    log.error("failed: %s", nodeid, extra=ring)
                    log.error("  %s", said.partition("\n")[0])
                    state.failed += 1
                elif outcome == "flaky":
                    log.warning("flaky: %s", nodeid)
            if missing := sum(outcome == "not_run" for outcome, _ in found.values()):
                gave_up = f"stopped after {worker.GIVE_UP_SECONDS // 60} min"
                how = gave_up if code is None else f"exited {code}"
                lost = number, how, count(missing, "test"), output
                log.error("piece %d %s: %s didn't run, see %s", *lost)
                state.failed += missing
            verdicts.update(found)  # last: a piece counts once all of this is done
    except KeyboardInterrupt as stop:  # the summary of the pieces that count follows
        stopped = stop
    failing = 0
    for nodeid in (i for piece in pieces for ids in piece.tests.values() for i in ids):
        if stopped and nodeid not in verdicts:  # its piece was still to come
            continue
        outcome, _ = verdicts.setdefault(nodeid, ("not_run", ""))
        if outcome in FAILED or outcome == "not_run":
            # Flushed: the summary is the last a run writes, wherever stdout goes
            print(json.dumps({"nodeid": nodeid, "outcome": outcome}), flush=True)
            failing += 1
    if stopped:
        total = sum(len(ids) for piece in pieces for ids in piece.tests.values())
        left = f"{total - len(verdicts):,}", count(total, "test")
        log.warning("stopped by Ctrl-C: %s of %s did not run", *left)
    beyond(cores, resources.cpu, "cores", "--cpu")
    beyond(memory, resources.memory / 1024, "GiB of memory", "--memory")
    now, used = time.monotonic(), containers | set(state.started)
    log.debug("on %s: %s", meter.named(len(used)), meter.where())
    tests, took = count(len(verdicts), "test"), progress.span(now - meter.began)
    words = magenta(meter.cost(now)), dim("for"), bold(tests), dim("in"), took
    ring = None if stopped else RING  # the run's end rings the bell as well
    log.info("%s%s", " ".join(words), dim(":") if verdicts else "", extra=ring)
    tally = Counter(outcome for outcome, _ in verdicts.values())
    counts = {o: f"{n:,} {o.replace('_', ' ')}" for o, n in tally.items()}
    # Under it the states that need no look, then in red those that fail the run
    fine = [counts[o] for o in FINE if o in counts]
    bad = [red(counts[o].upper()) for o in (*FAILED, "not_run") if o in counts]
    for states in filter(None, (fine, bad)):
        log.info("%s", dim(" • ").join(states), extra=CONTINUED)
    for name, where in (("Logs", directory), ("Modal", address)):
        log.info("%s", dim(str(where)), extra={**CONTINUED, "margin": name})
    if stopped:
        raise stopped
    return failing


def run_tests(
    image: Build,
    commit: str,
    resources: Resources,
    names: dict[str, list[str]],
    pytest_args: list[str],
    ramjet_dir: Path,
    listing: bool,
    rate: Decimal,
) -> int:
    """Run the tests asked for that these containers can run, or print each with its
    piece, one JSON object per line. Return the exit code: 1 if a test failed or
    didn't run, or if a file's tests couldn't be collected. rate is what a container
    bills an hour."""
    idle = SCALEDOWN_SECONDS if listing else HANDOVER_SECONDS
    with fleet(image, commit, resources, ramjet_dir, idle) as (workers, address):
        meter = progress.Meter(resources.gpu, rate)
        found = gather(workers, resources, names, pytest_args, meter)
        failed = found.failed
        if not found.kept:
            return 1 if failed else 0
        pieces = plan(found)
        if listing:
            piece_of: dict[str, int] = {}
            for number, piece in enumerate(pieces, 1):
                for ids in piece.tests.values():
                    piece_of.update(dict.fromkeys(ids, number))
            for ids in found.kept.values():
                for nodeid in ids:
                    print(json.dumps({"nodeid": nodeid, "piece": piece_of[nodeid]}))
        else:
            directory = ramjet_dir / time.strftime("test-%Y%m%d-%H%M%S")
            work = (workers, pieces, pytest_args, directory)
            failed += execute(*work, meter, resources, address)
    return 1 if failed else 0
