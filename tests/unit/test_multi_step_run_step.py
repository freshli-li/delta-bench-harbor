from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import ExecResult
from harbor.models.task.config import (
    TaskConfig,
    StepConfig,
    TaskOS,
    VerifierConfig,
    VerifierEnvironmentMode,
)
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.models.trial.result import ExceptionInfo, StepResult
from harbor.trial.errors import AgentTimeoutError
from harbor.trial.multi_step import MultiStepTrial


def _exception_info() -> ExceptionInfo:
    try:
        raise RuntimeError("prepare failed")
    except RuntimeError as exc:
        return ExceptionInfo.from_exception(exc)


def _trial_for_reset(
    *, mounted: bool = False, os: TaskOS = TaskOS.LINUX
) -> MultiStepTrial:
    trial = object.__new__(MultiStepTrial)
    trial.logger = MagicMock()
    trial.agent_env_paths = EnvironmentPaths.for_os(os)
    trial.agent_environment = MagicMock()
    trial.agent_environment.capabilities.mounted = mounted
    trial.agent_environment.os = os
    return trial


@pytest.mark.asyncio
async def test_fresh_step_carries_auto_memory_over_the_wipe() -> None:
    trial = _trial_for_reset()
    order: list[str] = []

    async def run(command: str, **kwargs: object) -> ExecResult:
        order.append("exec")
        return ExecResult(stdout="", stderr="", return_code=0)

    async def empty_dirs(*args: object, **kwargs: object) -> None:
        order.append("wipe")

    trial.agent_environment.exec = AsyncMock(side_effect=run)
    trial.agent_environment.empty_dirs = AsyncMock(side_effect=empty_dirs)

    await trial._reset_agent_logs_for_step(resume=False)

    assert order == ["exec", "wipe", "exec"]
    stash_cmd = trial.agent_environment.exec.await_args_list[0].args[0]
    restore_cmd = trial.agent_environment.exec.await_args_list[1].args[0]
    # The stash has to live outside the directory the wipe empties.
    assert "/logs/.harbor-memory-stash" in stash_cmd
    assert "/logs/.harbor-memory-stash" in restore_cmd
    trial.agent_environment.empty_dirs.assert_awaited_once_with(
        [EnvironmentPaths().agent_dir], chmod=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("mounted", "resume"), [(False, True), (True, False)])
async def test_agent_dir_untouched_when_resumed_or_mounted(
    mounted: bool, resume: bool
) -> None:
    trial = _trial_for_reset(mounted=mounted)
    trial.agent_environment.exec = AsyncMock()
    trial.agent_environment.empty_dirs = AsyncMock()

    await trial._reset_agent_logs_for_step(resume=resume)

    trial.agent_environment.exec.assert_not_awaited()
    trial.agent_environment.empty_dirs.assert_not_awaited()


@pytest.mark.asyncio
async def test_windows_fresh_step_empties_without_memory_carry_over() -> None:
    trial = _trial_for_reset(os=TaskOS.WINDOWS)
    trial.agent_environment.exec = AsyncMock()
    trial.agent_environment.empty_dirs = AsyncMock(
        return_value=ExecResult(stdout="", stderr="", return_code=0)
    )

    await trial._reset_agent_logs_for_step(resume=False)

    trial.agent_environment.exec.assert_not_awaited()
    trial.agent_environment.empty_dirs.assert_awaited_once_with(
        [EnvironmentPaths.for_windows().agent_dir], chmod=True
    )


@pytest.mark.asyncio
async def test_prepare_failure_archives_without_running_agent_or_collecting_artifacts() -> (
    None
):
    trial = object.__new__(MultiStepTrial)
    trial.logger = MagicMock()
    trial.config = SimpleNamespace(
        agent=SimpleNamespace(resume_trajectory=False, load_trajectory=None)
    )
    trial._create_step_dirs = MagicMock()

    async def fail_prepare(
        _step: StepConfig, step_result: StepResult, *, resume: bool = False
    ) -> None:
        step_result.exception_info = _exception_info()

    trial._prepare_step = AsyncMock(side_effect=fail_prepare)
    trial._run_step_agent = AsyncMock()
    trial._upload_agent_logs = AsyncMock()
    trial._archive_step_outputs = MagicMock()
    trial._collect_step_artifacts = AsyncMock()

    step = StepConfig(name="setup")
    step_result = StepResult(step_name=step.name)

    await trial._run_step(step, step_result, index=1, total=2)

    assert step_result.exception_info is not None
    trial._run_step_agent.assert_not_awaited()
    trial._upload_agent_logs.assert_not_awaited()
    trial._archive_step_outputs.assert_called_once_with(step, carry_agent_memory=True)
    trial._collect_step_artifacts.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_step_collects_artifacts_before_verifier() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.logger = MagicMock()
    events: list[str] = []
    stop_main_flags: list[bool] = []

    async def collect_step_artifacts(
        _step: StepConfig, *, stop_main_before_sidecars: bool
    ) -> Path:
        events.append("collect")
        stop_main_flags.append(stop_main_before_sidecars)
        return Path("/tmp/artifacts")

    async def run_step_verifier(*args, **kwargs) -> None:
        events.append("verify")

    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(disable=False),
        agent=SimpleNamespace(resume_trajectory=False, load_trajectory=None),
    )
    trial.task = SimpleNamespace(config=TaskConfig())
    trial._create_step_dirs = MagicMock()
    trial._prepare_step = AsyncMock()
    trial._run_step_agent = AsyncMock()
    trial._upload_agent_logs = AsyncMock()
    trial._collect_step_artifacts = AsyncMock(side_effect=collect_step_artifacts)
    trial._run_step_verifier = AsyncMock(side_effect=run_step_verifier)
    trial._archive_step_outputs = MagicMock()

    step = StepConfig(name="agent")
    step_result = StepResult(step_name=step.name)

    await trial._run_step(step, step_result, index=1, total=1)

    assert events == ["collect", "verify"]
    # Shared mode keeps the main service running for the verifier.
    assert stop_main_flags == [False]
    trial._run_step_verifier.assert_awaited_once_with(
        step,
        step_result,
        artifacts_dir=Path("/tmp/artifacts"),
        mode=VerifierEnvironmentMode.SHARED,
    )
    trial._archive_step_outputs.assert_called_once_with(
        step, preserve_agent=False, carry_agent_memory=False
    )


@pytest.mark.asyncio
async def test_run_step_stops_final_separate_step_before_verifier() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.logger = MagicMock()
    events: list[str] = []
    stop_main_flags: list[bool] = []

    async def collect_step_artifacts(
        _step: StepConfig, *, stop_main_before_sidecars: bool
    ) -> Path:
        events.append("collect")
        stop_main_flags.append(stop_main_before_sidecars)
        return Path("/tmp/artifacts")

    async def stop_agent_environment() -> None:
        events.append("stop")

    async def run_step_verifier(*args, **kwargs) -> None:
        events.append("verify")

    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(disable=False),
        agent=SimpleNamespace(resume_trajectory=False, load_trajectory=None),
    )
    trial.task = SimpleNamespace(config=TaskConfig())
    trial._create_step_dirs = MagicMock()
    trial._prepare_step = AsyncMock()
    trial._run_step_agent = AsyncMock()
    trial._upload_agent_logs = AsyncMock()
    trial._collect_step_artifacts = AsyncMock(side_effect=collect_step_artifacts)
    trial._stop_agent_environment = AsyncMock(side_effect=stop_agent_environment)
    trial._run_step_verifier = AsyncMock(side_effect=run_step_verifier)
    trial._archive_step_outputs = MagicMock()

    step = StepConfig(
        name="agent",
        verifier=VerifierConfig(environment_mode=VerifierEnvironmentMode.SEPARATE),
    )
    step_result = StepResult(step_name=step.name)

    await trial._run_step(step, step_result, index=2, total=2)

    assert events == ["collect", "stop", "verify"]
    # Final separate step: main may be stopped before sidecar collection.
    assert stop_main_flags == [True]
    trial._run_step_verifier.assert_awaited_once_with(
        step,
        step_result,
        artifacts_dir=Path("/tmp/artifacts"),
        mode=VerifierEnvironmentMode.SEPARATE,
    )
    trial._archive_step_outputs.assert_called_once_with(
        step, preserve_agent=False, carry_agent_memory=False
    )


@pytest.mark.asyncio
async def test_run_step_preserves_agent_dir_when_next_step_resumes() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.logger = MagicMock()
    first = StepConfig(name="first")
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(disable=True),
        agent=SimpleNamespace(resume_trajectory=True, load_trajectory=None),
    )
    trial.task = SimpleNamespace(config=TaskConfig())
    trial._create_step_dirs = MagicMock()
    trial._prepare_step = AsyncMock()
    trial._run_step_agent = AsyncMock()
    trial._upload_agent_logs = AsyncMock()
    trial._collect_step_artifacts = AsyncMock(return_value=Path("/tmp/artifacts"))
    trial._run_step_verifier = AsyncMock()
    trial._archive_step_outputs = MagicMock()

    step_result = StepResult(step_name=first.name)

    await trial._run_step(first, step_result, index=1, total=2)

    trial._archive_step_outputs.assert_called_once_with(
        first, preserve_agent=True, carry_agent_memory=True
    )


@pytest.mark.asyncio
async def test_run_step_stops_final_separate_step_when_verifier_disabled() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.logger = MagicMock()
    events: list[str] = []
    stop_main_flags: list[bool] = []

    async def collect_step_artifacts(
        _step: StepConfig, *, stop_main_before_sidecars: bool
    ) -> Path:
        events.append("collect")
        stop_main_flags.append(stop_main_before_sidecars)
        return Path("/tmp/artifacts")

    async def stop_agent_environment() -> None:
        events.append("stop")

    async def run_step_verifier(*args, **kwargs) -> None:
        events.append("verify")

    def archive_step_outputs(
        _step: StepConfig,
        *,
        preserve_agent: bool = False,
        carry_agent_memory: bool = False,
    ) -> None:
        events.append("archive")

    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(disable=True),
        agent=SimpleNamespace(resume_trajectory=False, load_trajectory=None),
    )
    trial.task = SimpleNamespace(config=TaskConfig())
    trial._create_step_dirs = MagicMock()
    trial._prepare_step = AsyncMock()
    trial._run_step_agent = AsyncMock()
    trial._upload_agent_logs = AsyncMock()
    trial._collect_step_artifacts = AsyncMock(side_effect=collect_step_artifacts)
    trial._stop_agent_environment = AsyncMock(side_effect=stop_agent_environment)
    trial._run_step_verifier = AsyncMock(side_effect=run_step_verifier)
    trial._archive_step_outputs = MagicMock(side_effect=archive_step_outputs)

    step = StepConfig(
        name="agent",
        verifier=VerifierConfig(environment_mode=VerifierEnvironmentMode.SEPARATE),
    )
    step_result = StepResult(step_name=step.name)

    await trial._run_step(step, step_result, index=2, total=2)

    assert events == ["collect", "stop", "verify", "archive"]
    assert stop_main_flags == [True]
    trial._run_step_verifier.assert_awaited_once_with(
        step,
        step_result,
        artifacts_dir=Path("/tmp/artifacts"),
        mode=VerifierEnvironmentMode.SEPARATE,
    )


@pytest.mark.asyncio
async def test_run_step_verifier_returns_when_verifier_disabled() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(disable=True),
        agent=SimpleNamespace(resume_trajectory=False),
    )
    trial._emit = AsyncMock()
    trial._run_shared_verifier = AsyncMock()
    trial._run_separate_verifier = AsyncMock()

    step = StepConfig(name="agent")
    step_result = StepResult(step_name=step.name)

    await trial._run_step_verifier(
        step,
        step_result,
        artifacts_dir=Path("/tmp/artifacts"),
        mode=VerifierEnvironmentMode.SHARED,
    )

    assert step_result.verifier is None
    trial._emit.assert_not_awaited()
    trial._run_shared_verifier.assert_not_awaited()
    trial._run_separate_verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_step_verifier_records_verifier_errors() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.config = SimpleNamespace(verifier=SimpleNamespace(disable=False, env={}))
    trial._emit = AsyncMock()
    trial._step_verifier_user = MagicMock(return_value=None)
    trial._step_verifier_timeout_sec = MagicMock(return_value=10)
    trial._reset_shared_step_verifier_dirs = AsyncMock()
    trial._run_shared_verifier = AsyncMock(side_effect=RuntimeError("missing reward"))

    step = StepConfig(name="agent")
    step_result = StepResult(step_name=step.name)

    await trial._run_step_verifier(
        step,
        step_result,
        artifacts_dir=Path("/tmp/artifacts"),
        mode=VerifierEnvironmentMode.SHARED,
    )

    assert step_result.exception_info is not None
    assert step_result.exception_info.exception_type == "RuntimeError"
    assert step_result.verifier is not None
    assert step_result.verifier.finished_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_error", "exception_type"),
    [
        (AgentTimeoutError("timed out"), "AgentTimeoutError"),
        (NonZeroAgentExitCodeError("exit 1"), "NonZeroAgentExitCodeError"),
    ],
)
async def test_run_step_agent_records_recoverable_agent_errors(
    agent_error: Exception,
    exception_type: str,
) -> None:
    trial = object.__new__(MultiStepTrial)
    trial.task = MagicMock()
    trial.task.step_instruction.return_value = "do the step"
    trial._step_agent_timeout_sec = MagicMock(return_value=10)
    trial._step_agent_user = MagicMock(return_value="agent")
    trial._run_agent_phase = AsyncMock(side_effect=agent_error)
    trial._sync_agent_output = AsyncMock()

    step = StepConfig(name="agent")
    step_result = StepResult(step_name=step.name)

    await trial._run_step_agent(step, step_result, resume=False)

    assert step_result.exception_info is not None
    assert step_result.exception_info.exception_type == exception_type
    trial._sync_agent_output.assert_awaited_once_with(step_result)


@pytest.mark.asyncio
async def test_run_step_agent_uses_resume() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.task = MagicMock()
    trial.task.step_instruction.return_value = "continue the work"
    current = StepResult(step_name="agent")
    trial._step_agent_timeout_sec = MagicMock(return_value=10)
    trial._step_agent_user = MagicMock(return_value="agent")
    trial._run_agent_phase = AsyncMock()
    trial._sync_agent_output = AsyncMock()

    step = StepConfig(name="agent")

    await trial._run_step_agent(step, current, resume=True)

    trial._run_agent_phase.assert_awaited_once_with(
        target=current,
        instruction="continue the work",
        timeout_sec=10,
        user="agent",
        step_cfg=step,
        resume=True,
        load=False,
    )
    trial._sync_agent_output.assert_awaited_once_with(current)


@pytest.mark.asyncio
async def test_run_step_agent_uses_load() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.task = MagicMock()
    trial.task.step_instruction.return_value = "pick up the prior session"
    current = StepResult(step_name="agent")
    trial._step_agent_timeout_sec = MagicMock(return_value=10)
    trial._step_agent_user = MagicMock(return_value="agent")
    trial._run_agent_phase = AsyncMock()
    trial._sync_agent_output = AsyncMock()

    step = StepConfig(name="agent")

    await trial._run_step_agent(step, current, resume=False, load=True)

    trial._run_agent_phase.assert_awaited_once_with(
        target=current,
        instruction="pick up the prior session",
        timeout_sec=10,
        user="agent",
        step_cfg=step,
        resume=False,
        load=True,
    )


def test_step_loads_only_first_step() -> None:
    trial = object.__new__(MultiStepTrial)

    trial._load_trajectory = Path("s.jsonl")
    assert trial._step_loads(1) is True
    assert trial._step_loads(2) is False

    trial._load_trajectory = None
    assert trial._step_loads(1) is False


def test_validate_resume_support_rejects_unsupported_agent() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.config = SimpleNamespace(agent=SimpleNamespace(resume_trajectory=True))
    trial.task = SimpleNamespace(
        config=TaskConfig(steps=[StepConfig(name="a"), StepConfig(name="b")])
    )
    trial.agent = MagicMock(capabilities=AgentCapabilities())
    trial.agent.name.return_value = "no-resume"

    with pytest.raises(ValueError, match="does not support resume"):
        trial._validate_resume_support()


def test_validate_resume_support_ignores_single_entry_steps() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.config = SimpleNamespace(agent=SimpleNamespace(resume_trajectory=True))
    trial.task = SimpleNamespace(config=TaskConfig(steps=[StepConfig(name="only")]))
    trial.agent = MagicMock(capabilities=AgentCapabilities())

    trial._validate_resume_support()


def test_validate_resume_support_rejects_unsupported_agent_for_per_step_override() -> (
    None
):
    # Run-level resume is OFF, but a single step opts in; an agent without
    # native resume support must still fail fast.
    trial = object.__new__(MultiStepTrial)
    trial.config = SimpleNamespace(agent=SimpleNamespace(resume_trajectory=False))
    trial.task = SimpleNamespace(
        config=TaskConfig(
            steps=[
                StepConfig(name="a"),
                StepConfig(name="b", resume_trajectory=True),
            ]
        )
    )
    trial.agent = MagicMock(capabilities=AgentCapabilities())
    trial.agent.name.return_value = "no-resume"

    with pytest.raises(ValueError, match="does not support resume"):
        trial._validate_resume_support()


def test_step_resumes_only_after_first_step() -> None:
    trial = object.__new__(MultiStepTrial)
    trial.task = SimpleNamespace(
        config=TaskConfig(
            steps=[
                StepConfig(name="a"),
                StepConfig(name="b"),
                StepConfig(name="c"),
            ]
        )
    )

    # No per-step override: every step inherits the run-level flag, and the
    # first step never resumes regardless.
    trial.config = SimpleNamespace(agent=SimpleNamespace(resume_trajectory=True))
    assert trial._step_resumes(1) is False
    assert trial._step_resumes(2) is True
    assert trial._step_resumes(3) is True

    trial.config = SimpleNamespace(agent=SimpleNamespace(resume_trajectory=False))
    assert trial._step_resumes(1) is False
    assert trial._step_resumes(2) is False
    assert trial._step_resumes(3) is False


def test_step_resumes_honors_per_step_override() -> None:
    trial = object.__new__(MultiStepTrial)
    # Run-level resume is ON, but step "c" opts out; step "b" stays None.
    trial.task = SimpleNamespace(
        config=TaskConfig(
            steps=[
                StepConfig(name="a"),
                StepConfig(name="b"),
                StepConfig(name="c", resume_trajectory=False),
            ]
        )
    )
    trial.config = SimpleNamespace(agent=SimpleNamespace(resume_trajectory=True))
    assert trial._step_resumes(1) is False
    assert trial._step_resumes(2) is True
    assert trial._step_resumes(3) is False


def test_step_resumes_per_step_override_can_enable_when_global_off() -> None:
    trial = object.__new__(MultiStepTrial)
    # Run-level resume is OFF, but step "b" opts in explicitly.
    trial.task = SimpleNamespace(
        config=TaskConfig(
            steps=[
                StepConfig(name="a"),
                StepConfig(name="b", resume_trajectory=True),
                StepConfig(name="c"),
            ]
        )
    )
    trial.config = SimpleNamespace(agent=SimpleNamespace(resume_trajectory=False))
    assert trial._step_resumes(1) is False
    assert trial._step_resumes(2) is True
    assert trial._step_resumes(3) is False


def test_step_resumes_first_step_override_is_ignored() -> None:
    trial = object.__new__(MultiStepTrial)
    # Even an explicit True on the first step cannot make it resume.
    trial.task = SimpleNamespace(
        config=TaskConfig(
            steps=[
                StepConfig(name="a", resume_trajectory=True),
                StepConfig(name="b"),
            ]
        )
    )
    trial.config = SimpleNamespace(agent=SimpleNamespace(resume_trajectory=False))
    assert trial._step_resumes(1) is False


def test_archive_step_outputs_can_preserve_live_agent_dir(tmp_path: Path) -> None:
    trial = object.__new__(MultiStepTrial)
    trial.paths = TrialPaths(tmp_path)
    trial.agent_env_paths = EnvironmentPaths()
    trial._artifact_handler = MagicMock()

    trial.paths.mkdir()
    (trial.paths.agent_dir / "session.jsonl").write_text("{}")
    (trial.paths.verifier_dir / "reward.txt").write_text("1")

    def move_dir_contents(src: Path, dst: Path) -> None:
        from harbor.trial.artifact_handler import ArtifactHandler

        ArtifactHandler.move_dir_contents(src, dst)

    trial._artifact_handler.move_dir_contents.side_effect = move_dir_contents

    trial._archive_step_outputs(StepConfig(name="first"), preserve_agent=True)

    assert (trial.paths.agent_dir / "session.jsonl").read_text() == "{}"
    assert (trial.paths.step_agent_dir("first") / "session.jsonl").read_text() == "{}"
    assert not (trial.paths.verifier_dir / "reward.txt").exists()


def _trial_for_archive(tmp_path: Path, *, mounted: bool) -> MultiStepTrial:
    from harbor.trial.artifact_handler import ArtifactHandler

    trial = object.__new__(MultiStepTrial)
    trial.paths = TrialPaths(tmp_path)
    trial.agent_env_paths = EnvironmentPaths()
    trial.agent_environment = MagicMock()
    trial.agent_environment.capabilities.mounted = mounted
    trial._artifact_handler = MagicMock()
    trial._artifact_handler.move_dir_contents.side_effect = (
        ArtifactHandler.move_dir_contents
    )
    trial._artifact_handler.move_dir_contents_preserving.side_effect = (
        ArtifactHandler.move_dir_contents_preserving
    )
    trial.paths.mkdir()
    return trial


@pytest.mark.parametrize("mounted", [True, False])
def test_archive_step_outputs_carries_auto_memory_only_when_mounted(
    tmp_path: Path, mounted: bool
) -> None:
    trial = _trial_for_archive(tmp_path, mounted=mounted)
    memory = trial.paths.agent_dir / "sessions" / "projects" / "-app" / "memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("deck conventions")
    (trial.paths.agent_dir / "sessions" / "session.jsonl").write_text("{}")

    trial._archive_step_outputs(StepConfig(name="first"), carry_agent_memory=True)

    # The archive always holds the step's own snapshot of the notebook.
    archived = trial.paths.step_agent_dir("first") / "sessions"
    assert (archived / "projects" / "-app" / "memory" / "MEMORY.md").read_text() == (
        "deck conventions"
    )
    assert (archived / "session.jsonl").read_text() == "{}"

    # A mounted environment shares this dir with the next step, so the notebook
    # is put back; elsewhere the wipe/restore runs inside the environment.
    if mounted:
        assert (memory / "MEMORY.md").read_text() == "deck conventions"
        assert not (trial.paths.agent_dir / "sessions" / "session.jsonl").exists()
    else:
        assert not memory.exists()


def test_archive_step_outputs_leaves_auto_memory_alone_on_final_step(
    tmp_path: Path,
) -> None:
    trial = _trial_for_archive(tmp_path, mounted=True)
    memory = trial.paths.agent_dir / "sessions" / "projects" / "-app" / "memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("deck conventions")

    trial._archive_step_outputs(StepConfig(name="last"), carry_agent_memory=False)

    assert not memory.exists()
