"""
Install llm-behavior-eval skills into your coding assistant's global context.

Supported targets:
  claude    — ~/.claude/commands/     (.md slash-command files)
  codex     — ~/.codex/AGENTS.md      (appended block)
  cursor    — ~/.cursor/rules/        (.mdc rule files)
  opencode  — ~/.opencode/AGENTS.md   (appended block)
  all       — all of the above
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path
from typing import Annotated, NamedTuple

import typer

app = typer.Typer(
    help="Install llm-behavior-eval skills into your coding assistant.",
    add_completion=False,
)

_MARKER_BEGIN = "<!-- lbe-skills-begin -->"
_MARKER_END = "<!-- lbe-skills-end -->"

_VALID_TARGETS = {"claude", "codex", "cursor", "opencode", "all"}


class _Skill(NamedTuple):
    module: str        # filename stem under llm_behavior_eval/skills/
    cmd_name: str      # slash-command / file name (e.g. "lbe-eval-assistant")
    description: str   # used in Cursor .mdc frontmatter


_SKILLS: list[_Skill] = [
    _Skill(
        module="lbe_eval_assistant",
        cmd_name="lbe-eval-assistant",
        description="Guide me through configuring and running an llm-behavior-eval evaluation",
    ),
    _Skill(
        module="lbe_results_analysis",
        cmd_name="lbe-results-analysis",
        description="Help me read and interpret llm-behavior-eval output files and metrics",
    ),
]


def _read(module: str) -> str:
    return (
        files("llm_behavior_eval.skills")
        .joinpath(f"{module}.md")
        .read_text(encoding="utf-8")
    )


def _write_file(dest: Path, content: str, label: str, force: bool) -> None:
    if dest.exists() and not force:
        typer.echo(f"  Skipped {label} — already exists (pass --force to overwrite)")
        return
    verb = "Overwrote" if dest.exists() else "Installed"
    dest.write_text(content, encoding="utf-8")
    typer.echo(f"  {verb} {label} → {dest}")


def _install_claude(force: bool) -> None:
    target_dir = Path.home() / ".claude" / "commands"
    target_dir.mkdir(parents=True, exist_ok=True)
    for skill in _SKILLS:
        _write_file(
            target_dir / f"{skill.cmd_name}.md",
            _read(skill.module),
            f"/{skill.cmd_name}",
            force,
        )


def _install_cursor(force: bool) -> None:
    target_dir = Path.home() / ".cursor" / "rules"
    target_dir.mkdir(parents=True, exist_ok=True)
    for skill in _SKILLS:
        content = (
            f"---\ndescription: {skill.description}\nalwaysApply: false\n---\n"
            + _read(skill.module)
        )
        _write_file(target_dir / f"{skill.cmd_name}.mdc", content, skill.cmd_name, force)


def _install_agents_md(target: Path, label: str, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_text(encoding="utf-8") if target.exists() else ""

    combined = "\n\n---\n\n".join(_read(s.module) for s in _SKILLS)
    new_block = f"{_MARKER_BEGIN}\n{combined}\n{_MARKER_END}"

    if _MARKER_BEGIN in existing:
        if not force:
            typer.echo(
                f"  Skipped {label} — lbe skills already present in {target} "
                "(pass --force to overwrite)"
            )
            return
        updated = re.sub(
            rf"{re.escape(_MARKER_BEGIN)}.*?{re.escape(_MARKER_END)}",
            new_block,
            existing,
            flags=re.DOTALL,
        )
        target.write_text(updated, encoding="utf-8")
        typer.echo(f"  Overwrote lbe skills block in {target}")
    else:
        with target.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write(f"\n{new_block}\n")
        typer.echo(f"  Appended lbe skills → {target}")


@app.command()
def main(
    target: Annotated[
        str,
        typer.Option(
            "--target",
            "-t",
            help="Coding assistant: 'claude', 'codex', 'cursor', 'opencode', or 'all'.",
            show_default=True,
        ),
    ] = "claude",
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing installations."),
    ] = False,
) -> None:
    """Install lbe-eval-assistant and lbe-results-analysis skills globally."""
    if target not in _VALID_TARGETS:
        typer.echo(
            f"Unknown target '{target}'. Choose from: {', '.join(sorted(_VALID_TARGETS))}",
            err=True,
        )
        raise typer.Exit(code=1)

    if target in {"claude", "all"}:
        typer.echo("Installing for Claude Code (~/.claude/commands/)...")
        _install_claude(force)

    if target in {"codex", "all"}:
        typer.echo("Installing for Codex (~/.codex/AGENTS.md)...")
        _install_agents_md(Path.home() / ".codex" / "AGENTS.md", "Codex", force)

    if target in {"cursor", "all"}:
        typer.echo("Installing for Cursor (~/.cursor/rules/)...")
        _install_cursor(force)

    if target in {"opencode", "all"}:
        typer.echo("Installing for OpenCode (~/.opencode/AGENTS.md)...")
        _install_agents_md(Path.home() / ".opencode" / "AGENTS.md", "OpenCode", force)

    typer.echo("Done.")


if __name__ == "__main__":
    app()
