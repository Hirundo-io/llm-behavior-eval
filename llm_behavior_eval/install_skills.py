"""
Install llm-behavior-eval skills into your coding assistant's global context.

Supported targets:
  claude    — copies command files to ~/.claude/commands/
  codex     — appends skill sections to ~/.codex/AGENTS.md
  opencode  — appends skill sections to ~/.opencode/AGENTS.md
  cursor    — writes .mdc rule files to ~/.cursor/rules/
  all       — all of the above
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="Install llm-behavior-eval skills into your coding assistant.",
    add_completion=False,
)

# Maps Python module name → slash-command file name
_SKILLS: dict[str, str] = {
    "lbe_eval_assistant": "lbe-eval-assistant",
    "lbe_results_analysis": "lbe-results-analysis",
}

_MARKER_BEGIN = "<!-- lbe-skills-begin -->"
_MARKER_END = "<!-- lbe-skills-end -->"

_VALID_TARGETS = {"claude", "codex", "cursor", "opencode", "all"}

# Human-readable descriptions used in Cursor .mdc frontmatter
_CURSOR_DESCRIPTIONS: dict[str, str] = {
    "lbe_eval_assistant": (
        "Guide me through configuring and running an llm-behavior-eval evaluation"
    ),
    "lbe_results_analysis": (
        "Help me read and interpret llm-behavior-eval output files and metrics"
    ),
}


def _skill_content(module_name: str) -> str:
    return (
        files("llm_behavior_eval.skills")
        .joinpath(f"{module_name}.md")
        .read_text(encoding="utf-8")
    )


def _install_claude(force: bool) -> None:
    target_dir = Path.home() / ".claude" / "commands"
    target_dir.mkdir(parents=True, exist_ok=True)
    for module_name, cmd_name in _SKILLS.items():
        dest = target_dir / f"{cmd_name}.md"
        if dest.exists() and not force:
            typer.echo(
                f"  Skipped /{cmd_name} — already exists (pass --force to overwrite)"
            )
            continue
        dest.write_text(_skill_content(module_name), encoding="utf-8")
        verb = "Overwrote" if dest.exists() else "Installed"
        typer.echo(f"  {verb} /{cmd_name} → {dest}")


def _install_cursor(force: bool) -> None:
    target_dir = Path.home() / ".cursor" / "rules"
    target_dir.mkdir(parents=True, exist_ok=True)
    for module_name, cmd_name in _SKILLS.items():
        dest = target_dir / f"{cmd_name}.mdc"
        if dest.exists() and not force:
            typer.echo(
                f"  Skipped {cmd_name} — already exists (pass --force to overwrite)"
            )
            continue
        description = _CURSOR_DESCRIPTIONS[module_name]
        content = _skill_content(module_name)
        mdc = f"---\ndescription: {description}\nalwaysApply: false\n---\n{content}"
        dest.write_text(mdc, encoding="utf-8")
        verb = "Overwrote" if dest.exists() else "Installed"
        typer.echo(f"  {verb} {cmd_name} → {dest}")


def _install_agents_md(target: Path, label: str, force: bool) -> None:
    """Append or replace the lbe skills block in an AGENTS.md file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_text(encoding="utf-8") if target.exists() else ""

    combined = "\n\n---\n\n".join(_skill_content(m) for m in _SKILLS)
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
            help=(
                "Coding assistant to install for: "
                "'claude', 'codex', 'cursor', 'opencode', or 'all'."
            ),
            show_default=True,
        ),
    ] = "claude",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing installations.",
        ),
    ] = False,
) -> None:
    """Install lbe-eval-assistant and lbe-results-analysis skills globally."""
    if target not in _VALID_TARGETS:
        typer.echo(
            f"Unknown target '{target}'. "
            f"Choose from: {', '.join(sorted(_VALID_TARGETS))}",
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
        _install_agents_md(
            Path.home() / ".opencode" / "AGENTS.md", "OpenCode", force
        )

    typer.echo("Done.")


if __name__ == "__main__":
    app()
