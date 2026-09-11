"""Delete one task's outputs across every sub-task of every episode.

The per-task artifacts are distributed over the sub-task folders of
utils.astribot_paths,

    <episodes-root>/ep{ep:03d}/subtask_{k:02d}/<task>[/<camera>]

so re-running one task (after a parameter change, say) otherwise means
deleting it once per (episode, sub-task) by hand.  This tool loops the
tree instead: ``--task`` names the task dir to drop, every selected
episode is swept, and each matching ``<task>`` dir is removed whole — its
camera subdirs included; the sub-task folder itself and every other task
stay.  Nothing is recomputed: the next run of that step simply finds the
task dirs gone.

Episodes and their sub-task folders are discovered from the tree itself
(utils.astribot_paths.discover_episodes / discover_subtasks) rather than
from dataset splits, so a sweep covers a half-processed dataset — an
episode that never reached the task is reported, never a failure.
Deletion is irreversible: use ``--dry-run`` to see the list first.

Usage
-----
    # What would go (nothing is touched)
    python tools/astribot/clear_task.py --data-root /data/astri_making_coffee
        --task traces --dry-run

    # Drop every traces/ dir of episode 0
    python tools/astribot/clear_task.py --data-root /data/astri_making_coffee
        --task traces --episode-idxes 0
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from utils import astribot_paths as ap


def human_bytes(n: int) -> str:
    """Binary size, one decimal from KiB up."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def dir_size(path: Path) -> int:
    """Bytes under a task dir — files only, symlinks neither followed nor
    counted (their targets are not what this sweep deletes)."""
    total = 0
    stack = [path]
    while stack:
        for entry in os.scandir(stack.pop()):
            if entry.is_dir(follow_symlinks=False):
                stack.append(entry.path)
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
    return total


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Delete one task's outputs (camera subdirs included) "
                    "from the subtask_XX folders of every selected episode — "
                    "the episodes tree of utils.astribot_paths. Episodes and "
                    "sub-tasks are discovered from the tree; nothing is "
                    "recomputed."
    )
    parser.add_argument("--data-root", "-d", required=True,
                        help="root passed to LeRobotDataset; the episodes "
                             "root defaults to <data-root>/episodes")
    parser.add_argument("--task", "-t", required=True, choices=ap.TASKS,
                        help="task dir to delete under each subtask_XX "
                             "folder")
    parser.add_argument("--episode-idxes", "-e", nargs="*", type=int,
                        default=None,
                        help="only sweep these episode indices (default: "
                             "every episode under the episodes root)")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="episodes root (default: <data-root>/episodes)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="list what would be removed and remove nothing")
    return parser.parse_args(argv)


class TaskClearer:
    """Delete one task dir across the sub-tasks of the selected episodes.

    Standalone: only the tree is consulted — episodes, their sub-task
    folders and each task dir's size are read from disk, so the sweep
    never has to match an earlier run's selection flags.
    """

    def __init__(self, args):
        self.args = args
        self.root = ap.episodes_root(args.data_root, args.out_dir)
        self.task = args.task
        self.n_dirs = 0
        self.n_bytes = 0

    def run(self) -> None:
        discovered = ap.discover_episodes(self.root)
        if not discovered:
            print(f"\nno episodes under {self.root} — nothing to clear")
            return
        eps = self._select_episodes(discovered)
        if self.args.dry_run:
            print("\ndry run — nothing will be removed")
        print(f"\n{len(eps)} episode(s) selected, task {self.task!r}:")
        for ep_idx in eps:
            self._clear_episode(ep_idx)
        verb = "would remove" if self.args.dry_run else "removed"
        print(f"\ndone: {verb} {self.n_dirs} {self.task!r} dir(s) "
              f"({human_bytes(self.n_bytes)}) under {self.root}")

    def _select_episodes(self, discovered: list[int]) -> list[int]:
        """The discovered episodes, filtered by -e (a requested episode
        absent from the tree is warned about and skipped, never a
        failure)."""
        if self.args.episode_idxes is None:
            return discovered
        requested = set(self.args.episode_idxes)
        for ep in sorted(requested - set(discovered)):
            print(f"  episode {ep}: not under {self.root} — skipped")
        return [ep for ep in discovered if ep in requested]

    def _clear_episode(self, ep_idx: int) -> None:
        """Remove this episode's <task> dir of every sub-task folder it
        has; an episode without the task is reported, not an error."""
        ep = ap.episode_name(ep_idx)
        found = False
        for k in ap.discover_subtasks(self.root, ep_idx):
            path = ap.task_dir(self.root, ep_idx, k, self.task)
            if path.is_symlink():
                # rmtree refuses symlinks, and a task dir is always a real
                # dir — never silently drop a path pointing elsewhere (this
                # also counts as found: the "no <task> dir(s)" line below
                # would otherwise contradict the skip message)
                found = True
                print(f"  {ep}/{ap.subtask_name(k)}/{self.task}: symlink "
                      f"-> {os.readlink(path)} — skipped, remove it by hand")
                continue
            if not path.is_dir():
                continue
            size = dir_size(path)
            if not self.args.dry_run:
                shutil.rmtree(path)
            found = True
            self.n_dirs += 1
            self.n_bytes += size
            print(f"  {ep}/{ap.subtask_name(k)}/{self.task}: "
                  f"{human_bytes(size)}")
        if not found:
            print(f"  {ep}: no {self.task} dir(s)")


def main(argv: list[str] | None = None) -> int:
    TaskClearer(parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
