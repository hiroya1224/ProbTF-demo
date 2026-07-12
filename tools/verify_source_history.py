#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCES = (
    {
        "name": "probik_demo",
        "head": "c06a2eabc258fbf55fe167f71a0253afa6b24e0b",
        "root": "9f9dedfd68dd8cac61d8f94346f2893386c93fe8",
        "commits": 9,
        "files": 56,
        "import_commit": "0d6bfa0f4bc729a6f1c4bf97fe8c90bf90eead51",
        "path": "ros/symaware_grasp",
    },
    {
        "name": "deflecomp",
        "head": "1508cac426b68d2e08546243223cf90ac200b7ce",
        "root": "a770384cf0e1a639a20068da5ae28673b9204bea",
        "commits": 25,
        "files": 88,
        "import_commit": "a2e8429cb56055bf7c8afac6b5b47a6aa8b9ba49",
        "path": "ros/deflecomp",
    },
)


def git(*arguments):
    return subprocess.check_output(
        ["git", *arguments],
        cwd=str(REPOSITORY),
        text=True,
    ).strip()


def verify(source):
    errors = []
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source["head"], "HEAD"],
            cwd=str(REPOSITORY),
            check=True,
        )
    except subprocess.CalledProcessError:
        errors.append("source HEAD is not an ancestor of HEAD")

    source_tree = git("rev-parse", f"{source['head']}^{{tree}}")
    imported_tree = git("rev-parse", f"{source['import_commit']}:{source['path']}")
    if source_tree != imported_tree:
        errors.append(f"tree mismatch: {source_tree} != {imported_tree}")

    roots = git("rev-list", "--max-parents=0", source["head"]).splitlines()
    if roots != [source["root"]]:
        errors.append(f"unexpected root commits: {roots}")

    commit_count = int(git("rev-list", "--count", source["head"]))
    if commit_count != source["commits"]:
        errors.append(f"commit count mismatch: {commit_count} != {source['commits']}")

    file_count = len(git("ls-tree", "-r", "--name-only", source["head"]).splitlines())
    if file_count != source["files"]:
        errors.append(f"file count mismatch: {file_count} != {source['files']}")
    return source_tree, errors


def main():
    failed = False
    for source in SOURCES:
        tree, errors = verify(source)
        if errors:
            failed = True
            for error in errors:
                print(f"FAIL {source['name']}: {error}")
        else:
            print(
                f"OK {source['name']}: {source['commits']} commits, "
                f"{source['files']} files, tree {tree}"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
