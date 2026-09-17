# Contributing

Four of us work on this repo, and we all use coding agents. That means more branches, more
commits, and bigger diffs than a team this size would normally produce. These conventions exist
so we don't step on each other.

## Branches

- Name branches `<username>/<short-topic>` — e.g. `sam/parse-fixtures`.
- One topic per branch. If the agent wanders into a second topic, start a second branch.
- Never commit directly to `main`.
- Rebase onto the latest `main` before opening a PR.

## Pull requests

- Small enough for a teammate to review in one sitting — aim for under ~400 changed lines.
- One logical change per PR. Split generated work that grew past that.
- Open a draft PR early. It's the cheapest way to claim work before you generate a big diff.

## Review

- At least one other teammate must approve before merge. No self-approval.
- The author merges, once there's an approval and CI is green.
- Say in the PR description which parts an agent wrote and which parts you've actually read.

## What an agent may change without asking

Free rein:

- Files inside the module or directory the task is scoped to.
- Tests and docs for code it's already changing.
- Its own branch — commit, amend, force-push, whatever.

Ask a human first:

- Anything outside the task's directory: shared config, CI workflows, build scripts,
  dependency manifests, `README.md`, this file.
- Renaming or moving files other branches might be touching.
- Changing a shared interface (a function signature, a schema, a public API).
- Deleting anything not created on the current branch.

## Avoiding collisions

- Before starting, check open PRs and branches touching the same paths.
- Claim a file or module before generating a big diff there — a draft PR, or a one-line
  comment on the issue.
- Prefer narrow, single-purpose modules over shared "god files". Two agents editing one
  1000-line module is how we get a bad day.
- If a collision happens anyway, the second PR rebases onto the first.

## Commit messages

- One logical change per commit.
- Imperative subject line, under ~72 characters: `Add retry to fetch loop`.
- The body explains *why*, not *what* — the diff already says what.
- If a commit is agent-generated and a human hasn't reviewed it closely yet, say so with a
  trailer:

  ```
  Agent: claude-code
  ```

## Repo structure

The repo is currently just a README, a license, and a Python `.gitignore` — nothing has landed
yet. So this is the layout we're committing to, not a description of what's there. New files go
in these directories:

- `/src` — one subdirectory per component or module.
- `/tests` — mirrors `/src` one-to-one.
- `/scripts` — one-off and ops scripts.
- `/docs` — design notes and ADRs.
- `/data` — small sample and fixture data only. Never secrets, never generated output.

Inside an existing component directory, an agent can add and edit files freely. A **new
top-level directory needs a PR discussion first** — that's a structural decision, not a
file-placement one.
