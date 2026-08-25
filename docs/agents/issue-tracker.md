# Issue tracker: Local Markdown

Issues and PRDs for this repo live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The PRD is `.scratch/<feature-slug>/PRD.md`
- Implementation issues are `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Triage state is recorded as a `Status:` line near the top of each issue file; see `triage-labels.md` for the role strings
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/`, creating the directory if needed.

## When a skill says "fetch the relevant ticket"

Read the referenced file. The user will normally pass the path or issue number directly.

## Wayfinding operations

Used by `/wayfinder`. The map is a file with one child file per ticket.

- Map: `.scratch/<effort>/map.md`
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`
- Ticket type: a `Type:` line containing `research`, `prototype`, `grilling`, or `task`
- Ticket status: a `Status:` line containing `claimed` or `resolved`
- Blocking: a `Blocked by: NN, NN` line; a ticket is unblocked when every listed file is resolved
- Frontier: the first numbered open, unblocked, and unclaimed ticket
- Claim: set `Status: claimed` and save before work
- Resolve: append the result under `## Answer`, set `Status: resolved`, and add a context pointer to the map
