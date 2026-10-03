# research-lab

A long-lived home for new research ideas in causal representation learning.

Every idea lives in its own isolated folder under `ideas/<name>/`. An idea never
gets its own repository until it has actually succeeded. Until then it stays here,
where it is cheap to start and cheap to kill.

## Where things run

- **The Mac writes.** All code is written, edited, and pushed from a laptop.
- **The A100 runs.** Experiment scripts are pulled onto the A100 and run there by
  the researcher. Nothing in this repo runs experiments on the Mac. Long jobs use
  `nohup`.

## Three rules

1. **Ideas are fully isolated.** Each idea is one folder under `ideas/<name>/`.
   Code inside an idea imports only from within that same folder — no shared code,
   no cross-idea imports. This keeps every idea independently movable and killable.
2. **Never commit data or results.** The root `.gitignore` ignores `results/`,
   `data/`, and the usual array/model/tabular blobs. Keep it that way.
3. **Each idea has one plain-English `EXPLAINER.md`** that is append-only. Never
   edit or delete past lines — only add dated entries at the bottom. It says what
   the idea is, where it stands, and, if killed, why. It is the only status file;
   there is no separate technical state file.

## Workflows

### Add an idea

Copy an existing idea folder, then make it yours:

```bash
cp -r ideas/axis ideas/<newname>
```

Then:
- Rewrite the first entry of `ideas/<newname>/EXPLAINER.md` for the new idea.
- Clear out the copied code and start fresh.
- Add a line for it to the idea list below.

### Kill an idea

Add a dated, plain-English entry to its `EXPLAINER.md` saying why it is being
killed, then move it to the graveyard:

```bash
git mv ideas/<name> graveyard/<name>
```

### Graduate an idea

Once an idea has succeeded, split it into its own history and push it to a new repo:

```bash
git subtree split --prefix=ideas/<name> -b <name>-export
# then push <name>-export to a fresh repository
```

## Ideas

- **axis** — NOT STARTED — a simple test for whether a change in setting (scanner,
  lab batch) really changes how the system works, or only changes how it is measured.
