# Contributing

Two-person workflow. `main` always works; every change goes through a pull request reviewed by the other person.

1. **Issue first.** Every task is a GitHub issue, assigned to one person and tracked on the project board
   (To do → In progress → In review → Done).
2. **Branch from `main`.** `git switch main && git pull`, then `git switch -c feature/face-detection`.
   Prefixes: `feature/`, `fix/`, `exp/` (experiments), `docs/`.
3. **Small commits** in English, imperative mood: `Add RetinaFace detector wrapper`.
4. **Open a PR** to `main` and write `Closes #<issue>` in the description. CI (ruff + pytest) must pass.
5. **Review.** The other person reviews (comment or approve), the author merges. No direct pushes to `main`.

## Conventions

- Images are BGR `uint8` `(H, W, 3)`, masks are `bool` `(H, W)`, boxes are `(x1, y1, x2, y2)`;
  see `src/anonymizer/types.py`.
- Never commit datasets, model weights or videos. Put them in `data/` (git-ignored) or a shared drive.
- Strip notebook outputs before committing: `nbstripout --install` once per clone.
- Run `ruff check . && ruff format . && pytest` before pushing.
