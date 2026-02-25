# Self-Update + Self-Restart Design

**Date:** 2026-02-25
**Goal:** Let rumi modify her own source code and restart herself on a bare-metal Debian VPS.

## Safety Model

1. **Git safety net** — every self-modification auto-commits before and after the change
2. **Syntax validation** — `.py` files validated with `compile()` before writing
3. **Crash = stop** — wrapper only restarts on exit code 42. Any other exit stops the loop.
4. **Protected files** — `.env`, `.git/`, `run.sh` cannot be modified
5. **Restart cooldown** — 30s minimum between restarts

## Built-in Tools

### `update_source(path, content)`

- Writes `content` to `path` (relative to project dir)
- Rejects: paths outside project dir, `.env`, `.git/*`, `run.sh`
- For `.py` files: validates syntax with `compile()` before writing
- Auto-commits previous state: `git commit -m "pre-update backup: <file>"`
- Writes new content
- Auto-commits update: `git commit -m "self-update: <file>"`
- Returns success/failure with diff summary

### `restart_self(reason)`

- Logs the reason
- Writes `.restart_reason` marker file
- Calls `sys.exit(42)`
- Wrapper catches exit code 42 and restarts

## Process Supervision

```
systemd (ruminate.service)
  └── run.sh (restart loop, exit 42 = restart, 30s cooldown)
       └── python -u ruminate.py
```

## File Changes

| File | Change |
|------|--------|
| `ruminate.py` | Add `update_source` and `restart_self` built-in tools |
| `run.sh` | New — wrapper script with restart loop |
| `ruminate.service` | New — systemd unit file |
