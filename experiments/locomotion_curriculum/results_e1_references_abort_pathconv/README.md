# E1 Phase A, attempt 1: infrastructure abort (no recordings)

All 91 recorder calls failed in 7.9 s with `CalledProcessError` before any gait was generated: the shell that
launched the runner (Git Bash) rewrote the Linux interpreter path passed to `--generator-python` into a Windows
path, so `wsl --exec timeout <python>` could not find it. No styled reference was produced or judged; nothing
here is a Phase A result. The rerun disables that path conversion; protocol, code and thresholds are unchanged.
