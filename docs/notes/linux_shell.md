# Linux / Shell Notes

Keep this file compact. Record reusable concepts, not every command encountered.

## PATH

```bash
echo "$PATH"
export PATH=/usr/local/cuda/bin:$PATH
```

- `$VAR`: expand a shell variable.
- `:`: separator between directories in `PATH`.
- `PATH`: list of directories searched for executable commands.
- placing a new directory before `$PATH` gives it higher lookup priority.
- `export`: makes a shell variable part of the environment inherited by child processes.

## LD_LIBRARY_PATH

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Used for runtime shared-library lookup.

It is different from `PATH`, which is for executable commands.

## Useful Troubleshooting Commands

### Find an executable

```bash
which nvcc
find /usr/local -name nvcc 2>/dev/null
```

### Inspect installed packages

```bash
dpkg -l | grep cuda
dpkg -l | grep tensorrt
```

### Inspect environment variables

```bash
echo "$PATH"
echo "$LD_LIBRARY_PATH"
env | grep -i proxy
```

### Inspect processes

```bash
ps aux
pgrep -af sshuttle
```

### Inspect listening ports

```bash
ss -lntp
ss -lntp | grep 1080
```

### Inspect systemd services

```bash
systemctl status <service>
systemctl is-active <service>
journalctl -u <service>
```

## Redirection and Pipes

```text
>        overwrite stdout to a file
>>       append stdout to a file
2>       redirect stderr
2>/dev/null
         discard stderr
2>&1     redirect stderr to the same destination as stdout
|        pipe stdout into another command
```

Example:

```bash
find /usr/local -name nvcc 2>/dev/null
```

`2>/dev/null` hides permission/error messages from stderr.

## Command Chaining

```text
cmd1 && cmd2
```

Run `cmd2` only if `cmd1` succeeds.

```text
cmd1 || cmd2
```

Run `cmd2` only if `cmd1` fails.

```text
cmd1 ; cmd2
```

Run `cmd2` regardless of whether `cmd1` succeeds.

## Learning Rule

When first encountering a Shell construct:

1. understand what it does in the current command;
2. write 1–3 lines here if it is likely to recur;
3. return to the project.

Do not turn every first encounter into a long standalone tutorial.
