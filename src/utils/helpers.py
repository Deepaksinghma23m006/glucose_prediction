import sys

def print_progress(current, total, bar_length=50):
    done = int(bar_length * current / total)
    sys.stdout.write(f"\r[{'=' * done}{' ' * (bar_length - done)}] {current} / {total} bytes downloaded")
    sys.stdout.flush()
