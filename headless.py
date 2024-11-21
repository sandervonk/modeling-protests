from graphs import run_all
import sys


if len(sys.argv) == 2:
    try:
        run_all(frames=False, num=int(sys.argv[1]))
    except ValueError:
        raise TypeError("Frame count must be castable to int e.g. 'python3 headless.py 2000'")
else:
    run_all(frames=False)
