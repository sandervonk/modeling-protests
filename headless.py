from graphs import run_all, render_all
import sys
import os
import shutil


def remove_run_data():
    if os.path.exists("./runs"):
        shutil.rmtree("./runs")
        print("* Removed run data")
    else:
        print("* No run data to remove")


if len(sys.argv) == 1:
    run_all(frames=False)
elif sys.argv[1] == "remove":
    remove_run_data()
elif sys.argv[1] == "render":
    render_all()
else:
    try:
        run_all(frames=False, num=int(sys.argv[1]))
    except ValueError:
        raise TypeError("Frame count must be castable to int e.g. 'python3 headless.py 2000'")

    if "render" in sys.argv:
        try:
            render_all(frames=int(sys.argv[1]))
        except ValueError:
            render_all()

    if "remove" in sys.argv:
        remove_run_data()
