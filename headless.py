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


def remove_repeat_data():
    seen = False
    for entry in os.listdir("./out"):
        path = f"./out/{entry}"
        if os.path.isdir(path):
            shutil.rmtree(path)
            seen = True
    print("* Removed repeat data" if seen else "* No repeat data to remove")


if len(sys.argv) == 1:
    run_all(frames=False)
elif sys.argv[1] == "remove":
    remove_run_data()
    remove_repeat_data()
elif sys.argv[1] == "render":
    render_all()
else:
    try:
        run_all(frames=False, num=int(sys.argv[1]))
    except ValueError:
        raise TypeError("Frame count must be castable to int e.g. 'python3 headless.py 2000'")

    if "render" in sys.argv:
        try:
            render_all(num=int(sys.argv[1]))
        except ValueError:
            render_all()

    if "remove" in sys.argv:
        remove_run_data()
        remove_repeat_data()
