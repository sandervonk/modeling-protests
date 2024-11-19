from graphs import drawGraphs
import sys
import json

if len(sys.argv) < 2:
    raise ValueError("Please provide a plot run name e.g. `python3 show.py plot`")

try:
    data = json.load(open(f'./out/{sys.argv[1]}.json', 'r'))
except FileNotFoundError:
    raise ValueError(f"Run '{sys.argv[1]}' does not exist or has not yet been generated")

drawGraphs(data["steps"], data["data"], show=True)
