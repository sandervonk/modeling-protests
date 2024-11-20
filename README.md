# Setup

## Module Installation
Install the required python package
```
pip3 install networkx matplotlib scipy numpy tqdm
```

**ffmpeg** also needs to be installed to create the videos
- run `brew install ffmpeg` on mac / unix _(or follow the guide if you don't have homebrew installed)_
- run `sudo apt install ffmpeg` on umbutu, debian
- follow [`this guide`](https://www.hostinger.com/tutorials/how-to-install-ffmpeg) on windows or other systems


# Running the model

## Model configuration and options
Model options can be configured in the config.json file. The options are as follow:
- **STEPS**: (int) the number of steps to run the model for
- **DPI**: (int) the dots per inch to save matplotlib renders at
- **FPS**: (int) the frames (steps) per second that the final video will play at; this will be corrected for the skip factor
- **SKIP**: (int) the number of steps to skip between rendered frames. default is 1
- Model: (dict) options for the model, as follows
    - Size: the "shape" or "size" of the initial model
        - **total**: (int) full number of nodes for the network to reach during the build stage
        - **seed**: (int) the number of disconnected verts to use for the network building method
    - Seed: the number of each type to start with
        - **UNINFORMED** (int): Number of uninformed protesters (S.I.R. model part) to start with
        - **POTENTIAL** (int): Number of potential protesters to start with
        - **NEW** (int): Number of new protesters to start with
        - **MATURE** (int): Number of mature protesters to start with
        - **RETIRED** (int): Number of retired protesters to start with
- Runs: maps names of runs (anything you choose) to their model coefficient options; _e.g. in the default config, two runs will occur on each model run (both will run on the same initial seeded social network shape), "plot" and "plot2", and save their respective files to the `./out/` folder (see [render and save](#))._ Each name key corresponds to a dictionary of the following constant values:
    - **delta_2** (float): Withdrawal rate for mature protestors
    - **delta_1** (float): Withdrawal rate for new protestors
    - **chi** (float): Rate of new protestors turning into mature protestors
    - **gamma** (float): Rate of retired protestors turining into potential protestors
    - **beta_1** (float): Attractiveness to become protestors from new protestors
    - **beta_2** (float): Attractiveness to become protestors from mature protestors
    - **inform_lone** (float): Probability of lone uninformed nodes learning about protests (_e.g. online_)
    - **inform_each** (float): Contribution of neighbor nodes to converting uninformed nodes
    - **forget** (float): Probability of forgetting experiences and becoming uninformed in retirement, corrected for other prob format


## Render and save
Runs the model with the options in `config.json` and saves outputs to `./out/`:
- `{code}.jpg` - graph of counts of each code by step
- `{code}.mp4` - network visualization
- `{code}.json` - step count data used for graphs

```
python3 graphs.py
```


## View interactive graph data
Opens an interactive matplotlib GUI for exploring the plot of code counts over time, **after having run the model via `graphs.py`**
```
python3 show.py [run key]
``` 
_e.g. try `python3 show.py plot` or `python3 show.py plot2` with the default configuration_