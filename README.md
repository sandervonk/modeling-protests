# Setup
## Module Installation
`pip install networkx matplotlib scipy threading`

# Running the model
## Render and save
```python3 graphs.py```

## Create video from frames only
`ffmpeg -framerate 20 -i ./plot/%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ./out/plot.mp4 -y`

