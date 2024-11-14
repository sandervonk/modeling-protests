# modeling-protests

`pip install networkx matplotlib scipy`

**Install pygraphviz**  
https://pygraphviz.github.io/documentation/stable/install.html


# Create video from frames
`ffmpeg -framerate 20 -i ./plot/%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ./plot/out.mp4 -y`

# render and save
```python3 graphs.py && ffmpeg -framerate 20 -i ./plot/%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ./plot/out.mp4 -y```