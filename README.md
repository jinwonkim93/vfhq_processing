# VFHQ preprocessing
This repository provides tools for preprocessing videos for VFHQ dataset
https://liangbinxie.github.io/projects/vfhq/

## Downloading videos and cropping according to precomputed bounding boxes
1. Install metadata
```
https://onedrive.live.com/?cid=aaca1803f11f470d&id=AACA1803F11F470D%211000&authkey=!ALmyA3IelxWV2iw

unzip meta_info.zip -> meta_info
```

2. run python script
```
python extract_high_quality_faces_with_meta.py --metadata meta_info --workers 8
```
