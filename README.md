## Step 1
Downoload YOLO the configuration files from the following [source](https://drive.google.com/drive/folders/18XcIOBNQ6jmuJgwUnfgsuITGiXRWZXTa?usp=sharing) 


## Step 2 (optional)
Downoload example videos from the following [source](https://drive.google.com/drive/folders/1H6aSQCOa0DPso053M7oFhSzvZMJrmcY2?usp=sharing) 

## Step 3
Download the source code and prepare the followig project structure
```
.
├── fd_approach
│   └── fd_approach.py
├── nn_approach
│   ├── nn_approach.py
│   └── yolo_config
│       ├── coco.names
│       ├── yolov3.cfg
│       └── yolov3.weights
├── videos
│   └── example.mp4
├── requirements.txt
├── prepared_calls.txt
└── README.md
```

## Step 4
Install the necessary packeges (using pip):
```
pip install -r requirements.txt
```

## Step 5
Usage:
```
python fd_approach.py --help
```
or
```
python nn_approach.py --help
```

## Hint
You can find prepared calls in `prepared_calls.txt`