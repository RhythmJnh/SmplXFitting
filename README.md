# SmplXFitting
  SmplXFitting is a Pytorch implementation of fitting openpose keypoints to smpl.

# Install
  Create a virtual environment, and run:
  ```
    sh install.sh
  ```

# Usage
  Download the [smpl models](https://pan.baidu.com/s/1JTiyaxgocJgN2-6lSNE53A), unzip to models folder.  
  If setup properly, the folder structure should look like:
  ```
    .
    ├── models/
        ├── smpl/
            ├── %^%$^#%^$.pkl
            ├── %^&$^%$^%.pkl
            ├── ##^$$$^%&.pkl
            └── ...
  ```
  Configure openpose and setup properly, then use ```scripts/pose.py``` to generate keypoints.  
  Activate this environment, then run ```test.py```.  
  Please find more details in the code.
