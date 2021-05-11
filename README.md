# Social distancing monitor
In this project, computer vision techniques are employed to monitor compliance with the social distancing protocol. In addition, some statistical data are obtained.

[![Demonstration](/media/video.png)](https://youtu.be/Wy0imwCRsg0)
Jupyter notebook [here](https://colab.research.google.com/github/doopees/social-distancing-monitor/blob/main/notebook/social_distancing_monitor.ipynb)

## Features
* Get analytics such as
    - Number of people at risk in a given frame
    - Average exposure time for a person
    - Average number of people a person is exposed to
* Identify people at higher risk
* Identify the times with the highest number of people at risk
* Visualize statistical graphs

## Instalation

### Install dependencies
#### Essential packages
* OpenCV\
        `conda install -c conda-forge opencv`
* NumPy\
        `conda install -c conda-forge numpy`
* pandas\
        `conda install -c conda-forge pandas`
#### Optional packages for data visualization
* Matplotlib\
        `conda install -c conda-forge matplotlib`
* seaborn\
        `conda install -c conda-forge seaborn`

### Clone the repository
        git clone https://github.com/doopees/social-distancing-monitor
        cd social-distancing-monitor/

### Get the dataset
        !wget https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi
        !wget https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentre-groundtruth.top

## Running the project
        python3 -i main.py
Tested on Ubuntu 18.04 using Python 3.7.

## Disclaimer
This project does not attempt to measure the exact metric distance between people, but merely to provide an estimate of safe distance compliance.

## Credits
[pyimagesearch](https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/)\
[Landing AI](https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/)\
[Active Vision Laboratory](https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html)
