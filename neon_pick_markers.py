import numpy as np 
import pandas as pd 
import os, glob, pickle, json
import matplotlib.pyplot as plt 
from decord import VideoReader, cpu
import cv2

def pick_markers(event, x, y, flag, param):
    """ Mouse callback function to capture 4 points on a frame"""
    points, n_markers = param
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: {x}, {y}")
        
        if len(points) == n_markers:
            cv2.destroyAllWindows()

def click_markers(frame,n_markers):
    """ Function to display a frame and let the user click 4 points """
    points = []
    cv2.namedWindow("Click four markers")
    cv2.setMouseCallback("Click four markers", pick_markers, (points, n_markers))
    
    while len(points)<n_markers:
        temp_frame = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(temp_frame, p, 10, (0, 255, 0), -1)
            cv2.putText(temp_frame, f"{i+1}", p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Click four markers", temp_frame)
        if cv2.waitKey(1) & 0xFF == 27:     # ESC to exit early
            break   
    
    cv2.destroyAllWindows()
    return points

if __name__ == "__main__":
    ## load video 
    data_path = glob.glob("jiwon_eyes/2024-12-1814-24*")[0]
    video_path = glob.glob(data_path + '/*.mp4')[0]
    n_markers = 4
    video = VideoReader(video_path, ctx=cpu(0))
    frame_numbers = np.linspace(100, len(video)-100, 5).astype(int)
    marker_points = []
    for i in frame_numbers:
        frame = video[i].asnumpy()
        marker_points.append(click_markers(frame, n_markers))
        
    # save the marker points as json
    with open(data_path + "/marker_points.json", "w") as f:
        json.dump(marker_points, f)
    