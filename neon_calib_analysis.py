""" 
    Run this code after running the calibration. If the light was not completely off during the calibration, 
    please run "neon_pick_markers.py" to pick the markers and then run this code.
    
    Jiwon Yeon, 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, json, cv2, pickle, re
# import argparse
from decord import VideoReader, cpu
from neon_pick_markers import click_markers, pick_markers

class MetaData:
    def __init__(self):
        self.data_dir = None
        self.pickle_dir = None

    def unpickle(self):
        if self.pickle_dir:
            with open(self.pickle_dir, 'rb') as f:
                data = CustomUnpickler(f).load()
                config = data['config']
                response = data['data']
        else:
            config, response = None, None
            
        return config, response
    
    def screen_markers(self):
        # check whether the marker_points.json exists
        marker_path = os.path.join(self.data_dir, 'marker_points.json')
        if os.path.exists(marker_path):
            with open(marker_path, 'r') as f:
                marker_points = json.load(f)
            # set four marker positions
            marker_points = np.array(marker_points)
            avg_marker_points = np.mean(marker_points, axis=0)
            
            self.markers = avg_marker_points
        else:
            self.markers = None
            
    def get_clean_gaze(self):
        # get blink removed gaze
        gaze = pd.read_csv(os.path.join(self.data_dir, 'gaze.csv'))
        blink = pd.read_csv(os.path.join(self.data_dir, 'blinks.csv'))
        
        # remove blinks from gaze data 
        gaze_clean = gaze
        for b in range(len(blink)):
            gaze_clean = gaze_clean[~((gaze_clean['timestamp [ns]'] >= blink['start timestamp [ns]'][b]) & 
                                    (gaze_clean['timestamp [ns]'] < blink['end timestamp [ns]'][b]))]
        
        return gaze_clean
    
    def find_target_pos(self, frame):        
        # filter the frame with white color
        lower_white = np.array([200, 200, 200])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(frame, lower_white, upper_white)
        frame_filtered = cv2.bitwise_and(frame, frame, mask=mask)

        # convert the frame         
        frame_gray = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
        frame_bitwise = cv2.bitwise_not(frame_gray)
        
        # crop the frame if needed 
        if self.markers is not None:
            min_x = int(np.min(self.markers[:, 0]))
            max_x = int(np.max(self.markers[:, 0]))
            min_y = int(np.min(self.markers[:, 1]))
            max_y = int(np.max(self.markers[:, 1]))
            frame_bitwise = frame_bitwise[min_y:max_y, min_x:max_x]
        
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0
        # params.filterByArea = True
        # params.filterByCircularity = True
        # params.minCircularity = .3
        # params.minArea = 15
        # params.maxArea = 200
        detector = cv2.SimpleBlobDetector_create(params)
        dots = detector.detect(frame_bitwise)
        
        target_pos = []
        for d in dots:
            x, y = d.pt
            if self.markers is not None:
                x, y = x + np.min(self.markers[:, 0]), y + np.min(self.markers[:, 1])
            target_pos.append([x, y])
            
        return target_pos

    def px_to_dva(self, x_px, y_px):
        # load camera intrinsics 
        camera_matrix = np.array(json.load(open(self.data_dir + '/scene_camera.json'))['camera_matrix'])
        distortion_coeffs = np.array(json.load(open(self.data_dir + '/scene_camera.json'))['distortion_coefficients'])
        
        pixel_coords = np.array([x_px, y_px], dtype=np.float32)

        # First, we need to undistort the points
        undistorted_point = cv2.undistortPoints(pixel_coords.reshape(1, 1, 2), camera_matrix, distortion_coeffs, P=camera_matrix)

        # Convert to normalized homogeneous coordinates
        norm_point = np.append(undistorted_point, 1)

        # transform to camera coordinates
        img_to_cam = np.linalg.inv(camera_matrix)
        cam_coords = np.dot(img_to_cam, norm_point)

        # Calculate elevation and azimuth based on the camera coordinates
        elevation = np.rad2deg(np.arctan2(-cam_coords[1], cam_coords[2]))
        azimuth = np.rad2deg(np.arctan2(cam_coords[0], cam_coords[2]))

        return azimuth, elevation
    
    def get_disposition(self, gaze, events):
        # Loop through events
        video = VideoReader(glob.glob(self.data_dir + '/*.mp4')[0], ctx=cpu(0))
        world_times = pd.read_csv(os.path.join(self.data_dir, 'world_timestamps.csv'))
        nEvents = int(re.search(r'trial (\d+), end', events['name'].iloc[-2]).group(1))
        
        # from the events, sort the target id 
        target_ids = events['name'][events['name'].str.contains(r'trial \d+, target location')].apply(lambda x: x.split(':')[1].replace(')', '').replace('(', '').strip(''))
        unique_targets = sorted(target_ids.unique())
        
        # assign the target id to the trials
        target_ids = target_ids.map({target: idx+1 for idx, target in enumerate(unique_targets)})
        target_ids = target_ids.values
        
        df = pd.DataFrame(columns=['trial', 'target id', 'target x [deg]', 'target y [deg]', 
                                   'target x [px]', 'target y [px]',
                                   'gaze x [px]', 'gaze y [px]', 
                                   'gaze x [deg]', 'gaze y [deg]',
                                   'distance [deg]'])
        
        for e in range(1, nEvents+1):
            event_start_time = events['timestamp [ns]'][events['name'].str.contains(f'trial {e}, target location*')].values[0]
            event_end_time = events['timestamp [ns]'][events['name'].str.contains(f'trial {e}, end')].values[0]
            
            # find the video frame
            frames = world_times[(world_times['timestamp [ns]'] >= event_start_time) & (world_times['timestamp [ns]'] <= event_end_time)].index
            
            # find the first frame to get the target position
            target_pos = []
            f = int(len(frames)-50)      # grab the frame in the middle 
            while len(target_pos) != 1: 
                target_pos = self.find_target_pos(video[frames[f]].asnumpy())
                f += 1 
                if f == len(frames):
                    break
            # target_pos = self.find_target_pos(video[frames[130]].asnumpy())
            
            # if unable to find the target position, manually set the target position 
            if len(target_pos) == 0:
                target_pos = click_markers(video[frames[-50]].asnumpy(), 1)
                target_pos = target_pos[0]
            else:
                target_pos = target_pos[0]
            
            # change target position to visual angle
            target_pos_deg = self.px_to_dva(target_pos[0], target_pos[1])
            
            # get displacement of the eyes from the target 
            gaze_trial = gaze[(gaze['timestamp [ns]'] >= event_start_time) & (gaze['timestamp [ns]'] <= event_end_time)]
            
            # get the median of the last 80 data points
            displacement = np.sqrt((gaze_trial['azimuth [deg]'].values[-80:] - target_pos_deg[0])**2 + (gaze_trial['elevation [deg]'].values[-80:] - target_pos_deg[1])**2) 
            
            # save the target position as a data frame 
            new_row = pd.DataFrame({
                        'trial': [e],
                        'target id': [target_ids[e-1]],
                        'target x [px]': [target_pos[0]],
                        'target y [px]': [target_pos[1]],
                        'target x [deg]': [target_pos_deg[0]],
                        'target y [deg]': [target_pos_deg[1]],
                        'gaze x [px]': [np.median(gaze_trial['gaze x [px]'].values[-80:])],
                        'gaze y [px]': [np.median(gaze_trial['gaze y [px]'].values[-80:])], 
                        'gaze x [deg]': [np.median(gaze_trial['azimuth [deg]'].values[-80:])],
                        'gaze y [deg]': [np.median(gaze_trial['elevation [deg]'].values[-80:])],
                        'distance [deg]': [np.median(displacement)]})
            df = pd.concat([df, new_row], ignore_index=True)
        
        df.to_csv(os.path.join(self.data_dir, 'gaze_vs_dot.csv'), index=False)
        
    
    ### TODO: NEED TO FILL OUT THIS SECTION
    
    def see_result(self):
        df = pd.read_csv(os.path.join(self.data_dir, 'gaze_vs_dot.csv'))
        
        # get the median target positions within the same target id 
        median_target = df.groupby('target id')[['target x [deg]', 'target y [deg]']].median()
        
        # in the first figure, plot the raw data with targets
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for t in range(len(median_target)):
            target = median_target.iloc[t]
            ax[0].scatter(target['target x [deg]'], target['target y [deg]'], marker='+', s=100)
        
        
        print(df)
        
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle unknown classes dynamically
        if module == "__main__" and name in ["Config", "Data"]:
            # Define a placeholder class for unknown classes
            class Placeholder:
                def __init__(self, *args, **kwargs):
                    pass
                
                def __repr__(self):
                    return f"<{name}Placeholder (original class not defined)>"
            
            return Placeholder
        
        # Fall back to the default behavior for known classes
        return super().find_class(module, name)

def main(meta):
    # load the data 
    config, response = meta.unpickle()
    
    # get the markers
    meta.screen_markers()
    
    # get the events
    events = pd.read_csv(os.path.join(meta.data_dir, 'events.csv'))
        
    # get the gaze data 
    gaze = meta.get_clean_gaze()
    
    # get the disposition of the eyes for each trial 
    meta.get_disposition(gaze, events)
    
    # print out the result
    meta.see_result()
    
        
if __name__ == "__main__":
    meta = MetaData()
    
    data_dir = 'jiwon_eyes/2024-12-1814-24-22-98722b4d'
    pkl_path = 'data/jiwon_241218_calib_3.pkl'
    
    meta.data_dir = data_dir
    meta.pickle_dir = None
    # parser = argparse.ArgumentParser(description="show calibration result")
    # parser.add_argument("--eye_path", required=True, help="Recording data directory")
    # parser.add_argument("--pkl_path", help="Pickle data directory (optional)")
    
    # args = parser.parse_args()
    # if not os.path.exists(args.eye_path):
    #     raise Exception('Download directory does not exist')
    # else:
    # meta.data_dir = args.eye_path

    # if args.pkl_path & os.path.exists(args.pkl_path):
    #     meta.pickle_dir = args.pkl_path
    # elif args.pkl_path & not os.path.exists(args.pkl_path):
    #     Exception('Pickle directory does not exist')

        
    main(meta)
    