""" 
    Run this code after running the calibration. If the light was not completely off during the calibration, 
    please run "neon_pick_markers.py" to pick the markers and then run this code.
    
    The code is dependent on the pupillabs_utils ### TODO: Need to migrate some of the functions to GazeFlow (?)
    
    Jiwon Yeon, 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, json, cv2, pickle, re, sys
# import argparse
from decord import VideoReader, cpu
from neon_pick_markers import click_markers, pick_markers
sys.path.append('../pupillabs_util')
from px_to_dva import px_to_dva

class MetaData:
    def __init__(self):
        self.data_dir = None
        self.pickle_dir = None
        
    def initialization(self):
        events = pd.read_csv(os.path.join(self.data_dir, 'events.csv'))
        self.nEvents = int(re.search(r'trial (\d+), end', events['name'].iloc[-2]).group(1))
        
        # from the events, sort the target id 
        targets_on_screen = events[events['name'].str.contains(', target location')]['name']
        trials, x, y = [], [], []
        for t in targets_on_screen:
            trials.append(int(re.search(r'trial (\d+)', t).group(1)))
            x.append(float(re.search(r'(-?\d+\.?\d*), (-?\d+\.?\d*)', t).group(1)))
            y.append(float(re.search(r'(-?\d+\.?\d*), (-?\d+\.?\d*)', t).group(2)))
        
        trials_df = pd.DataFrame({'trial': trials, 'x': x, 'y': y})
        target_pos = trials_df[['x', 'y']].drop_duplicates().sort_values(by=['x', 'y']).reset_index(drop=True)
        self.target_pos_screen = target_pos.values
        
        # map the target id
        target_id = np.zeros(len(trials_df))
        for t_id in range(len(target_pos)):
            matching = trials_df[(trials_df['x']==target_pos.iloc[t_id]['x']) & (trials_df['y']==target_pos.iloc[t_id]['y'])].index.values
            target_id[matching] = t_id+1
        trials_df['target id'] = target_id
        
        # save in self 
        self.target_pos = target_pos[['x', 'y']].values
        self.target_id = trials_df['target id'].values
        
        # get markers
        self.screen_markers()

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
    
    def dot_px_to_dva(self, x, y):
        camera_matrix = np.array(json.load(open(os.path.join(self.data_dir, 'scene_camera.json')))['camera_matrix'])
        distortion_coeffs = np.array(json.load(open(os.path.join(self.data_dir, 'scene_camera.json')))['distortion_coefficients'])
        dva = px_to_dva(x, y, camera_matrix, distortion_coeffs)
        
        return dva
    
    def frame_chop(self, frame, markers):
        min_x = int(np.min(markers[:, 0]))
        max_x = int(np.max(markers[:, 0]))
        min_y = int(np.min(markers[:, 1]))
        max_y = int(np.max(markers[:, 1]))
    
        frame_chopped = frame[min_y:max_y, min_x:max_x]
        return frame_chopped
        
    def dot_detector(self, frame):        
        if self.markers is not None:
            markers = self.markers
        else:
            markers = [[0,0], [np.shape(frame)[0], 0], [0, np.shape(frame)[1]], [np.shape(frame)[0], np.shape(frame)[1]]]     # if marker is not defined, use the size of the frame 
            
        # find center
        center = [int(np.max(markers[:,0])-np.min(markers[:,0]))/2,
                int(np.max(markers[:,1])-np.min(markers[:,1]))/2]
        
        # copy image    
        frame_c = frame.copy()
        frame_chopped = self.frame_chop(frame_c, markers)

        # filter image
        lower = np.array([200, 200, 200])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(frame_chopped, lower, upper)
        frame_filtered = cv2.bitwise_and(frame_chopped, frame_chopped, mask=mask)
        frame_gray = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
        frame_binary = cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY)[1]
        
        # detect blob
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 200
        
        detector = cv2.SimpleBlobDetector_create(params)
        dots = detector.detect(frame_binary)
        
        if len(dots) > 0:
            for d in range(len(dots)):
                # remove dots that are close to the center
                if np.linalg.norm(np.array(dots[d].pt) - np.array(center)) < 10:
                    dots.pop(d)
            
            if len(dots) > 0:   # if still the dots are more than one point, take the one farther from the rim
                dist = []
                for d in range(len(dots)):
                    dist.append(np.linalg.norm(np.array(dots[d].pt) - np.array(center)))
                dot = dots[np.argmax(dist)]
                dot = np.array([dot.pt[0] + np.min(markers[:,0]), dot.pt[1] + np.min(markers[:,1])])        
        else:
            dot = None
        
        return dot
    
    def get_dot_position(self):
        # get video, world_time, events
        video = VideoReader(glob.glob(self.data_dir + '/*.mp4')[0], ctx=cpu(0))
        world_times = pd.read_csv(os.path.join(self.data_dir, 'world_timestamps.csv'))
        events = pd.read_csv(os.path.join(self.data_dir, 'events.csv'))
        
        df = pd.DataFrame(columns=['trial', 'target id', 'frame', 'timestamp [ns]', 'dot x [px]', 'dot y [px]',
                           'azimuth [deg]', 'elevation [deg]'])
        
        for e in range(1,self.nEvents+1):
            print(f'Processing trial number {e}')
            start_time = events[events['name'].str.contains(f'trial {e}, target location')]['timestamp [ns]'].values[0]
            end_time = events[events['name'].str.contains(f'trial {e}, end')]['timestamp [ns]'].values[0]

            frames = world_times[(world_times['timestamp [ns]'] >= start_time) & (world_times['timestamp [ns]'] <= end_time)].index
            for f in range(len(frames)):
                frame = video[frames[f]].asnumpy()
                dots = self.dot_detector(frame)
                
                if dots is not None:
                    dva = self.dot_px_to_dva(dots[0], dots[1])
                    new_row = pd.DataFrame({
                    'trial': e,
                    'target id': self.target_ids[e-1],
                    'frame': frames[f],
                    'timestamp [ns]': world_times['timestamp [ns]'][frames[f]],
                    'dot x [px]': dots[0] if dots is not None else None,
                    'dot y [px]': dots[1] if dots is not None else None,
                    'azimuth [deg]': dva[0],
                    'elevation [deg]': dva[1]}, index=[0])
                
                    # add new row to df
                    df = pd.concat([df, new_row], ignore_index=True)

        # save to csv file 
        df.to_csv(data_dir + '/dots.csv', index=False)
        
    def find_target_positions(self):
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        dots = pd.read_csv(os.path.join(self.data_dir, 'dots.csv'))
        data = dots[['azimuth [deg]', 'elevation [deg]']].values

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        dbscan = DBSCAN(eps=0.1, min_samples=30)
        clusters = dbscan.fit_predict(data_scaled)

        # Get cluster centers (true positive positions)
        unique_clusters = np.unique(clusters[clusters != -1])  # Exclude noise points (-1)

        # make a data frame using the unique clusters 
        cluster_df = pd.DataFrame(columns = ['cluster id', 'x [deg]', 'y [deg]', 'x [px]', 'y [px]'])
        for c in unique_clusters:
            row = [{'cluster id': c, 
                    'x [deg]': data[clusters == c][:, 0].mean(),
                    'y [deg]': data[clusters == c][:, 1].mean(),
                    'x [px]': dots['dot x [px]'][clusters == c].mean(),
                    'y [px]': dots['dot y [px]'][clusters == c].mean()}]
            # append row to cluster_df
            cluster_df = pd.concat([cluster_df, pd.DataFrame(row)], ignore_index=True)

        # from true_positive_positions, remove the closest to the center, and remain 16 points that are closest to the center
        center = cluster_df[['x [deg]', 'y [deg]']].mean().values
        dist = [np.linalg.norm(pos - center) for pos in cluster_df[['x [deg]', 'y [deg]']].values]
        cluster_df['dist_center'] = dist
        cluster_df = cluster_df.drop(cluster_df['dist_center'].idxmax())
        cluster_df = cluster_df.drop(cluster_df['dist_center'].idxmin())
        
        # Plot results
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', s=10)
        plt.scatter(
            [pos for pos in cluster_df['x [deg]'].values], 
            [pos for pos in cluster_df['y [deg]'].values], 
            color='red', label='Cluster Centers', marker='x', s=100
        )
        plt.title(f"DBSCAN Clustering Results")
        plt.legend()
        plt.savefig(os.path.join(self.data_dir, 'target_dots_dbscan_clusters.png'))
        plt.close()
        
        ### TODO: how to save the target positions with the target id?
        ## save the cluster result that matches to the dots 
        ## from there, get the mean cluster positions in pixels 
        ## from dots, save the information of the target id
        ## rather than save this information, return the information to the main function
        
    
    
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
    # initialize meta
    meta.initialization()
        
    # get the dot position
    meta.get_dot_position()
    
    # from the dot positions, define target positions
    target_pos = meta.find_target_positions()
    
    # get the gaze data 
    gaze = meta.get_clean_gaze()
    
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
    