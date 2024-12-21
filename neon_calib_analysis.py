""" 
    Run this code after running the calibration. If the light was not completely off during the calibration, 
    please run "neon_pick_markers.py" to pick the markers and then run this code.
    
    The code is dependent on the pupillabs_utils ### TODO: Need to migrate some of the functions to GazeFlow (?)
    
    Jiwon Yeon, 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, json, cv2, re, sys
# import argparse
from decord import VideoReader, cpu
from neon_pick_markers import click_markers, pick_markers
sys.path.append('../pupillabs_util')
from px_to_dva import px_to_dva

class MetaData:
    def __init__(self):
        self.data_dir = None
        
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
        target_id = np.zeros(len(trials_df), dtype='int')
        for t_id in range(len(target_pos)):
            matching = trials_df[(trials_df['x']==target_pos.iloc[t_id]['x']) & (trials_df['y']==target_pos.iloc[t_id]['y'])].index.values
            target_id[matching] = t_id+1
        trials_df['target id'] = target_id
        self.target_id = trials_df['target id'].values
        
        # get markers
        self.screen_markers()
    
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
        
        df = pd.DataFrame(columns=['trial', 'target id', 'frame', 'timestamp [ns]', 'x [px]', 'y [px]',
                           'x [deg]', 'y [deg]', 'x screen pos', 'y screen pos'])
        
        for e in range(1,self.nEvents+1):
            print(f'Processing trial number {e}')
            start_time = events[events['name'].str.contains(f'trial {e}, target location')]['timestamp [ns]'].values[0]
            end_time = events[events['name'].str.contains(f'trial {e}, end')]['timestamp [ns]'].values[0]
            target_id = self.target_id[e-1]
            target_pos = self.target_pos_screen[target_id-1]

            frames = world_times[(world_times['timestamp [ns]'] >= start_time) & (world_times['timestamp [ns]'] <= end_time)].index
            for f in range(len(frames)):
                frame = video[frames[f]].asnumpy()
                dots = None
                dots = self.dot_detector(frame)
                
                if dots is not None:
                    dva = self.dot_px_to_dva(dots[0], dots[1])
                    new_row = pd.DataFrame({
                    'trial': e,
                    'target id': target_id,
                    'frame': frames[f],
                    'timestamp [ns]': world_times['timestamp [ns]'][frames[f]],
                    'x [px]': dots[0],
                    'y [px]': dots[1],
                    'x [deg]': dva[0],
                    'y [deg]': dva[1], 
                    'x screen pos': target_pos[0], 
                    'y screen pos': target_pos[1]}, index=[0])
                
                    # add new row to df
                    df = pd.concat([df, new_row], ignore_index=True)

        # save to csv file 
        df.to_csv(self.data_dir + '/dots.csv', index=False)
        
    def find_target_positions(self):
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        dots_csv = os.path.join(self.data_dir, 'dots.csv')
        if not os.path.exists(dots_csv):
            raise Exception('dots.csv does not exist. Run get_dot_position() first.')
        
        dots = pd.read_csv(dots_csv)
        data = dots[['x [deg]', 'y [deg]']].values

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
                    'x [px]': dots['x [px]'][clusters == c].mean(),
                    'y [px]': dots['y [px]'][clusters == c].mean(), 
                    'x screen pos': dots['x screen pos'][clusters == c].median(),
                    'y screen pos': dots['y screen pos'][clusters == c].median()
                    }]

            # append row to cluster_df
            cluster_df = pd.concat([cluster_df, pd.DataFrame(row)], ignore_index=True)

        # from true_positive_positions, remove the closest to the center, and remain 16 points that are closest to the center
        center = cluster_df[['x [deg]', 'y [deg]']].mean().values
        dist = [np.linalg.norm(pos - center) for pos in cluster_df[['x [deg]', 'y [deg]']].values]
        cluster_df['dist_center'] = dist
        cluster_df = cluster_df.drop(cluster_df['dist_center'].idxmax())
        cluster_df = cluster_df.drop(cluster_df['dist_center'].idxmin())

        # sort the cluster_df by the target screen position and assign the target id
        cluster_df = cluster_df.sort_values(by=['x screen pos', 'y screen pos'])
        cluster_df['target id'] = np.arange(1, len(cluster_df)+1)

        # Plot results
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', s=10)
        plt.scatter(cluster_df['x [deg]'], cluster_df['y [deg]'], 
            color='red', label='Cluster Centers', marker='x', s=100
        )
        plt.title(f"DBSCAN Clustering Results")
        plt.legend()
        plt.savefig(os.path.join(self.data_dir, 'target_dots_dbscan_clusters.png'))
        # plt.show()
        plt.close()
        
        dots_cluster = pd.concat([dots, pd.DataFrame(clusters, columns=['cluster id'])], axis=1)
        
        return dots_cluster, cluster_df
    
    def do_affine(self, target_fix_df):
        from skimage import transform as tf

        # draw the target and fixation positions 
        fig, ax = plt.subplots(1,2, figsize=(12, 6))

        # get the average offset
        eye_offset = np.sqrt((target_fix_df['target detected x [deg]'].values - target_fix_df['gaze pos x [deg]'].values)**2 + \
            (target_fix_df['target detected y [deg]'].values - target_fix_df['gaze pos y [deg]'].values)**2)

        
        # gaze after affine transformation
        transform_matrix = tf.AffineTransform(matrix=None)
        transform_matrix.estimate(src = target_fix_df[['gaze pos x [deg]', 'gaze pos y [deg]']].to_numpy(),
                                dst = target_fix_df[['target detected x [deg]', 'target detected y [deg]']].to_numpy())
        transformed_positions = transform_matrix(target_fix_df[['gaze pos x [deg]', 'gaze pos y [deg]']])

        affine_offset = np.sqrt((transformed_positions[:,0] - target_fix_df['target detected x [deg]'].values)**2 + \
            (transformed_positions[:,1] - target_fix_df['target detected y [deg]'].values)**2)

        # draw figure
        cluster_mean = target_fix_df[['target cluster x [deg]', 'target cluster y [deg]']].drop_duplicates().values
        ax[0].scatter(target_fix_df['gaze pos x [deg]'], target_fix_df['gaze pos y [deg]'], color='black', label='Gaze Fixation', marker='o', s=50)
        ax[0].scatter(cluster_mean[:,0], cluster_mean[:,1], color='red', label='Cluster Centers', marker='x', s=80)
        ax[0].set_title(f'Before transformation, average offset: {np.nanmean(eye_offset):.2f} deg')

        ax[1].scatter(transformed_positions[:,0], transformed_positions[:,1], color='black', label='Gaze Fixation', marker='o', s=50)
        ax[1].scatter(cluster_mean[:,0], cluster_mean[:,1], color='red', label='Cluster Centers', marker='x', s=80)
        ax[1].set_title(f'After transformation, average offset: {np.nanmean(affine_offset):.2f} deg')
        
        fig.savefig(os.path.join(self.data_dir, 'gaze_vs_target.png'))
        plt.close()
        
        # save the transformation matrix
        np.save(os.path.join(self.data_dir, 'affine_matrix.npy'), transform_matrix.params)
        
    
    def gaze_vs_target(self, dots_cluster, cluster_df):
        gaze = self.get_clean_gaze()

        target_fix_df = pd.DataFrame(columns=['trial', 'target id', 'target cluster x [deg]', 'target cluster y [deg]',
                'target pos screen x [px]', 'target pos screen y [px]', 'target detected x [deg]', 'target detected y [deg]', 
                'gaze pos x [deg]','gaze pos y [deg]'])

        for t in range(1, self.nEvents+1):
            this_trial = dots_cluster[dots_cluster['trial'] == t]
            target_id = this_trial['target id'].unique()[0]

            # find the average target position from the cluster_df
            target_pos = cluster_df[cluster_df['target id'] == target_id][['x [deg]', 'y [deg]']].values[0]
            distance = np.sqrt((this_trial['x [deg]'] - target_pos[0])**2 + (this_trial['y [deg]'] - target_pos[1])**2)
            mask = np.abs(distance) < 1
            # if there are no points within 1 degree, take the 10 closest points
            if len(this_trial[mask]) == 0:
                mask = distance.sort_values()
                mask = mask < mask.iloc[10]
                mask = mask.sort_index()

            # get average target position of the trial
            trial_target_pos = this_trial[mask][['x [deg]', 'y [deg]']].mean(skipna=True).values

            # find the gaze positions 
            this_gaze = gaze[(gaze['timestamp [ns]'] >= this_trial['timestamp [ns]'].min()) &
                            (gaze['timestamp [ns]'] <= this_trial['timestamp [ns]'].max())]

            # select the last 500 ms window
            time = ((this_gaze['timestamp [ns]'] - this_gaze['timestamp [ns]'].values[0])/1e9)
            index = time[time >= time.max()-0.5].index
            
            gaze_at_target = this_gaze.loc[index][['azimuth [deg]', 'elevation [deg]']].values
            
            # add to a data frame
            new_row = pd.DataFrame({
                'trial': t, 
                'target id': target_id, 
                'target cluster x [deg]': target_pos[0], 
                'target cluster y [deg]': target_pos[1],
                'target pos screen x [px]': cluster_df[cluster_df['target id'] == target_id]['x screen pos'].values[0],
                'target pos screen y [px]': cluster_df[cluster_df['target id'] == target_id]['y screen pos'].values[0],
                'target detected x [deg]': trial_target_pos[0],
                'target detected y [deg]': trial_target_pos[1],
                'gaze pos x [deg]': np.median(gaze_at_target[:,0]),
                'gaze pos y [deg]': np.median(gaze_at_target[:,1])
            }, index=[0])
            
            target_fix_df = pd.concat([target_fix_df, new_row], ignore_index=True)
            
        # generate affine transformation result 
        self.do_affine(target_fix_df)
        
        

def main(meta):
    # initialize meta
    meta.initialization()
        
    # get the dot position
    meta.get_dot_position()
    
    # from the dot positions, define target positions
    dots_cluster, cluster_df = meta.find_target_positions()
    
    # get the displacement between the gaze and the target
    meta.gaze_vs_target(dots_cluster, cluster_df)
    
        
if __name__ == "__main__":
    meta = MetaData()
    
    data_dir = 'jiwon_eyes/2024-12-1814-24-22-98722b4d'
    meta.data_dir = data_dir
        
    main(meta)
    