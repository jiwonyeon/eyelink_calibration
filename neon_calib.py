""" 
    This script is used to calibrate the eye tracker using Neon.
    To get the best result, either use the April tags, manually pick the monitor's four corner using 
    "neon_pick_markers.py" or turn the lights off when running the task. 
    
    Check configurations in the Config class, especially the IP address or the number of repetition.
    
    Jiwon Yeon, 2024
"""

import numpy as np
import os, glob, pickle, json
import asyncio
from pupil_labs.realtime_api import Device, receive_gaze_data
from psychopy import visual, core, event, monitors
from psychopy.hardware import keyboard
from datetime import datetime
import numpy as np

kb = keyboard.Keyboard()

class Config:
    def __init__(self):
        self.subject_id = "jiwon_241223"
        self.ip = "10.119.105.208"
        self.port = "8080" 
        self.use_device = False
        self.record = True
        self.april_tags = False
        self.repeat = 4
        self.display = 'WINDOW' # 'FULLSCREEN' or 'WINDOW'
        self.target_radius = .6     # in degree
        self.cursor_radius = .3     # in degree
        self.fixation_size = .3
        self.monitor_number = 0        
        self.monitor_resolution = [1920, 1080]
        self.monitor_width_cm = 52
        self.viewing_distance_cm = 34   # in cm
        self.background = [-1, -1, -1]
        self.units = 'deg'
        self.data_dir = 'data'
        utc = core.getAbsTime()
        self.time_UTC = utc
        self.time_readable = datetime.utcfromtimestamp(utc).strftime('%Y-%m-%d-%H-%M-%S')
        self.saving_name()
        
    def saving_name(self):
        pickle_list = glob.glob(os.path.join(self.data_dir, f'{self.subject_id}_calib_*.pkl'))
        if len(pickle_list) == 0:
            self.file_name = f'{self.subject_id}_calib_1'
        else:
            self.file_name = f'{self.subject_id}_calib_{len(pickle_list)+1}'
            
    def to_dict(self):
        return{'subject_id': self.subject_id,
               'device':  self.use_device,
                'monitor resolution': self.monitor_resolution,
                'monitor width cm': self.monitor_width_cm,
                'viewing distance cm' : self.viewing_distance_cm,
                'units': self.units, 
                'data dir': os.path.abspath(self.data_dir),
                'experiment time': self.time_readable,
                'saving name': self.file_name}
        
def square_points(len):
    half_len = len / 2
    x = [-half_len, 0, half_len]
    y = [-half_len, 0, half_len]

    X, Y = np.meshgrid(x, y)
    coords = np.array([X.ravel(), Y.ravel()]).T
    coords = coords[~np.all(coords == 0, axis=1)]
    
    return coords

def draw_aprilTags(screen):
    # four corners of the screen
    april_tags_dir = os.path.join('april_tags')
    pos = 0.8
    positions = [[-pos, pos], [pos, pos], [pos, -pos], [-pos, -pos]]  # Move positions inward
    for i in range(4):
        image_path = os.path.join(april_tags_dir, f"April0{i+1}.png")
        
        # Load the image to get its dimensions
        image = visual.ImageStim(screen, image=image_path, units='norm')
        image_width, image_height = image.size
        
        # Calculate the aspect ratio
        aspect_ratio = image_width / image_height
        
        # Set the size while maintaining the aspect ratio
        desired_height = 0.15  # Set the desired height
        desired_width = desired_height * aspect_ratio
        
        # Create the ImageStim with the correct size
        image = visual.ImageStim(screen, image=image_path, pos=positions[i], 
                                 units='norm', size=(desired_width, desired_height))
        image.draw()
        

class Data:
    def __init__(self):
        self.eye_pos = []
        self.cursor_pos = []
        self.target_pos = None
        self.rt = []
        self.trial_passed = None
        self.timestamps = None
        
    def to_dict(self):
        return {'eye_pos': self.eye_pos,
                'cursor_pos': self.cursor_pos,
                'target_pos': self.target_pos.tolist(),
                'target_number': self.target_number.tolist(),
                'rt': self.rt,
                'timestamps': self.timestamps,
                'trials_passed': self.trial_passed}
    
class Experiment:
    def __init__(self, config, device, screen):
        self.config = config
        self.device = device
        self.screen = screen
        self.data = Data()
        
    async def initialize(self):        
        self.threshold = 0.1    # offset between the mouse click and the target
        self.stim_locations_deg = np.vstack([square_points(20/2), square_points(20)])
        target_number = np.arange(0, len(self.stim_locations_deg))
        self.data.target_number = np.random.permutation(np.tile(target_number, self.config.repeat))
        self.data.target_pos = self.stim_locations_deg[self.data.target_number, :]
        self.n_targets = np.shape(self.data.target_pos)[0]
        print(f"Number of trials: {self.n_targets}")
        if self.device is not None:
            status = await self.device.get_status()
            self.sensor_gaze = status.direct_gaze_sensor()
            
    async def check_escape_key(self):
        keys = kb.getKeys(keyList=['escape'], waitRelease=False)
        if keys:
            print("Experiment stopped by user")
            await self.end_experiment()
            core.quit()  # Forcefully exit PsychoPy
            
    async def start_experiment(self):
        # show the screen
        mouse = event.Mouse(visible=False)
        fixation_size = self.config.fixation_size
        fixation = visual.ShapeStim(self.screen, vertices=((0, -fixation_size), 
                                                           (0, fixation_size), 
                                                           (0,0), 
                                                           (-fixation_size,0), 
                                                           (fixation_size, 0)),
            lineWidth=2, closeShape=False, lineColor=[1,1,1])        
        msg = visual.TextStim(self.screen, pos=[0,-2], text='The experiment will start soon...', height=1)
        msg.draw()
        fixation.draw()
        if self.config.april_tags:
            draw_aprilTags(self.screen)
        self.screen.flip()
        
        # start recording if needed
        if self.config.record and self.device is not None:
            await self.device.recording_start()
            core.wait(2)
        
        # wait for the return key to be pressed        
        key_pressed = None
        while not key_pressed:
            keys = kb.getKeys(keyList=['return'], waitRelease=False)
            if keys:
                key_pressed = keys[0]
                print('Experiment start!')
        
           
    async def start_loop(self):
        fixation_size = self.config.fixation_size
        fixation = visual.ShapeStim(self.screen, vertices=((0, -fixation_size), (0, fixation_size), (0,0),(-fixation_size,0), (fixation_size, 0)),
                                    lineWidth=2, closeShape=False, lineColor=[1,1,1])
        mouse = event.Mouse(visible=False)        
        cursor = visual.Circle(self.screen, units='deg', radius=self.config.cursor_radius, pos=(0,0), fillColor='red', lineColor='red', opacity=0.8)
        restart_on_disconnect = True

        for trial_num in range(self.n_targets):
            # check escape key
            # await self.check_escape_key()
            
            # initialize everything
            buff_eyes = []
            buff_cursor = []
            rt = None
            timestamp = []
            target_pos = self.data.target_pos[trial_num,:]
            flag = True
            first_frame = True

            # put the mouse at the center
            mouse.setPos((0,0))
            cursor.pos = (0,0)
            cursor.draw()
            
            # show the fixation 
            fixation.draw()
            
            # flip the screen - fixation presentation 
            if self.config.april_tags:
                draw_aprilTags(self.screen)
            self.screen.flip()
            timestamp.append(core.getAbsTime())
            
            # send the event 
            if self.config.record and self.device is not None:
                await self.device.send_event(f"trial {trial_num+1}, fixation") 
            
            # wait for 200ms
            core.wait(0.2)
            
            # prepare for the trial
            target = visual.GratingStim(self.screen, color=1, colorSpace='rgb',tex=None, mask='circle', 
                                        size=self.config.target_radius, pos=target_pos)

            while flag:                
                try:
                    # Check for escape key
                    # await self.check_escape_key()
                    
                    # if it is the first frame, set the mouse position to the center
                    if first_frame:
                        mouse.setPos((0,0))
                    
                    # draw cursor, target, and move the mouse pos                                
                    target.draw()
                    cursor.pos = mouse.getPos()                                              
                    cursor.draw()
                    if self.config.april_tags:
                        draw_aprilTags(self.screen)
                    
                    # flip the screen. Save the timestamp if it is the first frame
                    self.screen.flip()
                    if first_frame:
                        timestamp.append(core.getAbsTime())
                        if self.config.record and self.device is not None:
                           await self.device.send_event(f"trial {trial_num+1}, target location: ({target_pos[0], target_pos[1]})")            

                        first_frame = False
                    
                    # get the gaze data if possible    
                    if self.device is not None:
                        async for gaze in receive_gaze_data(self.sensor_gaze.url, run_loop=restart_on_disconnect):
                            # save eye and cursor position
                            buff_eyes.append([float(gaze.x), float(gaze.y)])

                    # save the cursor position
                    if self.config.record:
                        buff_cursor.append(list(cursor.pos))                            
        
                    # Check for mouse click
                    if mouse.getPressed()[0]:  # Left mouse button
                        print(f"Mouse clicked at: {cursor.pos}")
                        distance = np.linalg.norm(target_pos - cursor.pos)
                        
                        # save the cursor position
                        if self.config.record:
                            buff_cursor.append(list(cursor.pos))
                            
                        # check if the click is valid
                        if distance <= self.threshold:                            
                            trial_end = core.getAbsTime()
                            timestamp.append(trial_end)
                            rt = trial_end - timestamp[0]
                            print(f"Valid click! Target position: {target_pos}, Mouse position: {cursor.pos}")
                            print(f'Time took: {rt:.4}')
                            flag = False
                                    
                            # send the event 
                            if self.config.record and self.device is not None:
                                await self.device.send_event(f"trial {trial_num+1}, end")       
                            else:
                                buff_eyes = None                 
                                    
                            break
                        else:
                            print(f"Invalid click. Mouse {cursor.pos} and target {target_pos}. Try again.")
                  
                except Exception as e:
                    print(f"Error: {e}")
                    await self.end_experiment()
                    break
            
            # save the interim data
            self.data.eye_pos.append(buff_eyes)
            self.data.cursor_pos.append(buff_cursor)
            self.data.rt.append(rt)
            self.data.timestamps = timestamp
            self.data.trial_passed = trial_num + 1
            
            data_to_save = {'config': self.config.to_dict(), 'data': self.data.to_dict()}
            with open(os.path.join(self.config.data_dir, f'{self.config.file_name}_temp.json'), 'w') as file:
                json.dump(data_to_save, file)
            
            
    async def end_experiment(self):
        # thank you note
        msg = visual.TextStim(self.screen, text='The experiment ended. \nThank you.')
        msg.draw()
        self.screen.flip()
        
        # save the data if it was recorded
        if self.config.record and self.device is not None:
            await self.device.recording_stop_and_save()
        
        # save the experiment 
        data_to_save = {'config': self.config.to_dict(), 'data': self.data.to_dict()}
        with open(os.path.join(self.config.data_dir, f'{self.config.file_name}.json'), 'w') as file:
            json.dump(data_to_save, file)
            
        # remove interim data
        if os.path.exists(os.path.join(self.config.data_dir, f'{self.config.file_name}_temp.json')):
            os.remove(os.path.join(self.config.data_dir, f'{self.config.file_name}_temp.json'))
        
        # shut the window
        core.wait(3)
        self.screen.close()
        core.quit()
        

async def main(config):
    device = None
    if config.use_device:
        device = Device(address=config.ip, port=config.port)
        print(f"Attempting manual connection with IP: {config.ip} and port: {config.port}")
            
        # get the device status
        try:
            status = await device.get_status()
            print(f"Manual connection successful.")
        except Exception as e:
            print(f"Manual connection failed: {e}")
            return
    
    # initiate psychopy
    monitor = monitors.Monitor("default", width=config.monitor_width_cm, distance=config.viewing_distance_cm)
    monitor.setSizePix(config.monitor_resolution)
    if config.display == 'FULLSCREEN':
        screen = visual.Window(size=config.monitor_resolution, fullscr=True,
                               monitor=monitor,
                               color=config.background, units=config.units)
    elif config.display == 'WINDOW':
        screen = visual.Window(size=[1000,800], fullscr=False, 
                               monitor=monitor,
                               color=config.background, units=config.units)
    
    # initiate the experiment 
    experiment = Experiment(config, device, screen)
    await experiment.initialize()
    
    # start experiment
    await experiment.start_experiment()
    
    # start the loop
    await experiment.start_loop()
    
    # end the experiment and save the data
    await experiment.end_experiment()
    

if __name__ == "__main__":
    asyncio.run(main(Config()))