import time
import random
import math
import cv2
import yaml
import numpy as np
import gym
from gym import spaces
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from Pose import Pose, compare_poses, difference
from utils import mod



class MovableObject:
    """ 
    ATTRIBUTES:
        FIXED:
        - env (NAMOENV2D): environment in which the MovableObject has been created
        - Pose (Pose): Pose object that represents the position and orientation of the center of the MovableObject
        - param_dict (dict): parameters of the MovableObject (stored in a .yaml file in the "objects" folder)
        - pick_poses (list of Poses): list of poses that indicates the position and orientation of the "regions of pickability"
        - imageshape (np.float32): length,width of the MovableObject's image
        - image (np.uint8): image representing the MovableObject
    """
    def __init__(self, env, init_dict, name):
        self.env = env
        self.pose=Pose(init_dict['pose'], self.env.ORIGIN, name)
        path="objects/"+init_dict['type']+".yaml"
        with open(path) as f:
            self.param_dict=yaml.load(f, Loader=yaml.SafeLoader)
        self.pick_poses = []
        for pick_pose in self.param_dict['pick_poses']:
            self.pick_poses.append(Pose(pick_pose,self.pose))
        self.imageshape = np.array(self.param_dict['shape'],dtype=np.float32) * self.env.REAL_RESOLUTION
        self.image = cv2.resize(np.array(self.param_dict['image'], dtype=np.uint8), tuple(self.imageshape.astype(int)), interpolation=cv2.INTER_NEAREST)
        self.rollback_position = self.pose.position
        self.rollback_orientation = self.pose.orientation

    def __del__(self):
        self.pose.parent.children.remove(self.pose)
    
    def __repr__(self) -> str:
        return str(self.pose)

    def move(self, linear_displacement, angular_displacement):
        self.rollback_position = self.pose.position.copy()
        self.rollback_orientation = self.pose.orientation
        self.pose.move(linear_displacement,angular_displacement)
    
    def rollback(self):
        self.pose.position = self.rollback_position
        self.pose.orientation = self.rollback_orientation
    
    def get_mask(self):
        """mask is used to detect collisions
        
        OUTPUT
            - mask (np.uint8)
        """
        # create mask with shape of global map
        mask = np.zeros(self.env.FO_map.shape, dtype=np.uint8)
        # get the MovableObkect's pose in the mask array coordinates
        row, col, orientation = self.pose.get_image_pose(self.env.REAL_RESOLUTION,mask.shape[0])
        # rotate the image to show object orientation
        rotatedimage = ndimage.rotate(self.image.copy(), orientation, reshape=True)
        # compute the slicing indices
        LenRow, LenCol = rotatedimage.shape
        start_row = row - LenRow // 2
        end_row = row + (LenRow - LenRow // 2)
        start_col = col - LenCol // 2
        end_col = col + (LenCol - LenCol // 2)
        # Calculate the intersection between the rotated image and the map array
        mask_start_row = max(start_row, 0)
        mask_end_row = min(end_row, mask.shape[0])
        mask_start_col = max(start_col, 0)
        mask_end_col = min(end_col, mask.shape[1])
        img_start_row = max(0, -start_row)
        img_end_row = img_start_row + (mask_end_row - mask_start_row)
        img_start_col = max(0, -start_col)
        img_end_col = img_start_col + (mask_end_col - mask_start_col)
        # Check if the indices are within the bounds of the map
        if mask_start_row < mask_end_row and mask_start_col < mask_end_col:
            mask[mask_start_row:mask_end_row, mask_start_col:mask_end_col] += rotatedimage[img_start_row:img_end_row, img_start_col:img_end_col]
        return mask

class NAMOENV2D:
    """ custom environment for namo task.
    ATTRIBUTES:
        FIXED:
        - FO_map (np.uint8): fixed obstacle map, read from the map.pgm file in the map's subfolder. (0 -> free space ; 1 -> fixed obstacle)
        - SC_map (np.float32): Social cost map, read from the SCmap.npy in the map's subfolder.
        - config_dict (dict): dictionary descripting the initial configuration of the environment, read from the config_name.yaml in the map's subfolder.
        - REAL_RESOLUTION (float): resolution of the obstacle map (in pxl/m).
        - ORIGIN (pose): pose of the origin of the real coordinates on the map.
        - start (pose)
        - goal (pose)
    
        MODIFIABLE:
        - steps (int): current step number
        - ROB (MovableObject): robot for this episode.
        - MO_list (list of MovableObjects): MO list for this episode.
        - path_map (np.uint8): 1 in path, 0 outside of path
        - collision_map (np.uint8)
        - pickedMO (MovableObject): if None -> no picked MO at this step
                                    else -> this step's picked MO in the self.MO_list of movable obstacles.
    """


    def __init__(self,map_name="200x200_empty",config_name="config1",seed=10):
        """set several attributes that will be fixed for this environment object:
            INPUT:
                - map_name (str)
                - config_name (str)
                - seed (int)
        """
        t=time.time()
        random.seed(seed)
        # MAP parameters
        map_path="maps/"+map_name+"/map.pgm"
        SCmap_path="maps/"+map_name+"/SCmap.npy"
        self.FO_map =np.array( 1 - cv2.imread(map_path, 0)//230, dtype=np.uint8)
        self.SC_map = np.load(SCmap_path)
        # CONFIG parameters
        config_path="maps/"+map_name+"/"+config_name+".yaml"
        with open(config_path) as f:
            self.config_dict=yaml.load(f, Loader=yaml.SafeLoader)
        self.REAL_RESOLUTION=int(self.config_dict["real_resolution"])
        self.ORIGIN=Pose(self.config_dict["origin_pose"],name="ORIGIN")
        self.goal = Pose(self.config_dict['goal_pose'],self.ORIGIN,"goal",color=(0,0,255))
        self.start = Pose(self.config_dict['ROB_init_dict']['pose'],self.ORIGIN,"start",color=(0,0,255))
        self.ROB = MovableObject(self,self.config_dict['ROB_init_dict'],'ROB')
        self.MO_list=[]
        i=0
        for MO_init_dict in self.config_dict['MO_init_dicts_list']:
            self.MO_list.append(MovableObject(self,MO_init_dict,f"{i}"))
            i+=1
        self.pickedMO = None
        self.steps = 1
        self._check_collision()
        self.ping=time.time()-t
        if __name__ == "__main__":
            print(f"goal: {self.goal.get_local_pose()}, goal_image {self.goal.get_image_pose(self.REAL_RESOLUTION,self.FO_map.shape[0])}")
            print(f"start: {self.start.get_local_pose()}, start_image {self.start.get_image_pose(self.REAL_RESOLUTION,self.FO_map.shape[0])}")

    def add_obstacle(self, type=None, pose=None):
        t=time.time()
        if type is None:
            type = random.choice(["MOa", "MOb", "MOc"])
        if pose is None:
            max_tries = 1000
            for tries in range(max_tries):
                orig_x,orig_y = self.ORIGIN.position
                width = self.FO_map.shape[0] // self.REAL_RESOLUTION
                height= self.FO_map.shape[1] // self.REAL_RESOLUTION
                rand_x = random.uniform(-orig_x,width-orig_x)
                rand_y = random.uniform(-orig_y,height-orig_y)
                rand_orientation = random.uniform(-180,180)
                self.MO_list.append(MovableObject(self,{'type': type, 'pose': [rand_x,rand_y,rand_orientation]}, f"{len(self.MO_list)}" ))
                if self._check_collision():
                    self.MO_list.pop()
                else:
                    break
        self.ping=time.time()-t
    
    def step(self, linear_displacement, angular_displacement, interaction=False):
        """ the environment takes a timestep 
            the robot executes the inputed actions.

        INPUT:
            linear_displacement (float)
            angular_displacement (float)
        """
        t=time.time()
        self.steps += 1
        self.ROB.move(linear_displacement,angular_displacement)
        if self._check_collision():
            print("COLLISION")
            self.ROB.rollback()
            self._check_collision()
        if interaction:
            if self.pickedMO is None:
                done=False
                for MO in self.MO_list:
                    for MOpick in MO.pick_poses:
                        for ROBpick in self.ROB.pick_poses:
                            if compare_poses(ROBpick,MOpick,0.1,15):
                                MO.pose.to_brother_frame(self.ROB.pose)
                                # MO.pose.position = ROBpick.position - MOpick.position
                                # MO.pose.orientation = mod(ROBpick.orientation - MOpick.orientation)
                                self.pickedMO = MO
                                done=True
                                break
                        if done:
                            break
                    if done:
                        break
            else:
                self.pickedMO.pose.to_parent_frame()
                self.pickedMO = None
        self.ping=time.time()-t


    def _check_collision(self):
        """
        OUTPUT:
            - collision (bool): True when there is a collision
        """
        # build a map with 1 on fixed obstacles (FO) and every object (MO/ROB)
        self.collision_map = self.FO_map.copy()
        for object in self.MO_list+[self.ROB]:
            self.collision_map+=object.get_mask()
        # if objects overlap with each other or fixed obstacles, 1+1=2. 
        # if there is more pixels with 2 than 1/20 of the robot surface in pixels, we consider there is a collision.
        if (self.collision_map > 1).sum() > np.prod(self.ROB.imageshape)/50:
            return True
        else:
            return False
    
    def render(self, render_pose=True):
        t = time.time()
        image = np.where(self.collision_map == 1, 0, 255).astype(np.uint8)
        if render_pose:
            res_factor = 4
            image = cv2.resize(image, (image.shape[1]*res_factor,image.shape[0]*res_factor), interpolation=cv2.INTER_NEAREST)
            # Convert to a three-channel image using cvtColor
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.ORIGIN.draw_descendants(image,self.REAL_RESOLUTION*res_factor, draw_position=False)
        cv2.namedWindow("render", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("render", 400, 400)
        cv2.moveWindow("render", 0, 0)
        cv2.imshow("render", image)
        cv2.waitKey(1)
        self.render_ping = time.time()-t
    
    def info_display(self):
        display_text =  (f"steps:{self.steps} | ping:{self.ping:.4f} | render_ping:{self.render_ping:.4f}")  
        if self.pickedMO is not None:
            display_text += f"\npickedMO: {self.pickedMO.pose.name}"
        if __name__ == "__main__":
            display_text += f"\n -> z,q,s,d : move    -> e : pickup/putdown \n -> r : reset           -> t : add obstacle"
        # Create an image with text
        text_image = np.zeros([150,400])
        font_scale = 0.5
        color = 255
        thickness = 1
        x_position = 10
        y_position = 20
        line_height = 30
        # Split text into lines
        for i, line in enumerate(display_text.split('\n')):
            y = y_position + i * line_height
            cv2.putText(text_image, line, (x_position, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        window_name = "info"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 400, 240)
        cv2.moveWindow(window_name, 0, 430)
        cv2.imshow(window_name, text_image)


if __name__ == "__main__":
    from pynput import keyboard

    env_list=["200x200_empty","200x200_map1","200x200_map2"]
    i = 0
    env = NAMOENV2D(env_list[i%3])
    env.render()
    env.info_display()
    pressed_keys = set()

    def on_press(key):
        try:
            if key.char == 'e':
                pressed_keys.add('e')
            elif key.char == 'z':
                pressed_keys.add('z')
            elif key.char == 's':
                pressed_keys.add('s')
            elif key.char == 'q':
                pressed_keys.add('q')
            elif key.char == 'd':
                pressed_keys.add('d')
            elif key.char == 'r':
                pressed_keys.add('r')
            elif key.char == 't':
                pressed_keys.add('t')
        except AttributeError:
            if key == keyboard.Key.esc:
                pressed_keys.add('esc')

    def on_release(key):
        try:
            if key.char == 'z':
                pressed_keys.discard('z')
            elif key.char == 's':
                pressed_keys.discard('s')
            elif key.char == 'q':
                pressed_keys.discard('q')
            elif key.char == 'd':
                pressed_keys.discard('d')
        except AttributeError:
            if key == keyboard.Key.esc:
                pressed_keys.discard('esc')

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while True:
        time.sleep(0.01)
        linear_displacement=0
        angular_displacement=0
        interaction = False
        # Wait for a key press
        while not pressed_keys:
            cv2.waitKey(10)
        if 'esc' in pressed_keys:
            break
        if 'z' in pressed_keys:
            linear_displacement = 0.05
        if 's' in pressed_keys:
            linear_displacement = -0.05
        if 'q' in pressed_keys:
            angular_displacement = 2
        if 'd' in pressed_keys:
            angular_displacement = -2
        if 'e' in pressed_keys:
            interaction = True
            pressed_keys.discard('e')
        if 'r' in pressed_keys:
            i += 1
            env = NAMOENV2D(env_list[i % 3])
            env.render()
            pressed_keys.discard('r')
        if 't' in pressed_keys:
            env.add_obstacle()
            pressed_keys.discard('t')
        else:pass
        env.step(linear_displacement,angular_displacement, interaction)
        env.render()
        env.info_display()
    cv2.destroyAllWindows()







