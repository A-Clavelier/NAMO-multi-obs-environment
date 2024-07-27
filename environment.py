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
        self.imageshape = np.array(self.param_dict['shape'],dtype=np.float32)
        self.image = np.array(self.param_dict['image'], dtype=np.uint8)
        self.pick_poses = []
        for pick_pose in self.param_dict['pick_poses']:
            self.pick_poses.append(Pose(pick_pose,self.pose))
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
    
    def teleport(self, x, y, orientation):
        self.rollback_position = self.pose.position.copy()
        self.rollback_orientation = self.pose.orientation
        self.pose.teleport(x, y, orientation)

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
        row, col, orientation = self.pose.get_image_pose(self.env.RESOLUTION,mask.shape[0])
        # resize the image according to resolution and real shape
        image=cv2.resize(self.image.copy(), (int(self.imageshape[0]*self.env.RESOLUTION),int(self.imageshape[1]*self.env.RESOLUTION)), interpolation=cv2.INTER_NEAREST)
        # rotate the image to show object orientation
        rotatedimage = ndimage.rotate(image, orientation, reshape=True)
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
        - original_FO_map (np.uint8): fixed obstacle map, read from the map.pgm file in the map's subfolder. (0 -> free space ; 1 -> fixed obstacle)
        - original_SC_map (np.float32): Social cost map, read from the SCmap.npy in the map's subfolder.
        - config_dict (dict): dictionary descripting the initial configuration of the environment, read from the config_name.yaml in the map's subfolder.
        - MAP_SHAPE (list of floats): shape of the map in meters x meters
        - ORIGIN (pose): pose of the origin of the real coordinates on the map.
        - start (pose)
        - goal (pose)
    
        MODIFIABLE:
        - RESOLUTION (float): resolution of the obstacle map (in pxl/m).
        - FO_map (np.uint8): resizing of the original_FO_map to fit the MAP_SHAPE and RESOLUTION
        - SC_map (np.float32): resizing of the original_SC_map to fit the MAP_SHAPE and RESOLUTION
        - steps (int): current step number
        - ROB (MovableObject): robot for this episode.
        - MO_list (list of MovableObjects): MO list for this episode.
        - path_map (np.uint8): 1 in path, 0 outside of path
        - collision_map (np.uint8)
        - pickedMO (MovableObject): if None -> no picked MO at this step
                                    else -> this step's picked MO in the self.MO_list of movable obstacles.
    """


    def __init__(self,map_name="200x200_empty",config_name="config1", resolution=40, seed=10):
        """
        INPUT:
            - map_name (str)
            - config_name (str)
            - resolution (int)
            - seed (int)
        """
        t=time.time()
        random.seed(seed)
        # MAP parameters
        map_path="maps/"+map_name+"/map.pgm"
        SCmap_path="maps/"+map_name+"/SCmap.npy"
        self.original_FO_map =np.array( 1 - cv2.imread(map_path, 0)//230, dtype=np.uint8)
        self.original_SC_map = np.load(SCmap_path)
        # CONFIG parameters
        config_path="maps/"+map_name+"/"+config_name+".yaml"
        with open(config_path) as f:
            self.config_dict=yaml.load(f, Loader=yaml.SafeLoader)
        self.MAP_SHAPE=self.config_dict["map_shape"]
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
        self.set_resolution(resolution)
        self._check_collision()
        self.process_ping=time.time()-t
        if __name__ == "__main__":
            print(f"goal: {self.goal.get_local_pose()}, goal_image {self.goal.get_image_pose(self.RESOLUTION,self.FO_map.shape[0])}")
            print(f"start: {self.start.get_local_pose()}, start_image {self.start.get_image_pose(self.RESOLUTION,self.FO_map.shape[0])}")


    def set_resolution(self, resolution):
        self.RESOLUTION=int(resolution)
        self.FO_map = cv2.resize(self.original_FO_map, (int(self.MAP_SHAPE[0]*self.RESOLUTION),int(self.MAP_SHAPE[1]*self.RESOLUTION)), interpolation=cv2.INTER_NEAREST)
        self.SC_map = cv2.resize(self.original_SC_map, (int(self.MAP_SHAPE[0]*self.RESOLUTION),int(self.MAP_SHAPE[1]*self.RESOLUTION)), interpolation=cv2.INTER_NEAREST)


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
        if self.pickedMO is None:
            intersection_surface = (self.collision_map > 1).sum()
            if intersection_surface > 0:
                return True
            else:
                return False
        else:
            # if there is a picked MO, separate the collision check with the ROB and with the pickedMO to avoid them colliding.
            intersection_surface_1 = (self.collision_map-self.pickedMO.get_mask() > 1).sum()
            intersection_surface_2 = (self.collision_map-self.ROB.get_mask() > 1).sum()
            if intersection_surface_1 > 0 or intersection_surface_2 > 0:
                return True
            else:
                return False


    def add_obstacle(self, type=None, pose=None):
        t=time.time()
        if type is None:
            type = random.choice(["MOa","MOb","MOc"])
        if pose is None:
            max_tries = 1000
            for tries in range(max_tries):
                orig_x,orig_y = self.ORIGIN.position
                width, height = self.MAP_SHAPE
                rand_x = random.uniform(-orig_x,width-orig_x)
                rand_y = random.uniform(-orig_y,height-orig_y)
                rand_orientation = random.uniform(-180,180)
                new_MO = MovableObject(self,{'type': type, 'pose': [rand_x,rand_y,rand_orientation]}, f"{len(self.MO_list)}" )
                self.MO_list.append(new_MO)
                self.ROB.pose.to_brother_frame(new_MO.pose)
                for ROBpick in self.ROB.pick_poses:
                    for MOpick in new_MO.pick_poses:
                        pos_diff = MOpick.position - ROBpick.position
                        orientation_diff = mod(ROBpick.orientation - MOpick.orientation)
                        self.ROB.teleport(pos_diff[0], pos_diff[1], orientation_diff)
                        collision = self._check_collision()
                        self.render()
                        cv2.waitKey(0)
                        self.ROB.rollback()
                        if collision:
                            self.MO_list.pop()
                            break
                    if collision: break
                self.ROB.pose.to_parent_frame()
                if not collision: break
        else:
            self.MO_list.append(MovableObject(self,{'type': type, 'pose': pose}, f"{len(self.MO_list)}" ))
            if self._check_collision():
                    self.MO_list.pop()
                    print("created obstacle is in collision")
        self.process_ping=time.time()-t
    

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
            self.ROB.rollback()
            self._check_collision()
        if interaction:
            if self.pickedMO is None:
                done=False
                for MO in self.MO_list:
                    for MOpick in MO.pick_poses:
                        for ROBpick in self.ROB.pick_poses:
                            if compare_poses(ROBpick,MOpick,0.2,8):
                                MO.pose.to_brother_frame(self.ROB.pose)
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
        self.process_ping=time.time()-t


    def render(self, pose_resolution_factor=4):
        t = time.time()
        image = np.where(self.collision_map == 1, 0, 255).astype(np.uint8)
        if pose_resolution_factor >= 1:
            image = cv2.resize(image, (int(image.shape[1]*pose_resolution_factor),int(image.shape[0]*pose_resolution_factor)), interpolation=cv2.INTER_NEAREST)
            # Convert to a three-channel image using cvtColor
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.ORIGIN.draw_descendants(image,self.RESOLUTION*pose_resolution_factor, draw_position=False)
        cv2.namedWindow("render", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("render", 650, 650)
        cv2.moveWindow("render", 0, 0)
        cv2.imshow("render", image)
        cv2.waitKey(1)
        self.render_ping = time.time()-t
    

    def info_display(self):
        display_text =  (f"steps:{self.steps} \nRESOLUTION:{self.RESOLUTION}(pxl/m) \nprocess_ping:{self.process_ping:.4f} \nrender_ping:{self.render_ping:.4f} \npickedMO: {self.pickedMO}") 
        if __name__ == "__main__":
            display_text += f"\n -> z,q,s,d : move \n -> e : interact \n -> r : reset \n -> t : add obstacle "
            display_text += f"\n -> w, x : resolution*2, resolution/2  \n -> c, v : pose_resolution_factor +1, -1 "
        # Create an image with text
        text_image = np.zeros([650,300])
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
        cv2.resizeWindow(window_name, 300, 650)
        cv2.moveWindow(window_name, 650, 0)
        cv2.imshow(window_name, text_image)


if __name__ == "__main__":
    from pynput import keyboard

    env_list=["200x200_empty","200x200_map1","200x200_map2"]
    i = 0
    env = NAMOENV2D(env_list[i%3])
    pose_resolution_factor = 4
    env.render(pose_resolution_factor)
    env.info_display()
    pressed_keys = set()

    def on_press(key):
        try:
            if key.char == 'z':
                pressed_keys.add('z')
            elif key.char == 's':
                pressed_keys.add('s')
            elif key.char == 'q':
                pressed_keys.add('q')
            elif key.char == 'd':
                pressed_keys.add('d')
            elif key.char == 'e':
                pressed_keys.add('e')
            elif key.char == 'r':
                pressed_keys.add('r')
            elif key.char == 't':
                pressed_keys.add('t')
            elif key.char == 'f':
                pressed_keys.add('f')  
            elif key.char == 'w':
                pressed_keys.add('w')
            elif key.char == 'x':
                pressed_keys.add('x')
            elif key.char == 'c':
                pressed_keys.add('c')
            elif key.char == 'v':
                pressed_keys.add('v')       
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
        if 'w' in pressed_keys:
            env.set_resolution(env.RESOLUTION*2)
            pressed_keys.discard('w')
        if 'x' in pressed_keys:
            env.set_resolution(env.RESOLUTION/2)
            pressed_keys.discard('x')
        if 'c' in pressed_keys:
            pose_resolution_factor += 1
            pressed_keys.discard('c')
        if 'v' in pressed_keys:
            pose_resolution_factor -= 1
            pressed_keys.discard('v')
        env.step(linear_displacement,angular_displacement, interaction)
        env.render(pose_resolution_factor)
        env.info_display()
    cv2.destroyAllWindows()







