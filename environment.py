import time
import random
import cv2
import yaml
import numpy as np
from scipy import ndimage

from Pose import Pose, compare_poses
from utils import realco_to_arrayco


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
        self.rollback_pose = self.pose.get_pose_local()

    def __del__(self):
        self.pose.delete()
    
    def __repr__(self) -> str:
        return str(self.pose)

    def save_pose(self):
        self.rollback_pose = self.pose.get_pose_local()

    def rollback(self):
        self.pose.relocate(self.rollback_pose)
    
    def get_mask(self):
        """mask is used to detect collisions"""
        # create mask with shape of global map
        mask = np.zeros(self.env.FO_map.shape, dtype=np.uint8)
        # get the MovableObject's pose in the mask array coordinates
        x,y,a = self.pose.get_pose_global()
        row,col = realco_to_arrayco([x,y],self.env.RESOLUTION,mask.shape[0])
        # resize the image according to resolution and real shape
        image=cv2.resize(self.image.copy(), (int(self.imageshape[0]*self.env.RESOLUTION),int(self.imageshape[1]*self.env.RESOLUTION)), interpolation=cv2.INTER_NEAREST)
        # rotate the image to show object orientation
        rotatedimage = ndimage.rotate(image, a, reshape=True)
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
        # if there is more pixels with 2 than a max_surface, we consider there is a collision.
        max_surface = 0
        if self.pickedMO is None:
            intersection_surface = (self.collision_map > 1).sum()
            if intersection_surface > max_surface:
                return True
            else:
                return False
        else:
            # if there is a picked MO, separate the collision check with the ROB and with the pickedMO to avoid them colliding.
            intersection_surface_1 = (self.collision_map.copy()-self.pickedMO.get_mask() > 1).sum()
            intersection_surface_2 = (self.collision_map.copy()-self.ROB.get_mask() > 1).sum()
            if intersection_surface_1 > max_surface or intersection_surface_2 > max_surface:
                return True
            else:
                return False

    def add_obstacle(self, type=None, pose=None):
        t=time.time()
        if type is None:
            type = random.choice(["MOa","MOb","MOc"])
        if pose is None:
            max_tries = 1000
            for _ in range(max_tries):
                orig_x,orig_y = self.ORIGIN.x, self.ORIGIN.y
                width, height = self.MAP_SHAPE
                rand_x = random.uniform(-orig_x,width-orig_x)
                rand_y = random.uniform(-orig_y,height-orig_y)
                rand_orientation = random.uniform(-180,180)
                new_MO = MovableObject(self,{'type': type, 'pose': [rand_x,rand_y,rand_orientation]}, f"{len(self.MO_list)}" )
                self.MO_list.append(new_MO)
                if not self._check_collision():
                    pickable=False
                    for pick_pose in new_MO.pick_poses:
                        test_pose = Pose([0,0,0],pick_pose)
                        test_pose.move_dxdyda(-0.25,0,0)
                        #express the test pose in the origin of the environment reference frame
                        test_pose_2ndparent = test_pose.get_pose_nthparent(2)
                        self.MO_list.append(MovableObject(self,{'type': 'ROB', 'pose': test_pose_2ndparent}, "ROB_test"))
                        if not self._check_collision():
                            pickable=True
                        test_pose.delete()
                        self.MO_list.pop()
                    if pickable:
                        break
                self.MO_list.pop()
        else:
            self.MO_list.append(MovableObject(self,{'type': type, 'pose': pose}, f"{len(self.MO_list)}" ))
        self.process_ping=time.time()-t
    
    def step(self, dx, dy, da, interaction=False):
        """the environment takes a timestep, the robot executes the inputed actions."""
        t=time.time()
        self.steps += 1
        self.ROB.save_pose()
        self.ROB.pose.move_dxdyda(dx, dy, da)
        if self._check_collision():
            self.ROB.rollback()
        if interaction:
            if self.pickedMO is None:
                done=False
                for MO in self.MO_list:
                    for MOpick in MO.pick_poses:
                        for ROBpick in self.ROB.pick_poses:
                            if compare_poses(ROBpick,MOpick,0.2,45):
                                MO.pose.to_brother(self.ROB.pose)
                                self.pickedMO = MO
                                done=True
                                break
                        if done:
                            break
                    if done:
                        break
            else:
                self.pickedMO.pose.to_parentframe()
                self.pickedMO = None
        self.process_ping=time.time()-t

    def render(self, pose_resolution_factor=4, draw_position=False, thickness=7,  arrow_length=0.2):
        t = time.time()
        self._check_collision()
        image = np.where(self.collision_map.copy() == 1, 0, 255).astype(np.uint8)
        if pose_resolution_factor >= 1:
            image = cv2.resize(image, (int(image.shape[1]*pose_resolution_factor),int(image.shape[0]*pose_resolution_factor)), interpolation=cv2.INTER_NEAREST)
            # Convert to a three-channel image using cvtColor
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.ORIGIN.draw_descendants(image,self.RESOLUTION*pose_resolution_factor, draw_position, thickness, arrow_length)
        cv2.namedWindow("render", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("render", 650, 650)
        cv2.moveWindow("render", 0, 0)
        cv2.imshow("render", image)
        cv2.waitKey(1)
        self.render_ping = time.time()-t 

    def info_display(self, string=""):
        x,y,a=self.ROB.pose.get_pose_local()
        display_text = f"steps:{self.steps} \nROB:[{x:.2f},{y:.2f},{a:.1f}]"
        display_text += f"\npickedMO: {self.pickedMO}"
        display_text += f"\nRESOLUTION:{self.RESOLUTION}(pxl/m)"
        display_text += f"\nprocess_ping:{self.process_ping:.4f} \nrender_ping:{self.render_ping:.4f} "
        display_text += string
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
    # Initialize variables
    map_list = ["200x200_empty", "200x200_map1", "200x200_map2"]
    config_list = ["config1", "config2"]
    i = 0
    j = 0
    env = NAMOENV2D(map_list[i % 3], config_list[j % 2])
    pose_resolution_factor = 4
    env.render(pose_resolution_factor)
    info_string = "\n -> z,q,s,d : move \n -> a,e : rotate \n -> f : interact \n -> r : reset \n -> t : add obstacle \n -> w, x : resolution *2,/2 \n -> c, v : pose_res_factor +1,-1"
    env.info_display(info_string)
    pressed_keys = set()
    # Define key action mappings
    key_actions = {
        'z': lambda: update_motion(0, 0.1, 0),
        's': lambda: update_motion(0, -0.1, 0),
        'q': lambda: update_motion(-0.1, 0, 0),
        'd': lambda: update_motion(0.1, 0, 0),
        'a': lambda: update_motion(0, 0, 22.5),
        'e': lambda: update_motion(0, 0, -22.5),
        'r': lambda: reset_environment(),
        't': lambda: env.add_obstacle(),
        'f': lambda: update_interaction(),
        'w': lambda: update_resolution(env.RESOLUTION * 2),
        'x': lambda: update_resolution(env.RESOLUTION / 2),
        'c': lambda: update_pose_resolution(1),
        'v': lambda: update_pose_resolution(-1)
    }
    def update_motion(dx_change, dy_change, da_change):
        global dx, dy, da
        dx += dx_change
        dy += dy_change
        da += da_change
    def reset_environment():
        global i, j, env
        i += 1
        if i % 3 == 0:
            j += 1
        env = NAMOENV2D(map_list[i % 3], config_list[j % 2])
        env.render()
        pressed_keys.discard('r')
    def update_interaction():
        global interaction
        interaction = True
        pressed_keys.discard('f')
    def update_resolution(new_resolution):
        env.set_resolution(new_resolution)
        pressed_keys.discard('w')
        pressed_keys.discard('x')
    def update_pose_resolution(factor_change):
        global pose_resolution_factor
        pose_resolution_factor += factor_change
        pressed_keys.discard('c')
        pressed_keys.discard('v')
    # Keyboard event handlers
    def on_press(key):
        try:
            if key.char in key_actions:
                pressed_keys.add(key.char)
        except AttributeError:
            if key == keyboard.Key.esc:
                pressed_keys.add('esc')
    def on_release(key):
        try:
            if key.char in pressed_keys:
                pressed_keys.discard(key.char)
        except AttributeError:
            if key == keyboard.Key.esc:
                pressed_keys.discard('esc')
    # Start the keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    # Main loop
    while True:
        time.sleep(0.1)
        dx = dy = da = 0
        interaction = False
        if not pressed_keys:
            cv2.waitKey(0)
        if 'esc' in pressed_keys:
            break
        for key in pressed_keys.copy():
            if key in key_actions:
                key_actions[key]()
        env.step(dx, dy, da, interaction)
        env.render(pose_resolution_factor)
        env.info_display(info_string)
    cv2.destroyAllWindows()