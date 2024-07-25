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
        start_row = row-LenRow//2
        end_row = row+(LenRow-LenRow//2)
        start_col = col-LenCol//2
        end_col = col+(LenCol-LenCol//2)
        slice_height = end_row - start_row
        slice_width = end_col - start_col
        # Check if slice dimensions match the RotatedImage dimensions
        if slice_height==rotatedimage.shape[0] and slice_width==rotatedimage.shape[1] :
            # Check if the indices are within the bounds of the map
            if (0 <= start_row < mask.shape[0] and 0 <= end_row <= mask.shape[0] and
                0 <= start_col < mask.shape[1] and 0 <= end_col <= mask.shape[1]):
                mask[start_row : end_row, start_col : end_col] += rotatedimage
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
    
        MODIFIABLE:
        - episodes (int): current episode number
        - steps (int): current step number
        - start (pose)
        - goal (pose)
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
        random.seed(seed)
        ### INITIALISE FIXED ATTRIBUTES:
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
        ### INITIALISE MODIFIABLE ATTRIBUTES:
        # count the number of episodes
        self.episodes = 0
        self.reset()


    def reset(self, random_offset=False, random_position=False, MO_number = 1 ):
        """reset environment to initial configuration
            INPUT:
                - rand_offset (bool): add a random offset to initial positions
                - rand_pos (bool): reset environment to a randomly chosen initial configuration instead.
        """
        t=time.time()
        # update counters
        self.steps = 1
        self.episodes += 1
        # empty MO list
        self.MO_list=[]
        self.pickedMO = None

        max_tries = 10000
        min_path_len = 3 #unit: m
        if random_position:
            pass
        #     free_ImgPos = np.where(self.FO_map == 0)
        #     free_ImgPos_list = list(zip(free_ImgPos[0], free_ImgPos[1])) 
        #     for tries in range(max_tries):
        #         self.goal = Pose(free_ImgPos_list[np.random.choice(len(free_ImgPos_list))], random.randint(0,359), self.ORIGIN, "goal")
        #         self.ROB = Robot(env=self, init_dict={'name':'ROB', 'pose':free_ImgPos_list[np.random.choice(len(free_ImgPos_list))]+[random.randint(0,359)] })
        #         self.get_path_map()
        #         # units: cells x (pxl/cell) / (pxl/m) = m OK
        #         if len(self.path_row)*self.DIJKSTRA_RESOLUTION/self.REAL_RESOLUTION > min_path_len:
        #             break
        #     for i in range(MO_number):
        #         for tries in range(max_tries):
        #             MO_init_angle = random.randint(0,359)
        #             MO_init_ImgPos = free_ImgPos_list[np.random.choice(len(free_ImgPos_list))]
        #             MO_init_RealPos = convert_ImgPos_to_RealPos(MO_init_ImgPos, self.ORIGIN_PXL, self.REAL_RESOLUTION)
        #             MO_name = random.choice(['MO1','MO2'])
        #             self.MO.append(MovableObstacle(env=self, init_dict={'name': MO_name, 'init_RealPos': MO_init_RealPos, 'init_angle': MO_init_angle}))
        #             #check if there is enough space for the ROB to pick the MO
        #             for pick_pose in self.MO[i].pick_poses:
        #                 test_position = [x*1.5 for x in pick_pose["position"]]
        #                 pick_RealPos = [self.MO[i].get_RealPos()[0]+test_position[0]*math.cos(math.radians(self.MO[i].get_angle()))-test_position[1]*math.sin(math.radians(self.MO[i].get_angle())),
        #                                  self.MO[i].get_RealPos()[1]+test_position[0]*math.sin(math.radians(self.MO[i].get_angle()))+test_position[1]*math.cos(math.radians(self.MO[i].get_angle()))]
        #                 pick_angle = self.MO[i].get_angle() + pick_pose['angle']
        #                 self.MO.append(MovableObstacle(env=self, init_dict={'name': 'ROB', 'init_RealPos':pick_RealPos, 'init_angle':pick_angle}))
        #             if self.verbose > 1 : self.small_render(); cv2.waitKey(100)
        #             self._check_collision()
        #             if not self.collision :
        #                 for _ in range(len(self.MO[i].pick_poses)):
        #                     self.MO.pop()
        #                 self._check_success()
        #                 if self.MO[i].InPath:
        #                     break
        #                 else:
        #                     self.MO.pop()
        #             else:
        #                 for _ in range(len(self.MO[i].pick_poses)):
        #                     self.MO.pop()
        #                 self.MO.pop()

        # elif random_offset:
        #         for tries in range(max_tries):
        #             #add a random offset array to the goal_pos attribute
        #             self.goal_RealPos = self.goal_init_RealPos + np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)],dtype=np.float32)
        #             self.goal_ImgPos = convert_RealPos_to_ImgPos(self.goal_RealPos, self.ORIGIN_PXL, self.REAL_RESOLUTION)
        #             #add a random offset array to the robot's starting position
        #             self.ROB = Robot(env=self, init_dict=self.ROB_init_dict)
        #             self.ROB.set_RealPos(self.ROB.get_RealPos()+np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)],dtype=np.float32))
        #             self.ROB.set_angle(self.ROB.get_angle()+random.uniform(-30, 30))
        #             self.get_path_map()
        #             # units: cells x (pxl/cell) / (pxl/m) = m OK
        #             if len(self.path_row)*self.DIJKSTRA_RESOLUTION/self.REAL_RESOLUTION > min_path_len:
        #                 break
        #         for i in range(len(self.MO_init_dicts_list)):
        #             for tries in range(max_tries):
        #                 #for each MO, add a random offset array [offset_x,offset_y] to the pos array [x,y]
        #                 self.MO.append(MovableObstacle(env=self, init_dict=self.MO_init_dicts_list[i]))
        #                 self.MO[i].set_RealPos(self.MO[i].get_RealPos()+np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)],dtype=np.float32))
        #                 self.MO[i].set_angle(self.MO[i].get_angle()+random.uniform(-30, 30))
        #                 self._check_collision()
        #                 if not self.collision:
        #                     break
        #                 else:
        #                     self.MO.pop()

        else:
            self.goal = Pose(self.config_dict['goal_pose'],self.ORIGIN,"goal",color=(0,0,255))
            self.start = Pose(self.config_dict['ROB_init_dict']['pose'],self.ORIGIN,"start",color=(0,0,255))
            self.ROB = MovableObject(self,self.config_dict['ROB_init_dict'],'ROB')
            i=0
            for MO_init_dict in self.config_dict['MO_init_dicts_list']:
                self.MO_list.append(MovableObject(self,MO_init_dict,f"{i}"))
                i+=1
        self._check_collision()
        self.ping=time.time()-t
        # print the starting attributes 
        if __name__ == "__main__":
            print(f"goal: {self.goal.get_local_pose()}, goal_image {self.goal.get_image_pose(self.REAL_RESOLUTION,self.FO_map.shape[0])}")
            print(f"start: {self.start.get_local_pose()}, start_image {self.start.get_image_pose(self.REAL_RESOLUTION,self.FO_map.shape[0])}")
        
    
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
        print()
        if interaction:
            if self.pickedMO is None:
                done=False
                for MO in self.MO_list:
                    for MOpick in MO.pick_poses:
                        for ROBpick in self.ROB.pick_poses:
                            if compare_poses(ROBpick,MOpick,0.1,10):
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
        image = np.where(self.collision_map == 1, 0, 255).astype(np.uint8)
        if render_pose:
            res_factor = 4
            image = cv2.resize(image, (image.shape[1]*res_factor,image.shape[0]*res_factor), interpolation=cv2.INTER_NEAREST)
            # Convert to a three-channel image using cvtColor
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.ORIGIN.draw_descendants(image,self.REAL_RESOLUTION*res_factor)
        cv2.namedWindow("render", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("render", 400, 400)
        cv2.moveWindow("render", 0, 0)
        cv2.imshow("render", image)
        cv2.waitKey(1)
    
    def info_display(self):
        display_text =  (f"episodes:{self.episodes} | steps:{self.steps} | ping:{self.ping:.4f}")  
        if self.pickedMO is not None:
            display_text += f"\n  pickedMO: {self.pickedMO.pose.name}"
        if __name__ == "__main__":
            display_text += f"\n press z,q,s,d to move \n press spacebar to pick/put"
        # Create an image with text
        text_image = np.zeros([150,400])
        font_scale = 0.6
        color = 255
        thickness = 2
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
    env = NAMOENV2D("200x200_map2")
    while True:
        linear_displacement=0
        angular_displacement=0
        interaction = False
        # Wait for a key press
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  # Press "space" key
            interaction = True
        elif key == 122:  # Press "z" key
            linear_displacement=0.05
        elif key == 115:  # Press "s" key
            linear_displacement=-0.05
        elif key == 113:  # Press "q" key
            angular_displacement=2
        elif key == 100:  # Press "d" key
            angular_displacement=-2
        elif key == 27:
            break
        env.step(linear_displacement,angular_displacement, interaction)
        env.render()
        env.info_display()
    cv2.destroyAllWindows()







