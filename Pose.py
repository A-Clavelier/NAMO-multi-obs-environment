import numpy as np
import math
import cv2
from utils import mod, rotate_vec, pose_to_parent_frame, pose_to_brother_frame, draw_line, draw_vec, draw_text

#ONLY SUPPORTS ACYCLIC AND MONOPARENTAL HIERARCHY OF POSES (tree)
#THE ROOT OF THE POSE TREE SHOULD BE THE ORIGIN OF THE IMAGE 
# (because when displaying a pose, whe express its coordinates in the root frame)
class Pose:
    """
    Represents a pose in 2D space with x, y coordinates and an angle.
    Supports hierarchical poses with a parent-child relationship.

    ATTRIBUTES:
        - parent (pose or None): a reference to a parent pose for hierarchical transformations
        - children (list of poses): list of the poses for which this pose is the parent.
        - Name (str)
        - position (list): [x,y] coordinates of the pose
        - orientation (float): angle of orientation of the pose
    """

    def __init__(self, pose=[0.0,0.0,0.0], parent=None, name=None, color=None):
        """
        Initializes the pose with x, y coordinates and angle.

        INPUT:
            - position (list/array): x, y coordinates of the pose relative to the parent
            - orientation (float): angle of the pose relative to the parent
            - parent (Pose or None): another pose that is already in the same graph
            - name (str or None)
        """
        self.position = np.array(pose[0:2],dtype=np.float32)
        self.orientation = mod(pose[2])  # Ensure orientation is within [0, 360)
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        self.children = []
        self.name = name
        self.color = color

    def __repr__(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return "unnamed_pose"

    def get_ancestors(self):
        """
        Retrieves all ancestors of the current pose.
        
        OUTPUT:
            - (list): List of ancestor poses.
        """
        ancestors = []
        current_pose = self
        while current_pose is not None:
            ancestors.append(current_pose)
            current_pose = current_pose.parent
        return ancestors

    def get_descendants(self):
        """
        Retrieves all descendant poses of the current pose.
        (Retrieves all poses when called on the root pose.)
        
        OUTPUT:
            - (list): List of descendant poses.
        """
        descendants = [self]
        for child in self.children:
            descendants += child.get_descendants()
        return descendants
    
    def teleport(self, x, y, orientation):
        self.position = np.array([x,y],dtype=np.float32)
        self.orientation = mod(orientation)
    
    def relocate(self, dx, dy, dtheta):
        """
        relocates the pose by translating and/or rotating it.
        """
        self.position += np.array([dx, dy], dtype=np.float32)
        self.orientation = mod(self.orientation + dtheta)
    
    def move(self, linear_displacement=0.0, angular_displacement=0.0):
        """
        Moves the pose by translating along its orientation axis and rotating it.

        INPUT:
            - linear_displacement (float): displacement along the pose's orientation axis
            - angular_displacement (float): orientation angle variation
        """
        dx, dy = rotate_vec([linear_displacement, 0], self.orientation) #mod(self.orientation+angular_displacement) ???
        self.relocate(dx, dy, angular_displacement)
    
    def to_parent_frame(self):
        """
        express the pose in its parent's reference frame.
        """
        if self.parent.parent is not None:
            self.position,self.orientation=pose_to_parent_frame(self.position,self.orientation,self.parent.position,self.parent.orientation)
            self.parent.children.remove(self)
            self.parent.parent.children.append(self)
            self.parent = self.parent.parent
    
    def to_brother_frame(self, brother):
        """
        express the pose in a brother (pose with same parent) reference frame.

        INPUT:
            - brother (Pose)
        """
        if self.parent == brother.parent:
            self.position, self.orientation = pose_to_brother_frame(self.position, self.orientation, 
                                                                    brother.position, brother.orientation)
            if self.parent is not None:
                self.parent.children.remove(self)
            brother.children.append(self)
            self.parent = brother
            
    def get_local_pose(self):
        """
        OUTPUT:
            - (list): [x, y, orientation] in the local frame
        """
        return [self.position[0],self.position[1],self.orientation]
    
    def get_global_pose(self):
        """
        Express the pose in its root's reference frame.
        (ie going back to the last parent frame)

        OUTPUT:
            - (list): [x, y, orientation] in the image frame
        """
        # get the coordinates of this pose in the root frame by climbing up to the last ancestor (origin pose)
        current_position = self.position
        current_orientation = self.orientation
        current_parent = self.parent
        while current_parent is not None:
            current_position, current_orientation = pose_to_parent_frame(current_position,current_orientation,
                                                                        current_parent.position,current_parent.orientation)
            current_parent = current_parent.parent
        return  [current_position[0], current_position[1], current_orientation]

    def get_image_pose(self, resolution, img_height):
        """
        Express the pose in its root's reference frame.
        (ie going back to the last parent frame)
        And multiply by the resolution to display the pose. 
        (used to draw the poses on an image)

        INPUT:
            - resolution (int): resolution of the image in pxl/m
            - img_height (int)

        OUTPUT:
            - (list): [row, col, orientation] in the image frame
        """
        x,y,orientation = self.get_global_pose()
        # transform the coordinates in root frame to account for image resolution and inverse the y axis to get the line indices in an array
        row = int(img_height - y*resolution)
        col = int(x*resolution)
        return  [row, col, orientation]
    
    def draw(self, img, resolution, color, thickness, length, draw_position=True):
        """
        Draws an arrow representing the pose on the given image.

        INPUTS:
            img (array): The image to draw on.
            resolution (float): pxl/m
            length (float): the length in pixels of the arrow representing the pose.
            color (tuple): (B,G,R) The color of the line.
            thickness (int): The thickness of the line.
        """
        thickness = math.ceil(thickness * resolution/200)
        row, col, orientation = self.get_image_pose(resolution, img.shape[0])
        vec = rotate_vec([length*resolution,0], orientation)
        draw_vec(img, vec, [col,row], color, thickness)
        draw_text(img, self.name, [col,row+length*resolution], color, math.ceil(thickness/2), resolution/150)
        if draw_position==True:
            parent_pose = self.parent
            if parent_pose is not None:
                origin_row,origin_col = parent_pose.get_image_pose(resolution,img.shape[0])[0:2]
                end_row,end_col = self.get_image_pose(resolution,img.shape[0])[0:2]
                draw_line(img,[origin_col,origin_row],[end_col,end_row], color, math.ceil(thickness/3))

    def draw_descendants(self, img, resolution, draw_position=True, thickness=7, arrow_length=0.2):
        pose_list = self.get_descendants()
        pose_list.remove(self)
        colors = [(255, 0, 0),(0, 255, 0),(0, 165, 255),(238, 130, 238),(0, 0, 255)]
        for pose in pose_list:
            rank = len(pose.get_ancestors())-2
            if pose.color is not None:
                color = pose.color
            else:
                color = colors[rank]
            pose.draw(img, resolution, color, (thickness-rank), arrow_length-rank*arrow_length/10, draw_position)

def difference(pose1,pose2):
    pose1_x, pose1_y, pose1_orientation = pose1.get_global_pose()
    pose2_x, pose2_y, pose2_orientation = pose2.get_global_pose()
    return [pose2_x-pose1_x,pose2_y-pose1_y], mod(pose2_orientation-pose1_orientation)


def compare_poses(pose1,pose2,distance_lim,angle_lim):
    vec_p1p2, angle = difference(pose1,pose2)
    length = np.sqrt(vec_p1p2[0]**2+vec_p1p2[1]**2)
    if length < distance_lim and abs(angle) < angle_lim:
        return True
    else:
        return False


if __name__ == "__main__":
    import time
    # Create a blank image
    resolution = 100 #pxl/m
    img = np.ones((8*resolution, 8*resolution, 3), dtype=np.uint8) * 255
    # Define poses with parent-child relationships
    origin = Pose([4,4,0], name="origin")
    pose1 = Pose([1,0,45], origin, "1")
    pose11 = Pose([1,0,45], pose1, "11")
    pose111 = Pose([1,0,45], pose11, "111")
    poseX = Pose([1,0,0], pose111, "X")
    poseX1 = Pose([0.5,0,0], poseX, "X1")
    poseX2 = Pose([0,0.5,90], poseX, "X2")
    poseX3 = Pose([-0.5,0,180], poseX, "X3")
    poseX4 = Pose([0,-0.5,-90], poseX, "X4")

    print(pose11.get_local_pose())
    print(pose11.get_image_pose(resolution,img.shape[0]))

    origin.draw_descendants(img,resolution)

    cv2.namedWindow('Poses', cv2.WINDOW_NORMAL)
    cv2.imshow('Poses', img)
    cv2.resizeWindow('Poses', 500,500)
    cv2.waitKey(0)

    while True:
        t = time.time()
        # moving in circle
        pose1.move(0.05, 5)
        pose11.move(0.05,5)
        pose111.move(0.05,5)
        poseX.move(0.05, 5)
        # create the image
        img = np.ones((8*resolution, 8*resolution, 3), dtype=np.uint8) * 255
        origin.draw_descendants(img,resolution)
        # computre fps and add text info
        dt = time.time()-t
        cv2.putText(img, f"res={resolution} | dt={dt:.4f}", (0,resolution), cv2.FONT_HERSHEY_SIMPLEX, resolution/50, (0,0,0), int(resolution/50))
        cv2.putText(img, "press a,z,e to change referential of X", (0,img.shape[0]-resolution), cv2.FONT_HERSHEY_SIMPLEX, resolution/100, (0,0,0), int(resolution/100))
        cv2.putText(img, "press q,s,d to change resolution", (0,img.shape[0]-resolution//2), cv2.FONT_HERSHEY_SIMPLEX, resolution/100, (0,0,0), int(resolution/100))
        cv2.imshow('Poses', img)
        # Wait for a key press
        key = cv2.waitKey(0) & 0xFF
        # Check which key was pressed and execute the corresponding action
        if key == 97:  # Press "a" key
            poseX.to_parent_frame()
        elif key == 122:  # Press "z" key
            poseX.to_brother_frame(pose1)
        elif key == 101:  # Press "e" key
            poseX.to_brother_frame(pose11)
        elif key == 114:  # Press "r" key
            poseX.to_brother_frame(pose111)
        elif key == 113:  # Press "q" key
            resolution = 100
        elif key == 115:  # Press "s" key
            resolution *= 2
        elif key == 100:  # Press "d" key
            resolution //= 2 
        elif key == 27:  # Press "escape" key
            break
    cv2.destroyAllWindows()