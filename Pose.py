from utils import mod, realco_to_cv2co, rotate_vec, draw_line, draw_vec, draw_text

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

    def __init__(self, pose=[0.0,0.0,0.0], parent=None, name=None, color=None, thickness=3):
        """
        Initializes the pose with x, y coordinates and angle.

        INPUT:
            - position (list/array): x, y coordinates of the pose relative to the parent
            - orientation (float): angle of the pose relative to the parent
            - parent (Pose or None): another pose that is already in the same graph
            - name (str or None)
        """
        self.x = pose[0]
        self.y = pose[1]
        self.a = mod(pose[2])
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        self.children = []
        self.name = name
        self.color = color
        self.thickness = thickness

    def __repr__(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return "unnamed_pose"
    
    def delete(self):
        if self.parent is not None:
            self.parent.children.remove(self)
    
    def ancestors(self):
        """Retrieves all ancestors of the current pose."""
        ancestors = []
        current_pose = self
        while current_pose is not None:
            ancestors.append(current_pose)
            current_pose = current_pose.parent
        return ancestors

    def descendants(self):
        """Retrieves all descendant poses of the current pose.
        (Retrieves all existing poses when called on the root pose.)"""
        descendants = [self]
        for child in self.children:
            descendants += child.descendants()
        return descendants

    def get_pose_local(self):
        return [self.x,self.y,self.a]
    
    def get_pose_brother(self, brother):
        if self.parent == brother.parent:
            # inverse of the pose_to_parent_frame function
            x,y = rotate_vec([brother.x-self.x, brother.y-self.y] , 180-brother.a)
            a = mod(self.a - brother.a)
            return  [x,y,a]
        else: return [self.x,self.y,self.a]

    def get_pose_parent(self):
        if self.parent is not None:
            # rotate the pose coordinates by the angle of the origin_pose to align with the new frame.
            rotated_x, rotated_y = rotate_vec([self.x,self.y], self.parent.a)
            # and translate of the origin_position to account for new frame's origin.
            x = self.parent.x + rotated_x
            y = self.parent.y + rotated_y
            # Adjust the angle by adding the rotation of the origin_pose
            a = mod(self.a + self.parent.a)
            return  [x,y,a]
        else: return [self.x,self.y,self.a]
    
    def get_pose_nthparent(self,n):
        """Express the pose in the nth parent's reference frame."""
        current_pose = self.get_pose_local()
        temp_pose = Pose(current_pose, self.parent)
        for _ in range (n):
            current_pose = temp_pose.get_pose_parent()
            temp_pose.delete()
            temp_pose = Pose(current_pose, temp_pose.parent.parent)
        temp_pose.delete()
        return  current_pose

    def get_pose_global(self):
        """Express the pose in its root's reference frame.
        (ie going back to the last parent frame)"""
        # get the coordinates of this pose in the root frame by climbing up to the last ancestor (origin pose)
        current_pose = self.get_pose_local()
        temp_pose = Pose(current_pose, self.parent)
        while temp_pose.parent is not None:
            current_pose = temp_pose.get_pose_parent()
            temp_pose.delete()
            temp_pose = Pose(current_pose, temp_pose.parent.parent)
        temp_pose.delete()
        return  current_pose
    
    def relocate(self, pose):
        self.x = pose[0]
        self.y = pose[1]
        self.a = mod(pose[2])
    
    def move_dxdyda(self, dx, dy, da):
        self.x += dx
        self.y += dy
        self.a = mod(self.a + da)
    
    def move_drda(self, dr, da):
        self.a = mod(self.a + da)
        dx, dy = rotate_vec([dr, 0],  self.a)
        self.x += dx
        self.y += dy
    
    def to_parentframe(self):
        """express the pose in its parent's reference frame. (ie: the grandparent is the new parent)"""
        if self.parent.parent is not None:
            self.relocate(self.get_pose_parent())
            self.parent.children.remove(self)
            self.parent.parent.children.append(self)
            self.parent = self.parent.parent
    
    def to_brother(self, brother):
        """express the pose in a brother's (pose with same parent) reference frame. (ie: the brother is the new parent)"""
        if self.parent == brother.parent:
            self.relocate(self.get_pose_brother(brother))
            if self.parent is not None:
                self.parent.children.remove(self)
            brother.children.append(self)
            self.parent = brother
    
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
        x, y, a = self.get_pose_global()
        vec = rotate_vec([length,0], a)
        draw_vec(img, resolution, vec, [x,y], color, thickness)
        draw_text(img, resolution, self.name, [x,y-length], color, thickness)
        if draw_position==True:
            parent_pose = self.parent
            if parent_pose is not None:
                origin_x, origin_y = parent_pose.get_pose_global()[0:2]
                draw_line(img, resolution, [origin_x,origin_y], [x,y], color, thickness)

    def draw_descendants(self, img, resolution, draw_position=True, thickness=7, length=0.2):
        pose_list = self.descendants()
        pose_list.remove(self)
        colors = [(255, 0, 0),(0, 255, 0),(0, 165, 255),(238, 130, 238),(0, 0, 255)]
        for pose in pose_list:
            color = pose.color
            if color is None:
                rank = len(pose.ancestors())-2
                if rank>4: rank=4
                color = colors[rank]
            pose.draw(img, resolution, color, thickness, length, draw_position)

def difference(pose1,pose2):
    pose1_x, pose1_y, pose1_a = pose1.get_pose_global()
    pose2_x, pose2_y, pose2_a = pose2.get_pose_global()
    angle = mod(pose2_a-pose1_a)
    vec_p1p2 = [pose2_x-pose1_x,pose2_y-pose1_y]
    length = (vec_p1p2[0]**2+vec_p1p2[1]**2)**(1/2)
    return  length, angle

def compare_poses(pose1,pose2,distance_lim=0.2,angle_lim=45):
    pose1_x, pose1_y, pose1_a = pose1.get_pose_global()
    pose2_x, pose2_y, pose2_a = pose2.get_pose_global()
    rounded_pose1 = [round(pose1_x/distance_lim)*distance_lim, 
                     round(pose1_y/distance_lim)*distance_lim, 
                     round(pose1_a/angle_lim)*angle_lim]
    rounded_pose2 = [round(pose2_x/distance_lim)*distance_lim, 
                     round(pose2_y/distance_lim)*distance_lim, 
                     round(pose2_a/angle_lim)*angle_lim]
    return rounded_pose1==rounded_pose2

if __name__ == "__main__":
    import time
    import cv2
    import numpy as np
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

    print(pose11.get_pose_local())
    test = pose11.get_pose_global()
    print(test)
    print(realco_to_cv2co([test[0],test[1]],resolution,img.shape[0]))

    origin.draw_descendants(img,resolution)

    cv2.namedWindow('Poses', cv2.WINDOW_NORMAL)
    cv2.imshow('Poses', img)
    cv2.resizeWindow('Poses', 500,500)
    cv2.waitKey(0)

    while True:
        t = time.time()
        # moving in circle
        pose1.move_drda(0.05, 5)
        pose11.move_drda(0.05,5)
        pose111.move_drda(0.05,5)
        poseX.move_drda(0.05, 5)
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
            poseX.to_parentframe()
        elif key == 122:  # Press "z" key
            poseX.to_brother(pose1)
        elif key == 101:  # Press "e" key
            poseX.to_brother(pose11)
        elif key == 114:  # Press "r" key
            poseX.to_brother(pose111)
        elif key == 113:  # Press "q" key
            resolution = 100
        elif key == 115:  # Press "s" key
            resolution *= 2
        elif key == 100:  # Press "d" key
            resolution //= 2 
        elif key == 27:  # Press "escape" key
            break
    cv2.destroyAllWindows()