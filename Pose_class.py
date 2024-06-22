import numpy as np
import cv2
from utils import mod, rotate_vec, change_frame_pose, draw_vec, draw_line
from Graph_class import Graph

class Pose:
    """
    Represents a pose in 2D space with x, y coordinates and an angle.
    Supports hierarchical poses with a parent-child relationship.

    ATTRIBUTES:
        FIXED:
        - parent (pose or None): a reference to a parent pose for hierarchical transformations
        - scale (array): [x_scale, y_scale] to enable conversion (reverse axes, manage resolution)
        - Name (str)
        MODIFIABLE:
        - position (list): [x,y] coordinates of the pose
        - orientation (float): angle of orientation of the pose
    """

    def __init__(self, position=[0.0,0.0], orientation=0.0, graph=None, parent=None, name=None, scale=(1,1)):
        """
        Initializes the pose with x, y coordinates and angle.

        INPUT:
            - position (list/array): x, y coordinates of the pose relative to the parent
            - orientation (float): angle of the pose relative to the parent
            - graph (Graph or None): The graph managing the hierarchy
            - parent (Pose or None): another pose that is already in the same graph
            - name (str or None)
            - scale (tuple): [x_scale, y_scale] to enable conversion (reverse axes, manage resolution)
        """
        #MODIFIABLE:
        self.position = np.array(position,dtype=np.float32)
        self.orientation = mod(orientation)  # Ensure orientation is within [0, 360)
        #FIXED:
        if parent is not None:
            self.graph = parent.graph
            self.graph.add_node(self)
            self.set_parent(parent)
        elif graph is not None:
            self.graph = graph
            self.graph.add_node(self)
        else: self.graph = None
        # possibility of multiple parents for a pose transformed in another base???
        # child list attribute ???
        self.name = name
        self.scale = np.array(scale, dtype=np.float32) 
        
    def __repr__(self):
        """
        string representation of the pose object's attributes.
        """
        if self.name is None:
            return f"xy=[{','.join([f'{x:.2f}' for x in self.position])}] orientation={self.orientation:.2f}"
        else:
            return self.name
    
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
    
    def set_parent(self, parent_pose):
        """
        Sets the parent of this pose in the graph.

        INPUT:
            - parent_pose (Pose): The parent pose to be set
        """
        if self.graph is not None:
            self.graph.add_edge(parent_pose, self)
    
    def get_parent(self):
        """
        Get the parent of this pose in the graph.

        OUTPUT:
            - parent_pose (Pose)
        """
        if self.graph is None:
            return None
        ancestors = self.graph.get_ancestors(self)
        if not ancestors:
            # if the ancestors list is empty []
            return None
        parent_pose = ancestors[0]
        return parent_pose
    
    def _relative_to(self,ref_pose):
        """
        Transforms a pose to the reference frame constituted by the parent .
        That is to say give the pose relative to it's parent.

        OUTPUT:
            - (pose): the transformed pose relative to
        """
        if ref_pose is None:
            return self.relative_to_base()
        ancestors = self.graph.get_ancestors(self)
        # if the ref_pose relative to which we want to express this pose is an ancestor
        # we simply need to  express the pose in the parent's frames enought times,
        # iteratively climbing the hierarchy of parents.
        if ref_pose in ancestors:
            
            position = self.position
            orientation = self.orientation
            parent = self.get_parent()
            while parent is not ref_pose:
                relative_position = current_parent.position + rotate_vec(self.position, parent_pose.orientation)
                relative_orientation = mod(self.orientation + parent_pose.orientation)
                current_parent = current_parent.get_parent()
            return  current_position, current_orientation



        
        #create a temporary pose not linked in the graph just used for coordinates transform calculations
        dummy_pose = Pose(relative_position, relative_orientation)
        parent_frame = parent_pose.get_parent()
        return dummy_pose, parent_frame
    
    def _relative_to_child(self,child_pose):
    
    def relative_to_base(self):
        """
        Transforms the pose into the base reference frame by applying all
        transformations up the hierarchy of parent poses.

        OUTPUT:
            - (list, float): ([new_x, new_y], new_theta) coordinates of the position and orientation in the base frame
        """
        current_position = self.position
        current_orientation = self.orientation
        current_parent = self.get_parent()
        while current_parent is not None:
                current_position, current_orientation = change_frame_pose(current_position,current_orientation,
                                                                          current_parent.position,current_parent.orientation)
                current_parent = current_parent.get_parent()
        return  current_position, current_orientation
    
    # def to_nth_parent_frame(self, n=1):
    #     """
    #     Transforms the pose into the parent reference frame n times

    #     INPUT:
    #         - n (int>0): number of parents to go back

    #     OUTPUT:
    #         - (pose): the transformed pose in the nth parent reference frame
    #     """
    #     current_position = self.position
    #     current_orientation = self.orientation
    #     current_parent = self.get_parent()
    #     for _ in range(n):
    #             current_position, current_orientation = change_frame_pose(current_position,current_orientation,
    #                                                                       current_parent.position,current_parent.orientation)
    #             current_parent = current_parent.get_parent()
    #     return  current_position, current_orientation
    
    # def get_relative_to(self, reference_pose):
    #     """
    #     Transforms the pose into the reference frame of the specified reference pose.

    #     INPUT:
    #         - reference_pose (pose or None): we want to express the current pose in the reference_pose's frame
    #                                          if None, we consider that it is the base frame reference frame

    #     OUTPUT:
    #         - (pose): the transformed pose in the specified reference pose's frame.
    #     """
    #     if reference_pose is None:
    #         return self.to_base_frame()
    #     common_parent, self_path, ref_path = self._get_first_common_parent(reference_pose)
    #     # put this pose and the relative pose in the common_parent's frame
    #     # (if the common parent is None it is equivalent to putting them in the base frame)
    #     self_InCommonFrame = self.to_nth_parent_frame(n=len(self_path))
    #     ref_InCommonFrame = reference_pose.to_nth_parent_frame(n=len(ref_path))
    #     # now that self and ref are projected in a common frame, we can get the vector between their positions
    #     vec_ref_to_self = self_InCommonFrame.position - ref_InCommonFrame.position
    #     # compute the position and angle of the current pose relative to the ref pose
    #     rel_position = rotate_vec(vec_ref_to_self, -ref_InCommonFrame.orientation)
    #     rel_angle = self_InCommonFrame.orientation - ref_InCommonFrame.orientation
    #     return pose(rel_position, rel_angle, parent=reference_pose)
    
    # def _get_first_common_parent(self, reference_pose):
    #     """
    #     Finds the first common parent between this pose and a reference pose.
        
    #     INPUT:
    #         - reference_pose (pose or None):The pose to compare with.
    #                                         if None, will return (None, self_ancestors, None)
            
    #     OUTPUT:
    #         - (pose or None):   The first common parent pose. 
    #                             None if there is no common parent
    #         - (list):   Path from this pose to the common parent. 
    #                     self ancestors list if no common parent.
    #         - (list):   Path from the reference pose to the common parent.
    #                     ref ancestors list if no common parent.
    #     """
    #     self_ancestors = self._get_ancestors()
    #     if reference_pose is None:
    #         return None, self_ancestors, None
    #     ref_ancestors = reference_pose._get_ancestors()

    #     common_parent = None
    #     self_path = []
    #     ref_path = []

    #     for self_ancestor in self_ancestors:
    #         if self_ancestor in ref_ancestors:
    #             common_parent = self_ancestor
    #             break
    #         self_path.append(self_ancestor)

    #     for ref_ancestor in ref_ancestors:
    #         if ref_ancestor == common_parent:
    #             break
    #         ref_path.append(ref_ancestor)

    #     return common_parent, self_path, ref_path

    # def _get_ancestors(self):
    #     """
    #     Retrieves all ancestors of the current pose.
        
    #     OUTPUT:
    #         - (list): List of ancestor poses.
    #     """
    #     ancestors = []
    #     current_pose = self
    #     while current_pose is not None:
    #         ancestors.append(current_pose)
    #         current_pose = current_pose.parent
    #     return ancestors

    def draw(self, img, length=20, color=(0, 0, 255), thickness=2):
        """
        Draws an arrow representing the pose on the given image.
        INPUTS:
            img (array): The image to draw on.
            length (float): the length in pixels of the arrow representing the pose.
            color (tuple): (B,G,R) The color of the line.
            thickness (int): The thickness of the line.
        """
        position_InBaseFrame, orientation_InBaseFrame = self.relative_to_base()
        origin = position_InBaseFrame
        vec = rotate_vec([length,0],orientation_InBaseFrame)
        draw_vec(img, vec, origin, color, thickness)
    
    def draw_position(self, img, color=(255, 0, 0), thickness=1):
        """
        Draws an arrow representing the position [x,y] of the pose 
        on the given image with its parent position as origin.
        INPUTS:
            img (array): The image to draw on.
            length (float): the length in pixels of the arrow representing the pose.
            color (tuple): (B,G,R) The color of the line.
            thickness (int): The thickness of the line.
        """
        parent_pose = self.get_parent()

        if parent_pose is not None:
            draw_line(img, parent_pose.relative_to_base()[0], self.relative_to_base()[0], color, thickness)

if __name__ == "__main__":
    import random
    # Create a blank image
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    pose_graph = Graph()

    # Define poses with parent-child relationships
    base_pose = Pose([250, 250], 0, pose_graph, name="base_pose")  # Base pose at the center of the image
    branch1 = Pose([0, 0], 0, pose_graph, name="branch1")  # Child pose 1
    branch2 = Pose([-200, 0], -90, pose_graph, name="branch2")
    branch1_child = Pose([50, 0], 90, pose_graph, name="branch1_child")  # Child pose 2
    branch1.set_parent(base_pose)
    branch2.set_parent(base_pose)
    branch1_child.set_parent(branch1)

    v, w = 0, 0
    while True:
        # Update graph and pose visualization
        pose_graph.display_graph()
        
        # Clear the image
        img.fill(255)
        
        # Draw poses
        base_pose.draw(img)
        branch1.draw(img)
        branch2.draw(img)
        branch1_child.draw(img)
        
        # Draw positions
        base_pose.draw_position(img)
        branch1.draw_position(img)
        branch2.draw_position(img)
        branch1_child.draw_position(img)
        
        # Update poses
        # linear and angular accelerations
        DT = 0.5
        av = random.gauss(0, 1)
        aw = random.gauss(0, 1)
        v += av * DT
        w += aw * DT
        branch1.move(v * DT, w * DT)
        # moving in circle
        branch1_child.move(5, 12)
        branch2.move(35, 10)
        
        # Display the image
        cv2.imshow('Poses', img)
        if cv2.waitKey(0) & 0xFF == 27:  # Wait for 'ESC' key to exit
            break
    cv2.destroyAllWindows()