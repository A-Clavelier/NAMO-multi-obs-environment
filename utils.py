import numpy as np
import cv2

def mod(angle):
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle

def rotate_vec(vec, tetha):
    """Rotate a 2D vector by an angle theta using the rotation matrix.

    INPUT:
        - vec (list/array): [x, y] coordinates of the vector
        - tetha (float): angle of rotation in degrees

    OUTPUT:
        - (list): [rotated_x, rotated_y] coordinates of the rotated vector
    """
    x,y = vec
    # Convert the angle to radians
    tetha_rad = np.radians(tetha)
    # Apply the rotation matrix
    rotated_x = x * np.cos(tetha_rad) - y * np.sin(tetha_rad)
    rotated_y = x * np.sin(tetha_rad) + y * np.cos(tetha_rad)
    return np.array([rotated_x,rotated_y],dtype=np.float32)

def pose_to_parent_frame(position, orientation, parent_position, parent_orientation):
    """
     Change the frame of reference for a 2D pose.

    INPUT:
        - position (array): [x, y] coordinates of the position in the original frame
        - orientation (float): angle of orientation in the original frame in degrees
        - origin_position (array): [x, y] coordinates of the original frame's origin in the new frame
        - origin_orientation (float): angle of orientation of the original frame in the new frame in degrees

    OUTPUT:
        - (array, float): ([new_x, new_y], new_orientation) coordinates of the position and orientation in the new frame
    """
    # Adjust the angle by adding the rotation of the origin_pose
    new_orientation = mod(orientation + parent_orientation)
    # rotate the pose coordinates by the angle of the origin_pose to align with the new frame.
    # and translate of the origin_position to account for new frame's origin.
    new_position = parent_position + rotate_vec(position, parent_orientation)
    return [new_position,new_orientation]

def pose_to_brother_frame(position, orientation, brother_position, brother_orientation):
    """
     inverse of the pose_to_parent_frame function
    """
    new_orientation = mod(orientation - brother_orientation)
    new_position = rotate_vec(brother_position - position, 180-brother_orientation)
    return [new_position,new_orientation]

def draw_line(img, origin, end, color=(255, 0, 0), thickness=1):
    """
    Draws a line between origin and end points on the given image.

        INPUTS:
            img (array): The image to draw on.
            origin (list/array): [x,y] the position of the origin point.
            end (list/array): [x,y] the position of the end point.
            color (tuple): (B,G,R) The color of the line.
            thickness (int): The thickness of the line.
    """
    orig_x, orig_y = origin
    end_x, end_y = end
    cv2.line(img, (int(orig_x), int(orig_y)), (int(end_x), int(end_y)), color, thickness)

def draw_vec(img, vec, origin, color=(0, 0, 255), thickness=1):
    """Draws an arrow representing the vector at the given origin point on the given image.

        INPUTS:
            img (array): The image to draw on.
            vec (list/array): [x,y] the vector.
            origin (list/array): [x,y] the position of the origin point.
            color (tuple): (B,G,R) The color of the line.
            thickness (int): The thickness of the line.
    """
    x, y = vec
    orig_x, orig_y = origin
    end_x = orig_x + x
    end_y = orig_y - y
    cv2.arrowedLine(img, (int(orig_x), int(orig_y)), (int(end_x), int(end_y)), color, thickness, tipLength=0.2)

def draw_text(img, text, origin, color=(0, 0, 255), thickness=1, font_scale=1):
    """Draws text at the given origin point on the given image.

        INPUTS:
            img (array): The image to draw on.
            text (string)
            origin (list/array): [x,y] the position of the text on the image.
            color (tuple): (B,G,R) The color of the line.
            thickness (int): The thickness of the line.
            font_scale (float): size of the text
    """
    orig_x, orig_y = origin
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = int(orig_x-text_size[0]/2)
    y = int(orig_y+text_size[1])
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def crop_region(global_map, center_ImgPos, ImgShape, padding_value=0):
    """
    crop a patch from a global map array.

    INPUT:
        center_pos_img (tuple of ints): (row, col) position of the center of 
                                        the cropped image in the global map
        global_map (np.uint8): the map we want to crop (e.g. path_map, visibility_map...)
        shape (tuple of ints): (row, col) the shape of the cropped region output.
        padding_value (int): the value to use for padding (default is 0).

    OUTPUT:
        cropped_map (np.uint8): cropped region
    """
    # Add padding to the boundaries of the global_map to prevent cropping errors at the edges
    # Create a new array filled with the padding_value, with the shape of global_map,
    # enlarged by the dimensions specified in shape input 
    pad_row = ImgShape[0]
    pad_col = ImgShape[1]
    padded_global_map = np.full((global_map.shape[0] + pad_row, 
                                 global_map.shape[1] + pad_col), 
                                 padding_value, dtype=np.uint8)
    # Copy the original global_map into the center of the new padded_global_map array. 
    # That way, the padding surrounds the original map.
    padded_global_map[ImgShape[0]//2:-ImgShape[0]//2, 
                      ImgShape[1]//2:-ImgShape[1]//2] = global_map
    # Set the coordinates of the center of the crop region (taking the padding into account)
    center_row = center_ImgPos[0] + pad_row//2
    center_col = center_ImgPos[1] + pad_col//2
    # Set the coordinates of the start and end of the crop region (shape//2 away from the center)
    start_row = int(center_row - ImgShape[0]//2)
    end_row = int(center_row + ImgShape[0]//2)
    start_col = int(center_col - ImgShape[1]//2)
    end_col = int(center_col + ImgShape[1]//2)
    # Crop the padded global map into the desired region
    cropped_map = padded_global_map[start_row:end_row, start_col:end_col]
    return cropped_map

def convert_pos_to_ImgPos(pos, origin, resolution):
    """
    the pos is the coordinate in [m] from the origin position in the map array.
    the ImgPos corresponds to the index in the image array, its unit is [pxl]
    
    INPUT:
        pos (np.float32): (x, y)            #unit:[m,m]
        origin (list): [row,col] origin point of the real coordinates axes.
        resolution (float): Resolution in pixels per meter.

    OUTPUT:
        ImgPos (np.int32): (row, col)           #unit:[pxl,pxl]
    """
    #### ROW IMG COORDINATES ####################################################
    # -get the RealPos y coordinates 
    #   (row corresponds to y in real coordinates)
    # -convert it from m to pxl unit 
    # -invert the y axis by multiplicating by -1 
    #   (because rows are inverted (top to bottom)  )
    #   (compared to the real y axis (bottom to top))
    # -add the row offset of the origin point
    img_row = int(np.round(origin[0] - pos[1]*resolution))
    #unit verification:[pxl]=[pxl]-[m]*[pxl/m]
    #### COL IMG COORDINATES ####################################################
    # -get the RealPos x coordinates
    #   (col corresponds to x in real coordinates)
    # -convert it from m to pxl unit 
    # -don't invert the axis.
    #   (columns and real_x both go from left to right)
    # -add the col offset of the origin point
    img_col = int(np.round(origin[1] + pos[0]*resolution))
    #unit verification:[pxl]=[pxl]+[m]*[pxl/m]
    #############################################################################
    # put the row, col in a numpy array
    ImgPos = np.array([img_row, img_col],dtype=np.int32)
    return ImgPos

def convert_ImgPos_to_pos(ImgPos, origin, resolution):
    """    
    the ImgPos corresponds to the index in the image array, its unit is [pxl]
    the pos is the coordinate in [m] from the origin position in the map array.

    INPUT:
        ImgPos (np.int32): (row, col)           #unit:[pxl,pxl]
        origin (list): [row,col] origin point of the real coordinates axes.
        resolution (float): Resolution in pixels per meter.

    OUTPUT:
        pos (np.float32): (x, y)            #unit:[m,m]
    """
    #### X REAL COORDINATES ####################################################
    # -get the ImgPos col 
    #   (x corresponds to col in image coordinates)
    # -subtract the col offset of the origin point
    # -convert it from m to pxl unit 
    # -don't invert the axis.
    real_x = (ImgPos[1]-origin[1])/resolution  
    #unit:[m]=[pxl]-[pxl]/[pxl/m]
    #### Y REAL COORDINATES ####################################################
    # -get the ImgPos row 
    #   (y corresponds to row in image coordinates)
    # -subtract the row offset of the origin point
    # -convert it from m to pxl unit 
    # -invert the axis by multiplicating by -1
    real_y=-(ImgPos[0]-origin[0])/resolution
    #unit:[m]=-([pxl]-[pxl])/[pxl/m]
    #############################################################################
    # put the x, y in a numpy array
    pos = np.array([real_x, real_y], dtype=np.float32)
    return pos

def split_text(text, max_length):
    # Split the text into multiple lines if it exceeds the maximum length
    lines = []
    while len(text) > max_length:
        split_index = text[:max_length + 1].rfind(' ')
        if split_index == -1:
            split_index = max_length
        lines.append(text[:split_index].strip())
        text = text[split_index:].strip()
    lines.append(text)
    return '\n'.join(lines)


if __name__ == "__main__":
    print(rotate_vec([1,0],90))