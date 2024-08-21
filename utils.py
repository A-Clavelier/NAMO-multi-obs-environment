import numpy as np
import math
import cv2

def mod(angle):
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle

def realco_to_arrayco(co, resolution, img_height):
    # account for image resolution and inverse the y axis
    row = int(img_height - co[1]*resolution)
    col = int(co[0]*resolution)
    return  row,col

def realco_to_cv2co(co, resolution, img_height):
    y_cv2,x_cv2 = realco_to_arrayco(co, resolution, img_height)
    return  x_cv2,y_cv2

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

def draw_line(img, resolution, origin, end, color, thickness):
    """
    Draws a line between origin and end points on the given image.

        INPUTS:
            img (array): The image to draw on.
            origin (list/array): [x,y] the position of the origin point.
            end (list/array): [x,y] the position of the end point.
            color (tuple): (B,G,R) The color of the line.
            thickness (int): The thickness of the line.
    """
    img_height = img.shape[0]
    adjusted_thickness = math.ceil(thickness * resolution/600)
    orig_x, orig_y = realco_to_cv2co(origin, resolution, img_height)
    end_x, end_y = realco_to_cv2co(end, resolution, img_height)
    cv2.line(img, (int(orig_x), int(orig_y)), (int(end_x), int(end_y)), color, adjusted_thickness)

def draw_vec(img, resolution, vec, origin, color, thickness):
    """Draws an arrow representing the vector at the given origin point on the given image.

        INPUTS:
            img (array): The image to draw on.
            vec (list/array): [x,y] the vector.
            origin (list/array): [x,y] the position of the origin point.
            color (tuple): (B,G,R) The color of the line.
            thickness (int): The thickness of the line.
    """
    img_height = img.shape[0]
    adjusted_thickness = math.ceil(thickness * resolution/200)
    end = [origin[0]+vec[0], origin[1]+vec[1]]
    orig_x, orig_y = realco_to_cv2co(origin, resolution, img_height)
    end_x, end_y = realco_to_cv2co(end, resolution, img_height)
    cv2.arrowedLine(img, (int(orig_x), int(orig_y)), (int(end_x), int(end_y)), color, adjusted_thickness, tipLength=0.2)

def draw_text(img, resolution, text, origin, color, thickness):
    """Draws text at the given origin point on the given image.

        INPUTS:
            img (array): The image to draw on.
            text (string)
            origin (list/array): [x,y] the position of the text on the image.
            color (tuple): (B,G,R) The color of the line.
            thickness (int): The thickness of the line.
            font_scale (float): size of the text
    """
    img_height = img.shape[0]
    adjusted_thickness = math.ceil(thickness * resolution/400)
    font_scale = resolution/150
    orig_x, orig_y = realco_to_cv2co(origin, resolution, img_height)
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, adjusted_thickness)
    x = orig_x-text_size[0]/2
    y = orig_y+text_size[1]
    cv2.putText(img, text, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, adjusted_thickness)

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


if __name__ == "__main__":
    print(realco_to_arrayco([1,0],10,30))
    print(realco_to_cv2co([1,0],10,30))