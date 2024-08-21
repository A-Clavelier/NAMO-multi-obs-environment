from collections import deque
import heapq
import random
import time
import numpy as np
import cv2
from Pose import Pose, compare_poses, difference
from environment import NAMOENV2D
from utils import mod


def random_color():
    # Generate random values for Blue, Green, and Red channels
    B = random.randint(0, 25)*10
    G = random.randint(0, 25)*10
    R = random.randint(0, 25)*10
    return (B, G, R)

colors = [
    (0, 165, 255),   # Orange
    (130, 0, 75),    # Indigo
    (147, 20, 255),  # Deep Pink
    (47, 255, 173),  # Green Yellow
    (208, 224, 64)   # Turquoise
]
for i in range (100):
    colors += [random_color() for _ in range(100)]

def get_neighbors(current_pose, env, path_origin):
    """Generate 8 possible moves from the current pose."""
    neighbors = []
    for lin_d in [1, 0, -1]:
        for ang_d in [1, 0, -1]:
            if lin_d == 0 and ang_d == 0:
                continue
            neighbor = Pose(current_pose.get_local_pose(), path_origin)
            neighbor.move(lin_d * env.linear_velocity, ang_d * env.angular_velocity)
            neighbors.append((lin_d, ang_d, neighbor))
    return neighbors



def bfs_novisitedset(env, end_pose, render=False):
    path_origin = Pose(env.ROB.pose.get_local_pose(),env.ORIGIN)
    start_pose = Pose([0,0,0],path_origin)
    queue = deque([(start_pose, [])])
    t=time.time()
    i=0
    log=[]
    while queue:
        current_pose, path = queue.popleft()

        if render: env.render(5,False,2,0.1), cv2.waitKey(0)
        i+=1
        dt = round(time.time()-t, 1)
        print(dt,i,len(path),len(path_origin.get_descendants())-len(queue),len(queue),len(path_origin.get_descendants()))
        log.append([dt,i,len(path),len(path_origin.get_descendants())-len(queue),len(queue),len(path_origin.get_descendants())])
       
        if compare_poses(current_pose,end_pose,distance_lim,angle_lim):
            path_origin.delete()
            return path, log
        
        for lin_d, ang_d, neighbor in get_neighbors(current_pose, env,path_origin):
                neighbor.color = colors[len(path)]
                queue.append((neighbor, path+[[lin_d,ang_d]]))                
    path_origin.delete()
    return None, log


def bfs_exactvisitedset(env, end_pose, render=False):
    path_origin = Pose(env.ROB.pose.get_local_pose(),env.ORIGIN)
    start_pose = Pose([0,0,0],path_origin)
    queue = deque([(start_pose, [])])
    visited = set(start_pose.get_local_pose())
    t=time.time()
    i=0
    log=[]
    while queue:
        current_pose, path = queue.popleft()

        if render: env.render(5,False,2,0.1), cv2.waitKey(0)
        i+=1
        dt = round(time.time()-t, 1)
        print(dt,i,len(path),len(visited),len(queue),len(path_origin.get_descendants()))
        log.append([dt,i,len(path),len(visited),len(queue),len(path_origin.get_descendants())])

        if compare_poses(current_pose,end_pose,distance_lim,angle_lim):
            path_origin.delete()
            return path, log
        
        for lin_d, ang_d, neighbor in get_neighbors(current_pose, env,path_origin):
            neighbor_local = tuple(neighbor.get_local_pose())
            # If this exact neighbor has not been visited, add it to the queue
            if neighbor_local not in visited:
                visited.add(neighbor_local)
                neighbor.color = colors[len(path)]
                queue.append((neighbor, path + [[lin_d, ang_d]])) 
            else:
                neighbor.delete()
    path_origin.delete()
    return None, log


def bfs_approxvisitedset(env, end_pose, render=False):
    path_origin = Pose(env.ROB.pose.get_local_pose(),env.ORIGIN)
    start_pose = Pose([0,0,0],path_origin)
    queue = deque([(start_pose, [])])
    visited = [start_pose]
    t=time.time()
    i=0
    log=[]
    while queue:
        current_pose, path = queue.popleft()

        if render: env.render(5,False,2,0.1), cv2.waitKey(0)
        i+=1
        dt = round(time.time()-t, 1)
        print(dt,i,len(path),len(visited),len(queue),len(path_origin.get_descendants()))
        log.append([dt,i,len(path),len(visited),len(queue),len(path_origin.get_descendants())])

        if compare_poses(current_pose,end_pose,distance_lim,angle_lim):
            path_origin.delete()
            return path, log
        
        for lin_d, ang_d, neighbor in get_neighbors(current_pose, env,path_origin):
                skip = False
                # If this neighbor is close to a visited pose, skip it
                for visited_pose in visited:
                    if compare_poses(neighbor,visited_pose,distance_lim,angle_lim):
                        neighbor.delete()
                        skip = True
                        break
                if skip:
                    continue
                visited.append(neighbor)
                neighbor.color = colors[len(path)]
                queue.append((neighbor, path + [[lin_d, ang_d]]))     
    path_origin.delete()
    return None, log


def a_star_approx(env, end_pose, render=False):
    env.ROB.save_pose()
    path_origin = Pose([0,0,0], env.ORIGIN)
    start_pose = Pose(env.ROB.pose.get_local_pose(), path_origin)
    queue = []
    visited = {start_pose:[]}
    counter=0
    heapq.heappush(queue, (0, counter, start_pose, []))
    start_heuristic_cost = goal_heuristic(start_pose, end_pose, env.linear_velocity, env.angular_velocity)
    t=time.time()
    i=0
    log=[]
    while queue:
        _, _, current_pose, path = heapq.heappop(queue)

        if render and i%100==0: env.render(5,False,2,0.1)
        i+=1
        dt = round(time.time()-t, 1)
        print(dt,i,len(path),len(visited),len(queue),len(path_origin.get_descendants()))
        log.append([dt,i,len(path),len(visited),len(queue),len(path_origin.get_descendants())])

        if compare_poses(current_pose, end_pose,distance_lim,angle_lim):
            path_origin.delete()
            env.ROB.rollback()
            return path, log

        for lin_d, ang_d, neighbor in get_neighbors(current_pose, env, path_origin):
            skip = False
            # If this neighbor has not been visited or has a longer path, add it to the queue
            for visited_pose, visited_path in visited.items():
                if len(visited_path) < len(path) and compare_poses(neighbor,visited_pose,distance_lim,angle_lim):
                        neighbor.delete()
                        skip = True
                        break
            if skip:continue
            env.ROB.pose.teleport(*neighbor.get_local_pose())
            if env._check_collision():
                neighbor.delete()
                continue
            cost = len(path)
            heuristic_cost = goal_heuristic(neighbor, end_pose, env.linear_velocity, env.angular_velocity)*10
            heuristic_cost_scale = max(1.0,(heuristic_cost/start_heuristic_cost)*5)
            heuristic_cost *= heuristic_cost_scale
            total_cost = cost + heuristic_cost
            # print(f"{cost}, {heuristic_cost:.1f}")
            visited[neighbor] = path
            counter+=1
            neighbor.color = colors[len(path)]
            heapq.heappush(queue, (total_cost, counter, neighbor, path + [[lin_d, ang_d]]))   
    path_origin.delete()
    env.ROB.rollback()
    return None, log




def bfs_guide_path(env, end_pose, render=False):
    step_length = env.linear_velocity
    path_origin = Pose([0,0,0],env.ORIGIN)
    start_pose = Pose(env.ROB.pose.get_local_pose(),path_origin)
    queue = deque([(start_pose,[])])
    visited = set(tuple(start_pose.position))
    i=1
    while queue:
        i+=1
        if render and i%50==0: env.render(5,False,2,0.1)#,  cv2.waitKey(0)
        current_pose, guide_path = queue.popleft()
        if compare_poses(current_pose,end_pose,distance_lim,360):
            path_origin.delete()
            return guide_path
        current_local = current_pose.get_local_pose()
        neighbor_poses = [Pose(np.array(current_local)+np.array([step_length,0,0]),path_origin),
                          Pose(np.array(current_local)+np.array([-step_length,0,0]),path_origin),
                          Pose(np.array(current_local)+np.array([0,step_length,0]),path_origin),
                          Pose(np.array(current_local)+np.array([0,-step_length,0]),path_origin),
                          Pose(np.array(current_local)+np.array([step_length,step_length,0]),path_origin),
                          Pose(np.array(current_local)+np.array([-step_length,-step_length,0]),path_origin),]
        for neighbor in neighbor_poses:
            neighbor_position = tuple(neighbor.position)
            if neighbor_position not in visited:
                collision_map = env.collision_map.copy()-env.ROB.get_mask()
                neighbor_imagey,  neighbor_imagex= neighbor.get_image_pose(env.RESOLUTION,env.FO_map.shape[0])[:2]
                width = int(env.ROB.imageshape[0]*env.RESOLUTION/2)
                height = int(env.ROB.imageshape[1]*env.RESOLUTION/2)
                collision_checks = [collision_map[neighbor_imagey,neighbor_imagex],
                                    collision_map[neighbor_imagey-height,neighbor_imagex],
                                    collision_map[neighbor_imagey+height,neighbor_imagex],
                                    collision_map[neighbor_imagey,neighbor_imagex-width],
                                    collision_map[neighbor_imagey,neighbor_imagex+width]]
                if all(check == 0 for check in collision_checks):
                    visited.add(neighbor_position)
                    queue.append((neighbor, guide_path + [neighbor_position]))
            else: neighbor.delete()
    path_origin.delete()
    return None


def goal_heuristic(current, goal, linear_velocity, angular_velocity):
    """Estimate the cost from the current pose to the goal pose."""
    length, angle = difference(current, goal)
    step_length = length / linear_velocity
    step_angle = abs(angle) / angular_velocity
    return (step_length + step_angle / (10 * step_length))

def guided_heuristic(current, guide_path, linear_velocity):
    min_distance = float('inf')
    current_position = tuple(current.position)
    closest_index = 0
    for i, guide_position in enumerate(guide_path):
        distance = np.sqrt((guide_position[0] - current_position[0])**2 
                           +(guide_position[1] - current_position[1])**2)
        if distance < min_distance:
            closest_index = i
            min_distance = distance
    remaining_steps = len(guide_path) - closest_index - 1 
    return remaining_steps + min_distance/linear_velocity

def a_star(env, end_pose, guide_path, render=False):
    env.ROB.save_pose()
    path_origin = Pose([0,0,0], env.ORIGIN)
    start_pose = Pose(env.ROB.pose.get_local_pose(), path_origin)
    queue = []
    visited = {tuple(start_pose.get_local_pose()):[]}
    counter=0
    heapq.heappush(queue, (0, counter, start_pose, []))
    t=time.time()
    i=1
    log=[]
    while queue:
        if i>2000: break
        _, _, current_pose, path = heapq.heappop(queue)

        if render and i%10==0: env.render(5,False,2,0.1) #, cv2.waitKey(0)
        i+=1
        dt = round(time.time()-t, 1)
        print(dt,i,len(path),len(visited),len(queue),len(path_origin.get_descendants()))
        log.append([dt,i,len(path),len(visited),len(queue),len(path_origin.get_descendants())])

        if compare_poses(current_pose, end_pose,distance_lim,angle_lim):
            print(dt,i,len(path),len(visited),len(queue),len(path_origin.get_descendants()))
            env.render(5,False,2,0.1)
            path_origin.delete()
            env.ROB.rollback()
            return path, log

        for lin_d, ang_d, neighbor in get_neighbors(current_pose, env, path_origin):
            neighbor_local = tuple(neighbor.get_local_pose())
            # If this neighbor has not been visited or has a longer path, add it to the queue
            if neighbor_local not in visited or len(visited[neighbor_local]) > len(path):
                env.ROB.pose.teleport(*neighbor_local)
                if not env._check_collision():
                    cost = len(path)
                    guided_heuristic_cost = guided_heuristic(neighbor, guide_path, env.linear_velocity)
                    guided_heuristic_cost_scaled = guided_heuristic_cost * 2
                    goal_heuristic_cost = goal_heuristic(neighbor,end_pose,env.linear_velocity,env.angular_velocity)
                    goal_heuristic_cost_scaled = goal_heuristic_cost * ((1 - guided_heuristic_cost/len(guide_path))*1.5)**2
                    proximity_heuristic_cost = sum([all([abs(neighbor_local[0]-visited_local[0])<distance_lim, 
                                                         abs(neighbor_local[1]-visited_local[1])<distance_lim,
                                                         abs(neighbor_local[2]-visited_local[2])<18]) 
                                                    for visited_local in visited])
                    print(proximity_heuristic_cost)
                    proximity_heuristic_cost_scaled = proximity_heuristic_cost * (guided_heuristic_cost/len(guide_path))
                    total_cost = cost + guided_heuristic_cost_scaled + goal_heuristic_cost_scaled + proximity_heuristic_cost_scaled
                    # print(f"{cost}, {guided_heuristic_cost_scaled:.1f}, {goal_heuristic_cost_scaled:.1f},  {proximity_heuristic_cost_scaled:.1f}  ")
                    visited[neighbor_local] = path
                    counter+=1
                    neighbor.color = colors[len(path)]
                    heapq.heappush(queue, (total_cost, counter, neighbor, path + [[lin_d, ang_d]]))
                else:
                    neighbor.delete()     
            else:
                neighbor.delete()
    path_origin.delete()
    env.ROB.rollback()
    return None, log









if __name__ == "__main__":
    import csv
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    
    distance_lim = 0.1
    angle_lim = 18

while True:
    # env = NAMOENV2D("200x200_map2","config2", distance_lim, angle_lim, 40)

    env = NAMOENV2D("200x200_empty","config1", distance_lim, angle_lim, 40, time.time())
    for _ in range(4): env.add_obstacle()
    env.render(5,False,2,0.1)
    env.add_obstacle('ROB',env.goal.get_local_pose())
    if env._check_collision():
        continue
    env.MO_list.pop()
    env._check_collision()

    env.render(5,False,2,0.1)
    cv2.waitKey(1000)

    guide_path = bfs_guide_path(env, env.goal, render=True)
    if guide_path==None:
        continue

    # env.add_obstacle('MOa', [1, 1.7, 90])

    path_origin = Pose([0,0,0],env.ORIGIN)
    start_pose = Pose(env.ROB.pose.get_local_pose(),path_origin)
    pose_list = [start_pose]
    for position in guide_path:
        pose_list.append(Pose([position[0],position[1],0],path_origin,color=(208, 224, 64)))
        for bro_pose in pose_list[:-1]:
            pose_list[-1].to_brother_frame(bro_pose)
    env.render(7,True,7,0)
    
    cv2.waitKey(1000)

    path_origin.delete()
      
    path, log=a_star(env, env.goal, guide_path, render=True)
    if path==None:
        continue

    cv2.waitKey(1000)

    path_origin = Pose([0,0,0],env.ORIGIN)
    start_pose = Pose(env.ROB.pose.get_local_pose(),path_origin)
    current_pose = start_pose
    for step in path:
        next_step = Pose(current_pose.get_local_pose(),path_origin,color=(208, 224, 64))
        next_step.move(step[0]*env.linear_velocity,step[1]*env.angular_velocity)
        current_pose=next_step
    env.render(7,False,2,0.1)

    cv2.waitKey(1000)

    path_origin.delete()

    for i in path:
        env.step(i[0],i[1],False)
        env.render(7,False,2,0.1)

# Wait for the Escape key to close the window
while True:
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()






# Define the directory and file name
directory = 'path_finder logs/'
file_name = 'log_data.csv'
file_path = os.path.join(directory, file_name)
# Ensure the directory exists
os.makedirs(directory, exist_ok=True)
# Write the log to a CSV file
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Optionally, write the header
    writer.writerow(['Time', 'Step', 'Path Length', 'Visited', 'In Queue', 'Total Poses'])
    # Write the log data
    writer.writerows(log)


# Convert log to DataFrame
columns = ['Time', 'Step', 'Path Length', 'Visited', 'In Queue', 'Total Poses']
df = pd.DataFrame(log, columns=columns)

# Summary Statistics
summary_stats = df.describe().round(1)
print("Summary Statistics:")
print(summary_stats)

# Visualize the Data
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# Time vs Step
axs[0, 0].plot(df['Time'], df['Step'])
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Step')
axs[0, 0].set_title('Time vs Step')

# Time vs Path Length
axs[0, 1].plot(df['Time'], df['Path Length'])
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Path Length')
axs[0, 1].set_title('Time vs Path Length')

# Time vs Visited
axs[1, 0].plot(df['Time'], df['Visited'])
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Visited')
axs[1, 0].set_title('Time vs Visited')

# Time vs In Queue
axs[1, 1].plot(df['Time'], df['In Queue'])
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('In Queue')
axs[1, 1].set_title('Time vs In Queue')

# Time vs Total Poses
axs[2, 0].plot(df['Time'], df['Total Poses'])
axs[2, 0].set_xlabel('Time')
axs[2, 0].set_ylabel('Total Poses')
axs[2, 0].set_title('Time vs Total Poses')

# Step vs Total Poses
axs[2, 1].plot(df['Step'], df['Total Poses'])
axs[2, 1].set_xlabel('Step')
axs[2, 1].set_ylabel('Total Poses')
axs[2, 1].set_title('Total Poses vs Step')

# Adjust the layout and spacing
fig.tight_layout(pad=3.0)
plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.90, hspace=0.6, wspace=0.3)

# Display the plot
plt.show()

# Display Detailed Data
print("Detailed Log Data:")
print(df)