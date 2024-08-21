import cv2
from Pose import Pose, compare_poses
import heapq


def get_neighbor_poses(current_pose, path_origin, step_length):
    """Generate 8 possible moves from the current pose with associated costs."""
    neighbors = []
    directions = [
        (step_length, 0, 0, step_length),   # Right
        (-step_length, 0, 0, step_length),  # Left
        (0, step_length, 0, step_length),   # Up
        (0, -step_length, 0, step_length),  # Down
        (step_length, step_length, 0, step_length*1.414),    # Up-Right
        (step_length, -step_length, 0, step_length*1.414),   # Down-Right
        (-step_length, step_length, 0, step_length*1.414),   # Up-Left
        (-step_length, -step_length, 0, step_length*1.414)  # Down-Left
    ]
    current_local = current_pose.get_pose_local()
    for dx, dy, da, cost in directions:
        neighbor_pose = Pose([current_local[0]+dx,current_local[1]+dy,current_local[2]+da], path_origin, color=(208, 224, 64))
        neighbors.append((dx, dy, neighbor_pose, cost))
    return neighbors

def dijkstra_path(env, end_pose, step_length, distance_lim, render=False):
    env.ROB.save_pose()
    path_origin = Pose([0, 0, 0], env.ORIGIN)
    start_pose = Pose(env.ROB.pose.get_pose_local(), path_origin, color=(208, 224, 64))
    # Priority queue for Dijkstra, with the cost as the first element
    counter=0
    priority_queue = [(0, counter, start_pose, [])]
    visited = {(round(start_pose.x/distance_lim)*distance_lim,round(start_pose.y/distance_lim)*distance_lim):0}
    i = 1    
    while priority_queue:
        i += 1
        if render and i % 100 == 0:
            env.render(5, False, 2, 0.1)

        current_cost, _, current_pose, current_path = heapq.heappop(priority_queue)
        
        if compare_poses(current_pose, end_pose, distance_lim, 360):
            path_origin.delete()
            env.ROB.rollback()
            return current_path
        
        for dx, dy, neighbor, neighbor_cost in get_neighbor_poses(current_pose, path_origin, step_length):
            neighbor_position = (round(neighbor.x/distance_lim)*distance_lim,round(neighbor.y/distance_lim)*distance_lim)
            cost = current_cost + neighbor_cost
            #check if position is already visited
            if neighbor_position in visited and visited[neighbor_position] <= cost:
                neighbor.delete()
                continue
            # check if position is in collision
            env.ROB.pose.relocate(neighbor.get_pose_local())
            if env._check_collision():
                neighbor.delete()
                continue
            # add the neighbor
            visited[neighbor_position]= cost
            new_path = current_path.copy()
            if dx != 0: new_path.append([dx, 0])
            if dy != 0: new_path.append([0, dy])
            counter+=1
            heapq.heappush(priority_queue, (cost, counter, neighbor, new_path))   

    path_origin.delete()
    env.ROB.rollback()
    return None


def render_path(path):
        path_origin = Pose([0,0,0],env.ORIGIN)
        start_pose = Pose(env.ROB.pose.get_pose_local(),path_origin)
        pose_list = [start_pose]
        for i in path:
            pose_list.append(Pose([i[0],i[1],0], pose_list[-1],color=(208, 224, 64)))
        env.render(7,True,7,0)
        path_origin.delete()


if __name__ == "__main__":
    from environment import NAMOENV2D
    import time

    step_length = 0.1

    while True:
        env = NAMOENV2D("200x200_empty","config1", 40, time.time())
        for _ in range(4): env.add_obstacle()
        env.render(5,False,2,0.1)
        env.add_obstacle('ROB',env.goal.get_pose_local())
        if env._check_collision():
            continue
        env.MO_list.pop()
        env._check_collision()

        env.render(5,False,2,0.1)
        cv2.waitKey(1000)

        path = dijkstra_path(env, env.goal, step_length, step_length, render=True)
        if path==None:
            continue
        render_path(path)
        cv2.waitKey(1000)

        # env.add_obstacle('MOa', [1, 1.7, 90])
        for i in path:
            env.step(i[0],i[1],0,False)
            env.render(5,False,2,0.1)
        cv2.waitKey(1000)