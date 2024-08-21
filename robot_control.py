from environment import NAMOENV2D
from pathfinder import dijkstra_path


class NAMOtask:
    def __init__(self, env, v, w, render=True):
        self.env = env
        self.v = v
        self.w = w
        self.render = render
    
    def set_resolution(self, resolution):
        self.env.set_resolution(self, resolution)
    
    def add_obstacle(self, type=None, pose=None):
        self.env.add_obstacle(self, type, pose)

    def pick_obstacle(self, pick_pose):
        path = dijkstra_path(self.env, pick_pose, self.v, self.v)
        for move in path:
            rotation





if __name__ == "__main__":
    import time

    env = NAMOENV2D("200x200_empty","config2", 40, time.time())
    task = NAMOtask(env, 0.1, 22.5, True)
    task.pick_obstacle(env.MO_list[0].pick_poses[0])