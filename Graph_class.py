import cv2
import numpy as np
import networkx as nx
from utils import split_text

class Graph:
    #add a node class to the graph instead of the dictionary structure? Is it better?
    def __init__(self):
        self.graph = {}
        self.visualization_active = False
        self.img = None
    

    def add_node(self, node):
        if self.has_node(node):
            print(f"Node {node} already exists.")
        else:
            self.graph[node] = []
            if self.visualization_active: 
                    self._update_graph_img()
                    if cv2.waitKey(1) & 0xFF == 27: self.stop_display()
    
    def add_edge(self, node1, node2):
        if not self.has_node(node1):
            print(f"Node {node1} does not exist.")
        elif not self.has_node(node2):
            print(f"Node {node2} does not exist.")
        elif self.has_edge(node1, node2):
            print(f"Edge from {node1} to {node2} already exists.")
        else:
            self.graph[node1].append(node2)
            if self.visualization_active: 
                    self._update_graph_img()
                    if cv2.waitKey(1) & 0xFF == 27: self.stop_display()

    def remove_node(self, node):
        if not self.has_node(node):
            print(f"Node {node} does not exist.")
        else:
            self.graph.pop(node)
            for children in self.graph.values():
                if node in children:
                    children.remove(node)
                    if self.visualization_active: 
                        self._update_graph_img()
                        if cv2.waitKey(1) & 0xFF == 27: self.stop_display()

    def remove_edge(self, node1, node2):
        if not self.has_edge(node1, node2):
            print(f"Edge from {node1} to {node2} does not exist.")
        else:
            self.graph[node1].remove(node2)
            if self.visualization_active: 
                    self._update_graph_img()
                    if cv2.waitKey(1) & 0xFF == 27: self.stop_display()
        
    def has_node(self, node):
        return node in self.graph
    
    def has_edge(self, node1, node2):
        return self.has_node(node1) and node2 in self.graph[node1]
    
    def get_ancestors(self, node):
        ancestors = [parent for parent, children in self.graph.items() if node in children]
        return ancestors

    def bfs_shortest_path(self, start, end):
        if start not in self.graph:
            print(f"start node {start} does not exist.")
            return []
        if end not in self.graph:
            print(f"end node {end} does not exist.")
            return []
        queue = [[start]]
        visited_nodes = set()
        if self.visualization_active: cycle_paths = []
        while queue:
            # get the first path to test in the queue
            path = queue.pop(0)
            current_node = path[len(path)-1]
            # Check if Final state reached
            if current_node == end:
                if self.visualization_active: 
                    self._update_graph_img(start, end, queue, cycle_paths, best_path=path)
                    if cv2.waitKey(0) & 0xFF == 27: self.stop_display()
                return path
            visited_nodes.add(current_node)
            for neighbor in self.graph[current_node]:
                if neighbor not in visited_nodes:
                    queue.append(path + [neighbor])
                elif self.visualization_active: cycle_paths.append(path + [neighbor])
            if self.visualization_active:
                self._update_graph_img(start, end, queue, cycle_paths)
                if cv2.waitKey(1) & 0xFF == 27: self.stop_display()

        if self.visualization_active: 
            self._update_graph_img(start, end, queue, cycle_paths)
            if cv2.waitKey(0) & 0xFF == 27: self.stop_display()
        return []  # Return empty path if no path is found

    def _update_graph_img(self, start=None, end=None, queue=[], cycle_paths=[], best_path=[]):
        # Create a networkx graph from the current graph
        G = nx.DiGraph(self.graph)
        if len(G.nodes) == 0:
            # If the graph is empty, create a blank image
            self.img = np.zeros((800, 1000, 3), dtype=np.uint8)
            self.display_graph()
            return
        try:
            # Compute the layer dictionary for hierarchical layout
            layer_dict = {}
            for i, layer in enumerate(nx.topological_generations(G)):
                for node in layer:
                    layer_dict[node] = i
                    G.nodes[node]['subset'] = i  # Add the subset attribute
            # Compute positions using multipartite layout for hierarchy
            positions = nx.multipartite_layout(G, subset_key="subset")
        except nx.NetworkXUnfeasible:
            # If the graph contains a cycle, use shell layout
            layers = {}
            for node in G:
                layer = 0
                for ancestor in nx.ancestors(G, node):
                    layer = max(layer, layers.get(ancestor, 0) + 1)
                layers[node] = layer
            layer_dict = {}
            for node, layer in layers.items():
                if layer not in layer_dict:
                    layer_dict[layer] = []
                layer_dict[layer].append(node)
            nlist = [list(layer) for layer in layer_dict.values()]
            positions = nx.shell_layout(G, nlist=nlist)
        
        # Increase padding to ensure nodes do not go out of the screen
        img_width = 1000
        img_height = 800
        padding = 150  # Increased padding

        min_x, max_x = min(pos[0] for pos in positions.values()), max(pos[0] for pos in positions.values())
        min_y, max_y = min(pos[1] for pos in positions.values()), max(pos[1] for pos in positions.values())

        scale_x = (img_width - 2 * padding) / (max_x - min_x) if max_x - min_x > 0 else 1
        scale_y = (img_height - 2 * padding) / (max_y - min_y) if max_y - min_y > 0 else 1
        scale = min(scale_x, scale_y)

        translation = np.array([img_width / 2 - (min_x + max_x) * scale / 2, img_height / 2 - (min_y + max_y) * scale / 2])
        
        for node, pos in positions.items():
            positions[node] = (np.array(pos) * scale + translation).astype(int)
        
        # Create a blank image with the calculated size
        self.img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        # Draw nodes as ovals with text splitting
        max_length = 20  # Maximum length before splitting text
        ellipse_axes_dict = {}
        for node, pos in positions.items():
            string = node.__repr__()
            text = split_text(string, max_length)
            lines = text.split('\n')
            max_line_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0] for line in lines)
            ellipse_axes = (max_line_width // 2 + 20, 30 + 15 * (len(lines) - 1))
            ellipse_axes_dict[node] = ellipse_axes
            cv2.ellipse(self.img, tuple(pos), ellipse_axes, 0, 0, 360, (255, 255, 255), -1)
            y_offset = pos[1] - (len(lines) - 1) * 10
            for line in lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.putText(self.img, line, (pos[0] - text_size[0] // 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                y_offset += 20
        
        # Draw directed edges
        for node1, edges in self.graph.items():
            for node2 in edges:
                pt1 = positions[node1]
                pt2 = positions[node2]
                # Adjust the position to make the arrow not overlap with the nodes
                direction = np.array(pt2) - np.array(pt1)
                norm_direction = direction / np.linalg.norm(direction)
                adjusted_pt1 = (int(pt1[0] + ellipse_axes_dict[node1][0] * norm_direction[0]), int(pt1[1] + ellipse_axes_dict[node1][1] * norm_direction[1]))
                adjusted_pt2 = (int(pt2[0] - ellipse_axes_dict[node2][0] * norm_direction[0]), int(pt2[1] - ellipse_axes_dict[node2][1] * norm_direction[1]))
                cv2.arrowedLine(self.img, adjusted_pt1, adjusted_pt2, (0, 0, 255), 2, tipLength=0.05)
        
        #function to highlicht nodes during path search
        def draw_step(step, color):
            node1, node2 = step
            pt1 = positions[node1]
            pt2 = positions[node2]
            direction = np.array(pt2) - np.array(pt1)
            norm_direction = direction / np.linalg.norm(direction)
            adjusted_pt1 = (int(pt1[0] + ellipse_axes_dict[node1][0] * norm_direction[0]), int(pt1[1] + ellipse_axes_dict[node1][1] * norm_direction[1]))
            adjusted_pt2 = (int(pt2[0] - ellipse_axes_dict[node2][0] * norm_direction[0]), int(pt2[1] - ellipse_axes_dict[node2][1] * norm_direction[1]))
            cv2.arrowedLine(self.img, adjusted_pt1, adjusted_pt2, color, 2, tipLength=0.05)
            cv2.ellipse(self.img, tuple(pt1), ellipse_axes_dict[node1], 0, 0, 360, color, 2)
            cv2.ellipse(self.img, tuple(pt2), ellipse_axes_dict[node2], 0, 0, 360, color, 2)
        # if path search is running it gives in argument nodes to be highlighted.
        for path in queue:
            # draw neighbor nodes in yellow
            step = path[-2:]
            draw_step(step, (0, 255, 255))
        for path in queue+cycle_paths:
            # draw visited nodes in blue
            for step_to_visited_node in zip(path[:-2], path[1:-1]):
                draw_step(step_to_visited_node, (255, 0, 0))
        # start and end in cyan and magenta
        if start is not None:
            cv2.ellipse(self.img, tuple(positions[start]), ellipse_axes_dict[start], 0, 0, 360, (255, 255, 0), 2)
        if end is not None:
            cv2.ellipse(self.img, tuple( positions[end]), ellipse_axes_dict[end], 0, 0, 360, (255, 0, 255), 2)
        # draw best path in green
        for step in zip(best_path[:-1], best_path[1:]):
            draw_step(step, (0, 255, 0))
        self.display_graph()

    def display_graph(self):
        if self.img is None:
            self._update_graph_img()
        if not self.visualization_active:
            self.visualization_active = True
            cv2.namedWindow('Graph', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Graph', 800, 100)
            cv2.resizeWindow('Graph', 500, 400)
        cv2.imshow('Graph', self.img)
        if cv2.waitKey(100) & 0xFF == 27:  # Wait for 'ESC' key to exit
            self.stop_display()
        
    def stop_display(self):
        if self.visualization_active == True:
            cv2.destroyWindow('Graph')
            self.visualization_active = False


if __name__ == "__main__":
    import random
    g = Graph()
    g.display_graph()
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O']
    # Add nodes to the graph
    for node in nodes:
        g.add_node(node)
    for _ in range(30):
        from_node = random.choice(nodes)
        to_node = random.choice(nodes)
        if from_node != to_node:
            g.add_edge(from_node, to_node)
    for _ in range(20):
        from_node = random.choice(nodes)
        to_node = random.choice(nodes)
        path = g.bfs_shortest_path(from_node, to_node)
    g.stop_display()