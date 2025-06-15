# utilities for graph plotting

from typing import List, Tuple
import matplotlib.pyplot as plt

def plot_points_on_plane(points: List[Tuple], numbers, color: str = 'ro', alpha: float = 1.0):
    for i, point in zip(numbers, points):
        plt.plot(point[0], point[1], color, alpha=alpha)
        plt.annotate(str(i), (point[0], point[1]), color='green', alpha=alpha)

def plot_edges_on_plane(edges: List[Tuple], alpha: float = 1.0):
    for edge in edges:
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'b-', alpha=alpha)