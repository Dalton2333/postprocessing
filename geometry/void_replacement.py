"""
This module is to perform the void replacement process
Initial design outputs the node coordinates of the features
Final design should outputs abaqus cae file directly
"""
from shapely.geometry import Polygon
import math

def replace_poly(inside_clusters,):
    circles = []
    for cluster in inside_clusters:
        poly = Polygon(cluster)
        area = poly.area
        centre = list(list(poly.centroid.coords)[0])
        radius = math.sqrt(area/math.pi)
        # print(centre)
        # quit()
        edge = [centre[0]+radius,centre[1]]
        circle = [centre,radius,edge,area]
        circles.append(circle)
    return circles

if __name__ == '__main__':
    inside_clusters = [[[0,0],[1,0],[2,0],[2,1],[2,2],[1,2],[0,2],[0,1],[0,0]]]
    circles = replace_poly(inside_clusters)
    print(circles)
