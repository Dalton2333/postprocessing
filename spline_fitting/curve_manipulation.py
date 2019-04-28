"""
This module includes mesh subdivision and mesh smoothing
Input:  boundary points
        Original boundary points
Output: Spline
"""
import numpy as np
import copy

def curve_subdivision(points_list):
    """
    This is linear curve subdivision based on pure averaging
    :param points_list: original curve points
    :return: points_list_new: curve after subdivision
    """
    points_len = len(points_list)
    points_list_new = copy.deepcopy(points_list)
    for i in range(points_len-1):
        point_newx = (points_list[i][0]+points_list[i+1][0])/2
        point_newy = (points_list[i][1]+points_list[i+1][1])/2
        point_new = [point_newx, point_newy]
        points_list_new.insert(2*i+1, point_new)
    return points_list_new

def plot_curve_points(inputpoints):
    import matplotlib.pyplot as plt
    inputpoints = np.transpose(np.array(inputpoints))
    fig, ax = plt.subplots()
    ax.plot(inputpoints[0],inputpoints[1],'b-', lw=1)
    ax.plot(inputpoints[0], inputpoints[1], 'bo', markersize = 2)
    #ax.scatter(ts_temp[0],ts_temp[1],s=1,marker='o')
    ax.grid(True)
    plt.title('Final outside boundary')
    plt.show()


if __name__ == '__main__':
    boundary_points = [[-0.00231, 0.0091], [-0.00231, 0.006066666666666666], [-0.0017966666666666665, 0.006066666666666666], [-0.0017966666666666665, 0.0091], [-0.00077, 0.0091], [-0.00077, 0.006572222222222222], [-0.00025666666666666665, 0.006572222222222222], [-0.00025666666666666665, 0.008594444444444444], [0.00025666666666666665, 0.008594444444444444], [0.00025666666666666665, 0.0091], [0.0012833333333333334, 0.0091], [0.0012833333333333334, 0.008594444444444444], [0.0017966666666666665, 0.008594444444444444], [0.0017966666666666665, 0.008088888888888889], [0.002823333333333333, 0.008088888888888889], [0.002823333333333333, 0.008594444444444444], [0.0033366666666666666, 0.008594444444444444], [0.0033366666666666666, 0.0091], [0.0038499999999999997, 0.0091],[0.00385, -0.0091],[-0.00385, -0.0091],[-0.00385, 0.0091]]
    original_boundary_points = [[0.00385, 0.0091],[0.00385, -0.0091],[-0.00385, -0.0091],[-0.00385, 0.0091]]
    # boundary_points = [[0,0],[1,0],[2,0],[3,0],[3,1],[3,2],[3,3]]
    # boundary_points = np.array(boundary_points)
    subd1 = curve_subdivision(boundary_points)
    plot_curve_points(boundary_points)
    plot_curve_points(subd1)




