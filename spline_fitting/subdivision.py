"""
this is a trial on curve subdivision
"""

import numpy as np
from matplotlib import pyplot as plt


def chaikin_subd(points_list):
    """
    Qi = 3/4Pi + 1/4Pi+1 & Ri = 1/4Pi + 3/4Pi+1
    """
    new_points_list = [points_list[0]]
    for i in range(len(points_list)-1):
        new_point_1_x = points_list[i][0]*0.75 + points_list[i+1][0]*0.25
        new_point_1_y = points_list[i][1]*0.75 + points_list[i+1][1]*0.25
        new_point_2_x = points_list[i][0]*0.25 + points_list[i+1][0]*0.75
        new_point_2_y = points_list[i][1]*0.25 + points_list[i+1][1]*0.75
        new_point_1 = [new_point_1_x,new_point_1_y]
        new_point_2 = [new_point_2_x,new_point_2_y]
        # new_points_list.extend([new_point_1,new_point_2,points_list[i+1]])
        new_points_list.extend([new_point_1,new_point_2])
    return new_points_list

def average_subd(points_list):
    new_points_list = [points_list[0]]
    for i in range(len(points_list)-1):
        new_point_x = points_list[i][0]*0.5 + points_list[i+1][0]*0.5
        new_point_y = points_list[i][1]*0.5 + points_list[i+1][1]*0.5
        new_point = [new_point_x,new_point_y]
        new_points_list.extend([new_point,points_list[i+1]])
    return new_points_list

def average_smoothing(points_list, averaging_ratio=0.5,per=False):
    """
    For point (x1,y1), (x2,y2), (x3,y3)

    Shit, there is a easier solution:
    the mid point of p1 and p3 is:
    ((x1+x3)/2,(y1+y3)/2)
    the point p2 moves towards the mid point with give ratio:
    (x2,y2)+[((x1+x3)/2,(y1+y3)/2)-(x2,y2)]*ratio
    --> ((x2+(x1/2+x3/2-x2)*ratio), (y2+(y1/2+y3/2-y2)*ratio)
    """
    new_points_list = [points_list[0]]
    for i in range(1,len(points_list)-1):
        p1 = points_list[i-1]
        p2 = points_list[i]
        p3 = points_list[i+1]
        p2n_x = p2[0]+(p1[0]/2+p3[0]/2-p2[0])*averaging_ratio
        p2n_y = p2[1]+(p1[1]/2+p3[1]/2-p2[1])*averaging_ratio
        p2n = [p2n_x,p2n_y]
        new_points_list.append(p2n)
    new_points_list.append(points_list[-1])
    return new_points_list

def distance(pointa, pointb):
    pointa = np.array(pointa)
    pointb = np.array(pointb)
    dst = np.linalg.norm(pointa-pointb)
    return dst
        
def plot_points(points_list, title, closed_curve=False, rotation=None):
    points_list.append(points_list[0])
    tr_both = list(np.transpose(points_list))
    plt.scatter(tr_both[0],tr_both[1],c='r', marker='.',s=4)
    line, = plt.plot(tr_both[0], tr_both[1], 'black', lw=1)
    plt.title(title)
    plt.show()

class plot_boundary():
    pass

def split_boundary(full_points_list, keep_points_list):
    keep_index = []
    for point in keep_points_list:
        keep_index.append(full_points_list.index(point))
    sorted(keep_index)
    keep_index_slice = []
    for index in keep_index:
        if index-1 in keep_index:
            pass
        else:
            start = index
        if index+1 in keep_index:
            pass
        else:
            end = index
            keep_index_slice.append([start,end])
    return keep_index_slice

def curve_smoothing(all_points, keep_index_slice, smoothing_method="average_smoothing"):
    new_boundary = []
    if smoothing_method == "average_smoothing":
        if keep_index_slice[0][0] == 0:
            pass
        else:
            smoothing = average_smoothing(all_points[0:keep_index_slice[0][0]+1])
            smoothing = smoothing[0:-1]
            new_boundary.extend(smoothing)
        for i in range(len(keep_index_slice)):
            new_boundary.extend(all_points[keep_index_slice[i][0]:keep_index_slice[i][1]+1])
            if i != len(keep_index_slice)-1:
                smoothing = average_smoothing(all_points[keep_index_slice[i][1]:keep_index_slice[i+1][0]+1])
                smoothing = smoothing[1:-1]
                new_boundary.extend(smoothing)
            else:
                if len(all_points)-1 == keep_index_slice[i][1]:
                    pass
                else:
                    smoothing = average_smoothing(all_points[keep_index_slice[i][1]:len(all_points)])
                    smoothing = smoothing[1:-1]
                    new_boundary.extend(smoothing)
    else:
        pass
    return new_boundary

def curve_subd(all_points, keep_index_slice, subd_method="chaikin_subd"):
    new_boundary = []
    if subd_method == "chaikin_subd":
        if keep_index_slice[0][0] == 0:
            pass
        else:
            subdivision = chaikin_subd(all_points[0:keep_index_slice[0][0]+1])
            subdivision = subdivision[0:-1]
            new_boundary.extend(subdivision)
        for i in range(len(keep_index_slice)):
            new_boundary.extend(all_points[keep_index_slice[i][0]:keep_index_slice[i][1]+1])
            if i != len(keep_index_slice)-1:
                subdivision = chaikin_subd(all_points[keep_index_slice[i][1]:keep_index_slice[i+1][0]+1])
                subdivision = subdivision[1:-1]
                new_boundary.extend(subdivision)
            else:
                if len(all_points)-1 == keep_index_slice[i][1]:
                    pass
                else:
                    subdivision = chaikin_subd(all_points[keep_index_slice[i][1]:len(all_points)])
                    subdivision = subdivision[1:-1]
                    new_boundary.extend(subdivision)
    else:
        pass
    return new_boundary

def spline_fitting(curve_points, smoothness,per):
    """
    :param curve_points:
    :param smoothness:
    :return:    x_new:np array for x values
                y_new: np array for y values
    """
    from scipy.interpolate import splprep, splev
    all_points = np.array(curve_points,dtype=np.float64)
    length = len(all_points)
    # all_points = np.dot(all_points, 1000)

    # smoothness = 10
    tck, u = splprep(all_points.T, u=None, s=smoothness, per=per)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    # plt.plot(x_new, y_new, 'black', lw=1)
    # plt.show()
    return x_new, y_new


def curve_spline_fitting(all_points, keep_index_slice, smoothness,per):

    # all_points = np.dot(all_points,1000)
    x_new = np.array([])
    y_new = np.array([])
    # if subd_method == "chaikin_subd":
    if keep_index_slice[0][0] == 0:
        pass
    else:
        splinex, spliney = spline_fitting(all_points[0:keep_index_slice[0][0]+1],smoothness,per)
        x_new=np.append(x_new,splinex[1:-1])
        y_new=np.append(y_new,spliney[1:-1])
    for i in range(len(keep_index_slice)):
        line = all_points[keep_index_slice[i][0]:keep_index_slice[i][1]+1]
        line = np.array(line)
        line = np.transpose(line)
        # line = np.array(all_points[keep_index_slice[i][0]:keep_index_slice[i][1]+1]).T
        x_new=np.append(x_new,line[0])
        y_new=np.append(y_new,line[1])

        if i != len(keep_index_slice)-1:
            splinex, spliney = spline_fitting(all_points[keep_index_slice[i][1]:keep_index_slice[i+1][0]+1],smoothness,per)
            x_new=np.append(x_new,splinex[1:-1])
            y_new=np.append(y_new,spliney[1:-1])
        else:
            if len(all_points)-1 == keep_index_slice[i][1]:
                pass
            else:
                splinex, spliney = spline_fitting(all_points[keep_index_slice[i][1]:len(all_points)],smoothness,per)
                x_new=np.append(x_new,splinex[1:-1])
                y_new=np.append(y_new,spliney[1:-1])
    x_new=np.append(x_new,all_points[0][0])
    y_new=np.append(y_new,all_points[0][1])

    # This is how the curve get plotted:
    plt.plot(x_new, y_new, 'black', lw=1)
    plt.show()
    return x_new,y_new


    # else:
    #     pass
    # return new_boundary

def get_ext_boundary(boundary_points, original_boundary_points, subd=0, smoothness=1,per=0):
    keep_index = split_boundary(boundary_points, original_boundary_points)
    smoothed_boundary = curve_smoothing(boundary_points, keep_index)

    # subdivision = curve_subd(smoothed_boundary, keep_index)
    # subd_index = split_boundary(subdivision, original_boundary_points)

    x_new,y_new = curve_spline_fitting(smoothed_boundary,keep_index, smoothness,per)
    # boundary_new = np.array([x_new,y_new]).transpose()
    # boundary_new = np.stack(boundary_new)
    return x_new,y_new

def get_int_boundary(cluster_points,subd=0,smoothness=1,per=1):
    # keep_index = split_boundary(boundary_points, original_boundary_points)
    smoothed_boundary = curve_smoothing(boundary_points, keep_index)

    # subdivision = curve_subd(smoothed_boundary, keep_index)
    # subd_index = split_boundary(subdivision, original_boundary_points)

    x_new,y_new = curve_spline_fitting(smoothed_boundary,keep_index, smoothness,per)
    # boundary_new = np.array([x_new,y_new]).transpose()
    # boundary_new = np.stack(boundary_new)
    return x_new,y_new

if __name__ == "__main__":

    boundary_points = [[-0.003, 0.0091],[-0.00231, 0.0091], [-0.00231, 0.006066666666666666], [-0.0017966666666666665, 0.006066666666666666], [-0.0017966666666666665, 0.0091], [-0.00077, 0.0091], [-0.00077, 0.006572222222222222], [-0.00025666666666666665, 0.006572222222222222], [-0.00025666666666666665, 0.008594444444444444], [0.00025666666666666665, 0.008594444444444444], [0.00025666666666666665, 0.0091], [0.0012833333333333334, 0.0091], [0.0012833333333333334, 0.008594444444444444], [0.0017966666666666665, 0.008594444444444444], [0.0017966666666666665, 0.008088888888888889], [0.002823333333333333, 0.008088888888888889], [0.002823333333333333, 0.008594444444444444], [0.0033366666666666666, 0.008594444444444444], [0.0033366666666666666, 0.0091], [0.0038499999999999997, 0.0091],[0.00385, -0.0091],[-0.00385, -0.0091], [-0.00385, 0.0091]]

    test_points = [[0,0],[1,0],[2,0],[3,0],[3,1],[3,2],[3,3],[2,3],[1,3],[0,3],[0,0]]
    original_boundary_points = [[0.0038499999999999997, 0.0091],[0.00385, -0.0091],[-0.00385, -0.0091],[-0.00385, 0.0091]]
    keep_index = split_boundary(boundary_points, original_boundary_points)
    smoothed_boundary = curve_smoothing(boundary_points, keep_index)
    # print(smoothing1)
    # plot_points(boundary_points, "Original")
    # plot_points(smoothed_boundary, "smoothed boundary")
    subdivision = curve_subd(smoothed_boundary, keep_index)
    # print(subdivision)
    # plot_points(subdivision,"subdivision")
    subd_index = split_boundary(subdivision, original_boundary_points)
    x_new,y_new = curve_spline_fitting(subdivision,subd_index)
    # print(x_new,y_new)
    boundary_new = np.array([x_new,y_new]).transpose()
    boundary_new = np.stack(boundary_new)
    print(boundary_new)


    # plot_points(spline,"all fitting")
    # spline_curve = spline_fitting_v1(subdivision, subd_index)

