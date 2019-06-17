'''
11/12/2018
Author@Dedao
This is the main control function for post processing
'''
import os
import sys

import geometry.boundary_extractor
import matplotlib.pyplot as plt
import numpy
import spline_fitting.subdivision as ss
import utilities.abaqus.inp_tree_processor_v2
import utilities.abaqus.inp_reader_v2
import utilities.logger as lgr
from scipy.interpolate import splprep, splev


def get_desired_part(parts_list,ap):
	part_name_requested = ap['part_name']
	for part in parts_list:
		if part.label == part_name_requested:
			return part
	# This is reached if the requested part was never found.
	raise Exception("Could not find the part: " + part_name_requested +
				" as requested in the input file.")

def insert_points(start_coord, end_coord, no_of_insertion):
    points = []
    for point_index in range(no_of_insertion):
        point_coord = ((start_coord[0]+end_coord[0])*point_index/no_of_insertion,
                       (start_coord[1]+end_coord[1])*point_index/no_of_insertion)
        points.append(point_coord)
    return points

def extend_edge(edge_point0, edge_point1, no_of_insertion):
    points = []
    for point_index in range(no_of_insertion):
        point_coord = ((edge_point0[0]+edge_point1[0])*point_index/no_of_insertion/10+(edge_point0[0]-edge_point1[0]),
                       (edge_point0[1]+edge_point1[1])*point_index/no_of_insertion/10+(edge_point0[1]-edge_point1[1]))
        points.append(point_coord)
    return points

def get_final_boundary(outside_boundary_coords, original_outside_nodes_dict, normal):
    # (outside, s_to_e,new_outside):
    '''The final boundary consists of many points
    input:
    output: list of coords of points [(x,y,z)]
    '''
    # connecting smooth part with part of original boundary
    #boundary_piece_type = None
    normal_abs = numpy.absolute(normal)
    if normal_abs[0] == max(normal_abs):
        outside = [(b, c) for a, b, c in outside_boundary_coords]
    elif normal_abs[1] == max(normal_abs):
        outside = [(a, c) for a, b, c in outside_boundary_coords]
    elif normal_abs[2] == max(normal_abs):
        outside = [(a, b) for a, b, c in outside_boundary_coords]
    else:
        raise Exception("Max of normal not found: ",normal_abs)
    if original_outside_nodes_dict == None:
        s_to_e = [(0, len(outside_boundary_coords)-1)]
    else:
        s_to_e = []
        # print(sorted(original_outside_nodes_dict.keys()))
        #for checking_index in range(sorted(original_outside_nodes_dict.keys())[-1]+1):
        check_index = 0
        check_in_list = True
        while check_in_list:
            if check_index in original_outside_nodes_dict.keys():
                start_of_piece = check_index
                check_in_piece = True
                while check_in_piece:
                    check_index += 1
                    # print(check_index)
                    if not check_index in original_outside_nodes_dict.keys():
                        check_in_piece = False
                        end_of_piece = check_index - 1
                        s_to_e.append((start_of_piece, end_of_piece))
                    if check_index >= sorted(original_outside_nodes_dict.keys())[-1]:
                        check_in_list = False
            else:
                check_in_piece = False
                while not check_in_piece:
                    check_index += 1
                    print(check_index)
                    if check_index in original_outside_nodes_dict.keys():
                        check_in_piece = True
                    if check_index >= sorted(original_outside_nodes_dict.keys())[-1]:
                        check_in_list = False
        #checking_index = sorted(original_outside_nodes_dict.keys())[0]
        #for node_index in sorted(original_outside_nodes_dict.keys()):
        #    if node_index == checking_index:
        #        first_checking_index = checking_index
        #        checking_index += 1
        #    else:
        #        s_to_e.append((first_checking_index, checking_index-1))
    print("the keep list:")
    print(s_to_e)
    new_outside = []

    full_indexes = s_to_e.copy()
    if s_to_e[0][0] == 0:
        pass
    # else:
    #     full_indexes.append((0, s_to_e[0][0]-1))
    # for tuple_index in range(len(s_to_e)):
    #     if tuple_index < len(s_to_e) - 1:
    #         full_indexes.append((s_to_e[tuple_index][1]+1, s_to_e[tuple_index + 1][0]-1))
    #     elif tuple_index == len(s_to_e) - 1:
    #         if s_to_e[tuple_index][1] < len(outside_boundary_coords) - 1:
    #             full_indexes.append((s_to_e[tuple_index][1]+1, len(outside_boundary_coords)))
    else:
        full_indexes.append((0, s_to_e[0][0]))
    for tuple_index in range(len(s_to_e)):
        if tuple_index < len(s_to_e) -1:
            full_indexes.append((s_to_e[tuple_index][1], s_to_e[tuple_index+1][0]))
        elif tuple_index == len(s_to_e) - 1:
            if s_to_e[tuple_index][1] < len(outside_boundary_coords)-1:
                full_indexes.append((s_to_e[tuple_index][1], len(outside_boundary_coords)))
            elif s_to_e[tuple_index][1] == len(outside_boundary_coords)-1:
                pass
            else:
                raise Exception("The index exceeds the len of outside_boundary_coords: ", len(outside_boundary_coords))
        else:
            raise Exception("The index exceeds the len of list: ",s_to_e)
    full_indexes.sort()
    print("the full index is:", full_indexes)

    if 0 in original_outside_nodes_dict.keys():
        # start of s_to_e is original coords
        print("# start of outside coords is original coords")
        for i in range(len(full_indexes)):
            if i % 2 == 0:
                # this piece is original nodes
                if i == len(full_indexes) - 1:
                    end = full_indexes[i][1]
                else:
                    end = full_indexes[i][1] + 1
                for index in range(full_indexes[i][0], end):
                    new_outside.append(outside[index])
            else:
                # this piece is spline
                piece_coords = []
                if i == len(full_indexes) - 1:
                    end = full_indexes[i][1]
                else:
                    end = full_indexes[i][1] + 1
                for index in range(full_indexes[i][0], end):
                    piece_coords.append(outside[index])
                piece_coords = ss.average_smoothing(piece_coords)
                piece_coords = ss.average_smoothing(piece_coords)
                piece_coords = numpy.array(piece_coords)
                tck, u = splprep(piece_coords.T, s=1, per=0)
                u_new = numpy.linspace(u.min(), u.max(), 200)
                x_new, y_new = splev(u_new, tck, der=0)
                for coord_index in range(len(x_new)):
                    new_outside.append((x_new[coord_index], y_new[coord_index]))
    else:
        # start of s_to_e is spline
        print("start of s_to_e is spline")
        for i in range(len(full_indexes)):
            if i % 2 > 0:
                # this piece is original nodes
                for index in range(full_indexes[i][0], full_indexes[i][1]+1):
                    new_outside.append(outside[index])
            else:
                # this piece is spline
                piece_coords = []

                for index in range(full_indexes[i][0], full_indexes[i][1]+1):
                    piece_coords.append(outside[index])
                    # try:
                    #     piece_coords.append(outside[index])
                    # except:
                    #     print("")

                piece_coords = ss.average_smoothing(piece_coords)
                piece_coords = numpy.array(piece_coords)
                tck, u = splprep(piece_coords.T, u=None, s=1, per=0)
                u_new = numpy.linspace(u.min(), u.max(), 200)
                x_new, y_new = splev(u_new, tck, der=0)
                for coord_index in range(len(x_new)):
                    new_outside.append((x_new[coord_index], y_new[coord_index]))
    new_outside.append(outside[0])
    print("The outside nodes are: ",new_outside)
    return new_outside

def get_final_boundary_long_cant(full_indexes, outside_boundary_coords, original_outside_nodes_dict):
    new_outside = []
    outside = [(c, b) for a, b, c in outside_boundary_coords]
    if 0 in original_outside_nodes_dict.keys():
        # start of s_to_e is original coords
        print("# start of outside coords is original coords")
        # for original_nodes_index in s_to_e
        for i in range(len(full_indexes)):
            if i % 2 == 0:
                # this piece is original nodes
                for index in range(full_indexes[i][0], full_indexes[i][1]+1):
                    new_outside.append(outside[index])
            else:
                # this piece is spline
                piece_coords = []
                # edge0 =outside[full_indexes[i-1][1]]
                # edge1 =outside[full_indexes[i-1][1]-1]
                # start_extends = extend_edge(edge0, edge1, 3)
                # for point in start_extends:
                #     piece_coords.append(point)
                for index in range(full_indexes[i][0], full_indexes[i][1]+1):
                    piece_coords.append(outside[index])
                # if i < len(full_indexes) -1:
                #     edge0 = outside[full_indexes[i+1][0]]
                #     edge1 = outside[full_indexes[i+1][0]+1]
                #     end_extends = extend_edge(edge0, edge1, 3)
                #     for point in end_extends[::-1]:
                #         piece_coords.append(point)
                piece_coords = numpy.array(piece_coords)
                tck, u = splprep(piece_coords.T, s=0.7)
                u_new = numpy.linspace(u.min(), u.max(), 100)
                x_new, y_new = splev(u_new, tck, der=0)
                print("Spline related variables: tck, u:",tck, u)
                for coord_index in range(len(x_new)):
                    new_outside.append((x_new[coord_index], y_new[coord_index]))
    else:
        # start of s_to_e is spline
        print("start of s_to_e is spline")
        for i in range(len(full_indexes)):
            if i % 2 > 0:
                # this piece is original nodes
                for index in range(full_indexes[i][0], full_indexes[i][1]+1):
                    new_outside.append(outside[index])
            else:
                # this piece is spline
                piece_coords = []
                for index in range(full_indexes[i][0], full_indexes[i][1]+1):
                    piece_coords.append(outside[index])
                piece_coords = numpy.array(piece_coords)
                tck, u = splprep(piece_coords.T, u=None, s=0.7)
                u_new = numpy.linspace(u.min(), u.max(), 1000)
                x_new, y_new = splev(u_new, tck, der=0)
                for coord_index in range(len(x_new)):
                    new_outside.append((x_new[coord_index], y_new[coord_index]))
    # print("The outside nodes are: ",new_outside)
    return new_outside


def spline_fitting_scp():
    outside_plot = []
    for node in outside_boundary.values():
        outside_plot.append(node.coordinates)
    final_outside_boundary_coords = get_final_boundary(outside_plot, original_outside_boundary, normal)

    return final_outside_boundary_coords

def spline_fitting_v2():
    fig, ax = plt.subplots()
    for cluster in inside_boundary:
        cluster_nodes = []
        for node in cluster.values():
            if normal_abs[0] == max(normal_abs):
                cluster_nodes.append(node.coordinates[1:3:1])
            elif normal_abs[1] == max(normal_abs):
                cluster_nodes.append(node.coordinates[0:3:2])
            elif normal_abs[2] == max(normal_abs):
                cluster_nodes.append(node.coordinates[0:2:1])
            else:
                raise Exception("Max of normal is not found: ",normal_abs)
        # m = len(cluster_nodes)
        # import math
        # m = m - math.sqrt(2*m)
        cluster_nodes = numpy.array(cluster_nodes)
        print("The inside_boundary are:")
        if len(cluster_nodes) < 50:
            smoothing = 5
        elif 100 <= len(cluster_nodes) < 200:
            smoothing = 40
        elif 200 <= len(cluster_nodes) < 300:
            smoothing = 40
        else:
            smoothing = 80
        tck, u = splprep(cluster_nodes.T, u=None, s=smoothing, per=1)
        u_new = numpy.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)

        # plt.plot(cluster_nodes[:,0], cluster_nodes[:,1], 'ro')
        line, = plt.plot(x_new, y_new, 'black', lw=1.5)
        # plt.plot(x_new, y_new, 'b--', lw=1)

    # ------------- Plotting outside boundary
    outside_plot = []
    for node in outside_boundary.values():
        if normal_abs[0] == max(normal_abs):
            outside_plot.append(node.coordinates[1:3:1])
        elif normal_abs[1] == max(normal_abs):
            outside_plot.append(node.coordinates[0:3:2])
        elif normal_abs[2] == max(normal_abs):
            outside_plot.append(node.coordinates[0:2:1])
        else:
            raise Exception("Max of normal is not found: ",normal_abs)
    # check list
    # [(0, 135), (213, 215), (293, 391)]
    # [(0, 135), (135, 213), (213, 215), (215, 293), (293, 391)]
    # full_indexes = [(0, 130), (130, 213), (213, 215), (215, 298), (298, 391)]
    # final_outside_boundary_coords = get_final_boundary_long_cant(full_indexes, outside_plot, original_outside_boundary)

    import spline_fitting.all_function
    final_outside_boundary_coords = spline_fitting.all_function.get_final_boundary(outside_plot, original_outside_boundary)
    final_outside_boundary_coords.append(final_outside_boundary_coords[0])
    # final_outside_boundary_coords = numpy.array(final_outside_boundary_coords)
    tr_f_out_bd_coords = list(numpy.transpose(numpy.array(final_outside_boundary_coords)))
    # ax.plot(tr_f_out_bd_coords[0], tr_f_out_bd_coords[1], lw=2)
    line, = plt.plot(tr_f_out_bd_coords[0], tr_f_out_bd_coords[1], 'black', lw=1.5)
    plt.show()


    # ---------------Plotting boundary extraction results
    fig, ax = plt.subplots()
    for cluster in inside_boundary:
        cluster_nodes = []
        for node in cluster.values():
            if normal_abs[0] == max(normal_abs):
                cluster_nodes.append(node.coordinates[1:3:1])
                if node == list(cluster.values())[-1]:
                    cluster_nodes.append(list(cluster.values())[0].coordinates[1:3:1])
            elif normal_abs[1] == max(normal_abs):
                cluster_nodes.append(node.coordinates[0:3:2])
                if node == list(cluster.values())[-1]:
                    cluster_nodes.append(list(cluster.values())[0].coordinates[0:3:2])
            elif normal_abs[2] == max(normal_abs):
                cluster_nodes.append(node.coordinates[0:2:1])
                if node == list(cluster.values())[-1]:
                    cluster_nodes.append(list(cluster.values())[0].coordinates[0:2:1])
            else:
                raise Exception("Max of normal is not found: ",normal_abs)
        cluster_nodes = numpy.array(cluster_nodes)
        tr_cluster_nodes = numpy.transpose(cluster_nodes)
        # ax.plot(tr_cluster_nodes[0], tr_cluster_nodes[1],'b--', lw=1)
        line, = plt.plot(tr_cluster_nodes[0], tr_cluster_nodes[1], 'black', lw=1.5)
    plt.show()

    fig, ax = plt.subplots()
    outside_plot.append(outside_plot[0])
    tr_outside_plot = numpy.transpose(numpy.array(outside_plot))
    line, = plt.plot(tr_outside_plot[0], tr_outside_plot[1], 'black', lw=1.5)

    # if normal_abs[0] == max(normal_abs):
    #     line, = plt.plot(tr_outside_plot[1], tr_outside_plot[2],'black', lw=1.5)
    # elif normal_abs[1] == max(normal_abs):
    #     line, = plt.plot(tr_outside_plot[0], tr_outside_plot[2],'black', lw=1.5)
    # elif normal_abs[2] == max(normal_abs):
    #     line, = plt.plot(tr_outside_plot[0], tr_outside_plot[1],'black', lw=1.5)
    # else:
    #     raise Exception("Max of normal is not found: ",normal_abs)
    plt.show()
    f_out_bd_coords = numpy.array(tr_f_out_bd_coords).transpose().tolist()
    return f_out_bd_coords

def plot_original():
    fig, ax = plt.subplots()
    inside_plot = []
    for cluster in inside_boundary:
        cluster_nodes = []
        for node in cluster.values():
            if normal_abs[0] == max(normal_abs):
                cluster_nodes.append(node.coordinates[1:3:1])
            elif normal_abs[1] == max(normal_abs):
                cluster_nodes.append(node.coordinates[0:3:2])
            elif normal_abs[2] == max(normal_abs):
                cluster_nodes.append(node.coordinates[0:2:1])
            else:
                raise Exception("Max of normal is not found: ",normal_abs)
        cluster_nodes.append(cluster_nodes[0])
        inside_plot.append(cluster_nodes)
        tr_cluster = numpy.transpose(cluster_nodes)
        line, = plt.plot(tr_cluster[0], tr_cluster[1], 'black', lw=1.5)

    # ------------Plotting outside boundary
    outside_plot = []

    for node in outside_boundary.values():
        if normal_abs[0] == max(normal_abs):
            outside_plot.append(node.coordinates[1:3:1])
        elif normal_abs[1] == max(normal_abs):
            outside_plot.append(node.coordinates[0:3:2])
        elif normal_abs[2] == max(normal_abs):
            outside_plot.append(node.coordinates[0:2:1])
        else:
            raise Exception("Max of normal is not found: ",normal_abs)
    outside_plot.append(outside_plot[0])

    original_outside_plot = []
    for node in original_outside_boundary.values():
        if normal_abs[0] == max(normal_abs):
            original_outside_plot.append(node.coordinates[1:3:1])
        elif normal_abs[1] == max(normal_abs):
            original_outside_plot.append(node.coordinates[0:3:2])
        elif normal_abs[2] == max(normal_abs):
            original_outside_plot.append(node.coordinates[0:2:1])
        else:
            raise Exception("Max of normal is not found: ",normal_abs)

    tr_out = list(numpy.transpose(outside_plot))
    # line, = plt.plot(tr_out[0], tr_out[1], 'black', lw=1.5)
    # plt.show()
    inside_nodes_list = numpy.array(inside_plot).tolist()
    outside_nodes_list = numpy.array(outside_plot).tolist()
    original_nodes_list = numpy.array(original_outside_plot).tolist()
    return inside_nodes_list,outside_nodes_list,original_nodes_list

def get_nodes_coords_2d(boundary):
    coords = []
    for node in boundary.values():
        if normal_abs[0] == max(normal_abs):
            coords.append(node.coordinates[1:3:1])
        elif normal_abs[1] == max(normal_abs):
            coords.append(node.coordinates[0:3:2])
        elif normal_abs[2] == max(normal_abs):
            coords.append(node.coordinates[0:2:1])
        else:
            raise Exception("Max of normal is not found: ",normal_abs)
    return coords


def make_plots(ext_org=True, ext_spl=True, ext_spl_pac="scp", both_org=True, both_spl_void=True, both_spl_spl=True,
               lw=1):

    # fig, ax = plt.subplots()
    outside_plot=get_nodes_coords_2d(outside_boundary)
    original_outside_plot = get_nodes_coords_2d(original_outside_boundary)

    inside_plot_org = []
    for cluster in inside_boundary:
        cluster_nodes = get_nodes_coords_2d(cluster)
        # cluster_nodes = numpy.array(cluster_nodes)
        inside_plot_org.append(cluster_nodes)

    from geometry.void_replacement import replace_poly as v_rplc
    circles = v_rplc(inside_plot_org)
    #circles in format:[[centrex,centrey],r,[edgex,edgey],A]

    if ext_org:
        # outside_plot=get_nodes_coords_2d(outside_boundary)
        outside_plot.append(outside_plot[0])
        tr_out_org = list(numpy.array(outside_plot).transpose())
        plt.figure(1)
        line, = plt.plot(tr_out_org[0], tr_out_org[1], 'black', lw=lw)
        plt.title("Original external boundary")
        # plt.show()

    if ext_spl:
        if ext_spl_pac == "kz":
            import spline_fitting.all_function
            final_outside_boundary_coords = spline_fitting.all_function.get_final_boundary(outside_plot,
                                                                                           original_outside_boundary,
                                                                                           c1=True)
            final_outside_boundary_coords.append(final_outside_boundary_coords[0])
            tr_f_out_bd_coords = list(numpy.transpose(numpy.array(final_outside_boundary_coords)))
        elif ext_spl_pac == "scp":
            final_outside_boundary_coords = spline_fitting_scp()
            tr_f_out_bd_coords = list(numpy.transpose(numpy.array(final_outside_boundary_coords)))

        plt.figure(2)
        line, = plt.plot(tr_f_out_bd_coords[0], tr_f_out_bd_coords[1], 'black', lw=lw)
        plt.title("External boundary with spline fitting")

    if both_org:
        tr_out_org = list(numpy.array(outside_plot).transpose())
        plt.figure(3)
        plt.title("Both original boundary")
        line, = plt.plot(tr_out_org[0], tr_out_org[1], 'black', lw=lw)

        for cluster in inside_plot_org:
            cluster.append(cluster[0])
            tr_cluster = list(numpy.array(cluster).transpose())
            line, = plt.plot(tr_cluster[0], tr_cluster[1], 'black', lw=lw)

    if both_spl_void:
        fig,ax = plt.subplots()
        fig = plt.figure(4)
        plt.title("External spline and internal voids")

        def circle(x, y, radius=0.15):
            from matplotlib.patches import Circle
            from matplotlib.patheffects import withStroke
            circle = Circle((x, y), radius, clip_on=False, zorder=10, lw=lw,
                            edgecolor='black', facecolor=(0, 0, 0, 0),
                            path_effects=[withStroke(linewidth=5, foreground='w')])
            ax.add_artist(circle)
        for circ in circles:
            circle(circ[0][0],circ[0][1],circ[1])

        # ax.plot(tr_f_out_bd_coords[0], tr_f_out_bd_coords[1], lw=2)
        # plt.figure(2)
        line, = plt.plot(tr_f_out_bd_coords[0], tr_f_out_bd_coords[1], 'black', lw=lw)

    if both_spl_spl:
        plt.figure(5)
        plt.title("Both spline")
        inside_spl_clusters = []
        for cluster_nodes in inside_plot_org:
            smoothness = 0.3
            # if len(cluster_nodes) < 50:
            #     smoothness = 1
            # elif 100 <= len(cluster_nodes) < 200:
            #     smoothness = 40
            # elif 200 <= len(cluster_nodes) < 300:
            #     smoothness = 40
            # else:
            #     smoothness = 80
            cluster_nodes_sm = ss.average_smoothing(cluster_nodes)
            tck, u = splprep(numpy.array(cluster_nodes_sm).transpose(), u=None, s=smoothness, per=1)
            u_new = numpy.linspace(u.min(), u.max(), 50)
            x_new, y_new = splev(u_new, tck, der=0)
            line, = plt.plot(x_new, y_new, 'black', lw=lw)
            inside_spl_cluster = []
            for i in range(len(x_new)):
                inside_spl_cluster.append([x_new[i], y_new[i]])
            inside_spl_clusters.append(inside_spl_cluster)
        line, = plt.plot(tr_f_out_bd_coords[0], tr_f_out_bd_coords[1], 'black', lw=lw)
    plt.show()

    return circles, final_outside_boundary_coords, outside_plot, original_outside_plot, inside_spl_clusters
    # f_out_bd_coords = numpy.array(tr_f_out_bd_coords).transpose().tolist()
    # return f_out_bd_coords


if __name__ == '__main__':
    main_log = lgr.main_log
    main_log.info("Post-processing starts.")

    # para_path = os.getcwd()+'/../Cases/Impeller/'
    # para_path = os.getcwd()+'/../CantD/'
    para_path = os.getcwd() + '/../CantD/'
    sys.path.insert(0, para_path)
    import parameters
    ap = parameters.ap
    parsed_inp_file = utilities.abaqus.inp_reader_v2.parse_inp_file(ap['inp_path'])
    parts_list = utilities.abaqus.inp_tree_processor_v2.process_tree(parsed_inp_file)
    part = get_desired_part(parts_list,ap)
    ## delete parsed parts to release memory
    del parsed_inp_file
    del parts_list

    ##------------ The boundary output is list of node instances
    (inside_boundary, outside_boundary, original_outside_boundary, ext_plane) = \
        geometry.boundary_extractor.get_all_boundary_nodes(ap,part)

    normal = ext_plane[1]
    normal_abs = numpy.absolute(normal)

    circles, f_out_bd_coords, outside_plot, original_outside_plot, inside_spl_clusters = make_plots()

    f_out_bd_coords_list = f_out_bd_coords.copy()
    for i in range(len(f_out_bd_coords)):
        if isinstance(f_out_bd_coords[i],numpy.ndarray):
            f_out_bd_coords_list[i] = f_out_bd_coords[i].tolist()
        elif isinstance(f_out_bd_coords[i],tuple):
            f_out_bd_coords_list[i] = list(f_out_bd_coords[i])
        else:
            pass

    outside_plot_list = outside_plot.copy()
    for i in range(len(outside_plot)):
        if isinstance(outside_plot[i], numpy.ndarray):
            outside_plot_list[i] = outside_plot[i].tolist()
        elif isinstance(outside_plot[i], tuple):
            outside_plot_list[i] = list(outside_plot[i])
        else:
            pass

    inside_points = []
    for cluster in inside_boundary:
        cluster_nodes = get_nodes_coords_2d(cluster)
        cluster_nodes_list = []
        for node in cluster_nodes:
            cluster_nodes_list.append(node.tolist())
        inside_points.append(cluster_nodes_list)

    import json
    with open(ap['test_dir_path'] + 'boundary.txt', 'w') as file:
        out_put = {"circles": circles, "ext_bd": outside_plot_list, "int_bd": inside_points}
        # json.dump(out_put, file)
        out_put=str(out_put)
        file.write(out_put)
        file.close()

    quit()

    ext_keep_coords = original_outside_plot.copy()
    for i in range(len(original_outside_plot)):
        if isinstance(original_outside_plot[i], numpy.ndarray):
            ext_keep_coords[i] = original_outside_plot[i].tolist()
        elif isinstance(original_outside_plot[i], tuple):
            ext_keep_coords[i] = list(original_outside_plot[i])
        else:
            pass

    with open(ap['test_dir_path'] + 'Opted_boundary.txt', 'w') as file:
        out_put = {"inside_boundary": inside_boundary, "outside_boundary": outside_boundary}
        json.dump(out_put, file)
        file.close()

    with open(ap['test_dir_path'] + 'smoothing.txt', 'w') as file:
        out_put = {"circles": circles, "ext_bd": f_out_bd_coords_list, "ext_bd_keep": ext_keep_coords}
        json.dump(out_put, file)
        # json.dump(f_out_bd_coords_list,file)
        file.close()

    with open(ap['test_dir_path'] + 'both_spl.txt', 'w') as file:
        out_put = {"inside_spl": inside_spl_clusters, "ext_bd": f_out_bd_coords_list, "ext_bd_keep": ext_keep_coords}
        json.dump(out_put, file)
        file.close()




