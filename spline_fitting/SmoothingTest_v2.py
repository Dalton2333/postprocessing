import utilities.abaqus.inp_reader_v2
import utilities.abaqus.inp_tree_processor_v2
import read_inp_file_v2 as read_inp_file
import matplotlib.pyplot as plt
import parameters
import numpy

if __name__ == '__main__':
    ## get parsed part
    parsed_inp_file = utilities.abaqus.inp_reader_v2.parse_inp_file(parameters.inp_path)
    parts_list = utilities.abaqus.inp_tree_processor_v2.process_tree(parsed_inp_file)
    part = read_inp_file.get_desired_part(parts_list)
    ## delete parsed parts to release memory
    del parsed_inp_file
    del parts_list

    # --------- Following is the boundary with all element nodes, version 1
    #both_boundary = read_inp_file.get_both_boundary(part)
    #both_boundary_nodes = []
    #for element in both_boundary.elements.values():
    #    for node_label in element.nodes:
    #        coordinate = part.nodes[node_label].coordinates
    #        both_boundary_nodes.append((coordinate[0],coordinate[1],coordinate[2]))
    #both_boundary_nodes = list(set(both_boundary_nodes))
    #tr_both = list(numpy.transpose(both_boundary_nodes))

    # ----------Following is the boundary nodes only, version 2
    both_boundary = read_inp_file.get_both_boundary(part)
    both_nodes_coords = []
    for node_index in both_boundary.nodes:
        #print(part.nodes[node_index].coordinates)
        node_coord = tuple(part.nodes[node_index].coordinates)
        both_nodes_coords.append(node_coord)
    print(both_nodes_coords)
    tr_both = list(numpy.transpose(both_nodes_coords))

    plt.scatter(tr_both[0],tr_both[1],c='r', marker='.',s=1)
    plt.title("Both boundary elements")
    plt.show()
    # Plotting boundary elements
    #from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(nodes_x, nodes_y, nodes_z, c='r', marker='.', s = 1)
    #ax.set_zlim(0, 10)
    #plt.title("Elements in 3D")
    
    ## Plotting outside boundary:
    outside_boundary = read_inp_file.get_outside_boundary(both_boundary, part)
    outside_boundary_coords = []
    for node_index in sorted(outside_boundary.nodes.keys()):
        coordinate = outside_boundary.nodes[node_index].coordinates
        outside_boundary_coords.append((coordinate[0], coordinate[1], coordinate[2]))
    #plt.scatter(nodes_x, nodes_y, c='r', marker='.', s = 1)
    tr_outside_boundary_coords = list(numpy.transpose(outside_boundary_coords))
    plt.plot(tr_outside_boundary_coords[0], tr_outside_boundary_coords[1], 'ro-', markersize = 2)
    plt.grid(True)
    #plt.plot(out_x, out_y, 'b-')
    plt.title("External boundary nodes")
    plt.show()
        
    ## Plotting original boundary:
    original_outside_nodes_list = read_inp_file.get_original_boundary_nodes(outside_boundary, part)
    original_outside_nodes_dict = {}
    for node_index, node in outside_boundary.nodes.items():
        if node in original_outside_nodes_list:
            original_outside_nodes_dict[node_index] = node
    original_outside_coords = []
    for node_index in sorted(original_outside_nodes_dict.keys()):
        original_outside_coords.append(outside_boundary.nodes[node_index].coordinates)
    tr_org_out_coords = list(numpy.transpose(original_outside_coords))

    fig, ax = plt.subplots()
    # ax.scatter(ts_coordinate_bspline[0],ts_coordinate_bspline[1],s=0.5,marker='o')
    ax.plot(tr_org_out_coords[0],tr_org_out_coords[1],'ro', markersize = 2)
    #ax.scatter(ts_temp[0],ts_temp[1],s=1,marker='o')
    ax.grid(True)
    plt.title('Original outside boundary')
    plt.show()

    # Get final boundary
    import all_function
    final_outside_boundary_coords = all_function.get_final_boundary(outside_boundary_coords, original_outside_nodes_dict)
    tr_f_out_bd_coords = list(numpy.transpose(final_outside_boundary_coords))
    
    fig, ax = plt.subplots()
    ax.plot(tr_f_out_bd_coords[0],tr_f_out_bd_coords[1],'r-', lw=1)
    ax.plot(tr_outside_boundary_coords[0], tr_outside_boundary_coords[1], 'ro', markersize = 2)
    #ax.scatter(ts_temp[0],ts_temp[1],s=1,marker='o')
    ax.grid(True)
    plt.title('Final outside boundary')
    plt.show()
    
    # Get internal boundary






    # -----------final design with void
    #fig, ax = plt.subplots()
    #ax.plot(tr_f_out_bd_coords[0],tr_f_out_bd_coords[1],'r-', lw=1)
    #ax.plot(tr_outside_boundary_coords[0], tr_outside_boundary_coords[1], 'ro', markersize = 2)
    ##ax.scatter(ts_temp[0],ts_temp[1],s=1,marker='o')
    #ax.grid(True)

    #from matplotlib.patches import Circle
    #from matplotlib.patheffects import withStroke

    ## ---------circle
    # #(51.5, 19.3), 1.784
    ## (68.3, 17.6), 1.693
    #circle=Circle((51.5, 19.3), 1.784, clip_on=False, zorder=10, linewidth=1,
    #                edgecolor='red', facecolor='none', label='Internal void',
    #                path_effects=[withStroke(linewidth=1, foreground='w')])
    #ax.add_artist(circle)


    #plt.title('overall outside boundary')
    #ax.grid(True)
    #plt.legend(loc="best")
    #plt.show()
    



