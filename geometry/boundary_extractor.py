'''
Updated 10/08/2018 
Updated 01/01/2019
@Dedao
This version uses string and integer in the bounadry instance
No more instance in instance
'''

import copy

import geometry.constants
import geometry.flat_geometry
import math
import matplotlib.pyplot as plt
import numpy as np


def get_cross_section(ap, part):
    cross_section_set_name = ap['cross_section_set_name']
    cross_section_set = geometry.flat_geometry.get_set(
                                cross_section_set_name,
                                "elset",
                                part)
    return cross_section_set


def get_keep_set(ap, part):
    keep_set_name = ap['keep_set_name']
    keep_set = geometry.flat_geometry.get_set(keep_set_name, "nset", part)
    return keep_set.data

def get_coords_average(list_of_coords):
    x = 0
    y = 0
    z = 0
    coords_len = len(list_of_coords)
    for coord in list_of_coords:
        x+=coord[0]
        y+=coord[1]
        z+=coord[2]
    x/=coords_len
    y/=coords_len
    z/=coords_len
    return (x,y,z)


def get_coords_normal(list_of_coords):
    # print("list_of_coords is:",list_of_coords)
    # for coord in list_of_coords:
    #     coord = np.array(coord)
    # print("list_of_coords is:",list_of_coords)
    v1 = np.array(list_of_coords[0]) - np.array(list_of_coords[1])
    v2 = np.array(list_of_coords[0]) - np.array(list_of_coords[2])
    normal = np.cross(v1,v2)
    normal = normal/np.linalg.norm(normal)
    # print("The normal is:",normal)
    return normal


def get_surface_normal(element_list, part):
    centre_coords_list = []
    for element_label in element_list:
        test_node_coords = []
        element = part.elements[element_label]
        for node_label in element.nodes:
            test_node_coords.append(part.nodes[node_label].coordinates)
        centre_coords_list.append(get_coords_average(test_node_coords))
    normal = get_coords_normal(centre_coords_list)
    print("Normal:", normal)
    return normal


def get_cross_section_top_nodes_pattern(normal, element_label, part):
    """
    :param normal:
    :param element_label:
    :param part:
    :return: pattern: node sequence that is on the surface
    """
    node_labels = part.elements[element_label].nodes
    # print(part.elements[element_label])
    coords = []
    for node_label in node_labels:
        coords.append(part.nodes[node_label].coordinates)
    edge_list = [[(1,2),np.array([coords[1]])-np.array([coords[0]])],
                [(2,3),np.array([coords[2]])-np.array([coords[1]])],
                [(3,4),np.array([coords[3]])-np.array([coords[2]])],
                [(4,1),np.array([coords[0]])-np.array([coords[3]])],
                [(5,6),np.array([coords[5]])-np.array([coords[4]])],
                [(6,7),np.array([coords[6]])-np.array([coords[5]])],
                [(7,8),np.array([coords[7]])-np.array([coords[6]])],
                [(8,5),np.array([coords[4]])-np.array([coords[7]])],
                [(1,5),np.array([coords[4]])-np.array([coords[0]])],
                [(2,6),np.array([coords[5]])-np.array([coords[1]])],
                [(3,7),np.array([coords[6]])-np.array([coords[2]])],
                [(4,8),np.array([coords[7]])-np.array([coords[3]])]]
    edge_along_normal = []
    for edge in edge_list:
        # print("The edge vector is: ",edge)
        if abs(np.dot(edge[-1], normal)) < 0.01:
            pass
        else:
            edge_along_normal.append(edge)
    # print("The edge along normal is: ",edge_along_normal)
    # print(edge_along_normal)
    if len(edge_along_normal) == 4:
        pass
    else:
        raise Exception("More than 4 edges along the normal.",edge_along_normal)

    sample_edge = edge_along_normal[0][0]  # (2,3)
    sample_direction_sign = np.dot(edge_along_normal[0][1],normal)
    start_node_label = node_labels[sample_edge[0]]
    end_node_label = node_labels[sample_edge[1]]
    start_node = part.nodes[start_node_label]
    end_node = part.nodes[end_node_label]
    if len(start_node.neighbours) > len(end_node.neighbours):
        chosen_no = 1
    elif len(start_node.neighbours) == len(end_node.neighbours):
        chosen_no = 0
    else:
        chosen_no = 0

    pattern = []
    for edge in edge_along_normal:
        if np.dot(edge[-1],normal)*sample_direction_sign>0:
            pattern.append(edge[0][chosen_no]-1)
        elif np.dot(edge[-1],normal)*sample_direction_sign==0:
            raise ValueError("The direction of edge should not be zero.",edge)
        else:
            pattern.append((edge[0][1-chosen_no]-1))
    if len(pattern) != len(set(pattern)):
        raise ValueError("The pattern is wrong: {}".format(pattern))
    # print("The cross section top nodes pattern is: ",pattern)
    # Following code is to correct the direction of normal so they will point into the top surface
    if sample_edge[0] == pattern[0]+1:
        correct_vector = edge_along_normal[0][1]
    elif sample_edge[1] == pattern[0]+1:
        correct_vector = edge_along_normal[0][1]*-1
    else:
        raise ValueError("The correcting vector is not found with "
                         "nodes pair: {} and pattern: {}".format(sample_edge, pattern))
    if np.dot(correct_vector, normal) > 0 :
        pass
    elif np.dot(correct_vector, normal) < 0 :
        normal = normal * -1
    else:
        raise ValueError("The correcting vector is wrong: ",correct_vector)
    return pattern, normal


def get_corrected_normal(normal, pattern, sample_elements_list, part):
    """
    This is to avoid the situation where general normal is from varying thickness cross-section
    """
    surface_coords = []
    for element_label in sample_elements_list:
        element = part.elements[element_label]

        node = part.nodes[element.nodes[pattern[0]]]
        surface_coords.append(node.coordinates)
    corrected_normal = get_coords_normal(surface_coords)
    if abs(np.dot(corrected_normal, normal)) > 0.9:
        pass
    else:
        raise Exception("The original normal and corrected normal are:", normal, corrected_normal)
    if np.dot(corrected_normal, normal) > 0:
        pass
    elif np.dot(corrected_normal, normal) < 0:
        corrected_normal *= -1
    else:
        raise ValueError("The dot product of two normals should not be zero: ",normal, corrected_normal)
    return corrected_normal



def get_cross_section_sample_elements(cross_section_set, part):
    cross_section_elements = cross_section_set.data
    # while True:
    #     import random
    #     index = random.randrange(len(cross_section_elements))
    #     element_label = list(cross_section_elements)[index]
    for element_label in cross_section_elements:
        element = part.elements[element_label]
        if len(element.neighbours) >= 3:
            sample_elements = list(element.neighbours)
            return sample_elements
        else:
            pass


def get_cross_section_normal(ap,part):
    if type(ap['cross_section_normal']) == np.ndarray:
        normal = ap['cross_section_normal']
    elif ap['cross_section_normal'] == None:
        cross_section_set = get_cross_section(ap,part)
        test_elements = get_cross_section_sample_elements(cross_section_set, part)
        normal = get_surface_normal(test_elements,part)
        print("The normal is:", normal)
        element_label = test_elements[0]
        pattern, normal_sign_correction = get_cross_section_top_nodes_pattern(normal, element_label, part)
        # corrected_normal = get_corrected_normal(normal_sign_correction, pattern, test_elements, part)
    else:
        raise Exception("Cross section normal in parameters is expected to be a ndarray or None.")


    # print("The cross section normal is: ", corrected_normal)
    return normal


def get_cross_section_nodes(cross_section_set, ap,part):
    cross_section_set_data = list(cross_section_set.data)
    test_elements = get_cross_section_sample_elements(cross_section_set, part)
    if ap['cross_section_normal'] == None:
        normal = get_surface_normal(test_elements,part)
    elif type(ap['cross_section_normal']) == np.ndarray:
        normal = ap['cross_section_normal']
    else:
        raise Exception("Cross section normal in parameters is expected to be a ndarray or None.")
    # print("the normal vector is: ",normal)

    pattern, normal = get_cross_section_top_nodes_pattern(normal, test_elements[0], part)
    # pattern = get_cross_section_top_nodes_pattern(normal, 4000, part)
    # print("the pattern is:",pattern)
    cross_section_nodes = []
    for element_label in cross_section_set_data:
        element = part.elements[element_label]
        for no in pattern:
            cross_section_nodes.append(element.nodes[no])
    cross_section_nodes = list(set(cross_section_nodes))
    print("The length of cross section nodes is: ",len(cross_section_nodes))
    return cross_section_nodes, pattern


def node_on_plane(node, plane, part):
    coord = part.nodes[node].coordinates
    if np.dot(coord - plane[0], plane[1]) == 0:
        on_plane = True
    else:
        on_plane = False
    return on_plane


def get_external_plane(test_element_label, normal, part):
    node_labels = part.elements[test_element_label].nodes
    coords = []
    for node_label in node_labels:
        coords.append(part.nodes[node_label].coordinates)
    edge_list = [[(0, 1), np.array([coords[1]]) - np.array([coords[0]])],
                 [(1, 2), np.array([coords[2]]) - np.array([coords[1]])],
                 [(2, 3), np.array([coords[3]]) - np.array([coords[2]])],
                 [(3, 0), np.array([coords[0]]) - np.array([coords[3]])],
                 [(4, 5), np.array([coords[5]]) - np.array([coords[4]])],
                 [(5, 6), np.array([coords[6]]) - np.array([coords[5]])],
                 [(6, 7), np.array([coords[7]]) - np.array([coords[6]])],
                 [(7, 4), np.array([coords[4]]) - np.array([coords[7]])],
                 [(0, 4), np.array([coords[4]]) - np.array([coords[0]])],
                 [(1, 5), np.array([coords[5]]) - np.array([coords[1]])],
                 [(2, 6), np.array([coords[6]]) - np.array([coords[2]])],
                 [(3, 7), np.array([coords[7]]) - np.array([coords[3]])]]
    edge_along_normal = []
    for edge in edge_list:
        # print("The edge vector is: ",edge)
        if abs(np.dot(edge[-1], normal)) < 0.1:
            pass
        else:
            edge_along_normal.append(edge)
    # print("The edge along normal is: ",edge_along_normal)
    # print(edge_along_normal)
    if len(edge_along_normal) == 4:
        pass
    else:
        raise Exception("More than 4 edges along the normal.", edge_along_normal)

    ext_nodes = []

    edge = edge_along_normal[0]
    edge_label = edge[0]
    start_node = part.nodes[node_labels[edge_label[0]]]
    end_node = part.nodes[node_labels[edge_label[1]]]
    if len(start_node.neighbours) >= len(end_node.neighbours):
        if np.dot(edge[1], normal) > 0:
            pass
        elif np.dot(edge[1], normal) < 0:
            normal = normal * -1
        else:
            raise Exception()
    # elif len(start_node.neighbours) == len(end_node.neighbours):
    else:
        if np.dot(edge[1], normal) < 0:
            pass
        elif np.dot(edge[1], normal) > 0:
            normal = normal * -1
        else:
            raise Exception()

    for edge in edge_along_normal:
        edge_label = edge[0]  # (2,3)
        # sample_direction_sign = np.dot(edge_along_normal[0][1],normal)
        start_node_label = node_labels[edge_label[0]]
        end_node_label = node_labels[edge_label[1]]
        start_node = part.nodes[start_node_label]
        end_node = part.nodes[end_node_label]
        if np.dot(edge[1], normal) > 0:
            ext_nodes.append(end_node_label)
        elif np.dot(edge[1], normal) < 0:
            ext_nodes.append(start_node_label)
        else:
            raise Exception()
    ext_plane = [part.nodes[ext_nodes[0]].coordinates, normal]
    for node in ext_nodes[1:]:
        if node_on_plane(node, ext_plane, part):
            pass
        else:
            raise ValueError("Point not on plane:", node, part.nodes[node], ext_plane)
    return ext_plane


def get_cross_section_nodes_v2(cross_section_set, ap, part):
    cross_section_set_data = list(cross_section_set.data)
    test_elements = get_cross_section_sample_elements(cross_section_set, part)
    if ap['cross_section_normal'] == None:
        normal = get_surface_normal(test_elements, part)
    elif type(ap['cross_section_normal']) == np.ndarray:
        normal = ap['cross_section_normal']
    else:
        raise Exception("Cross section normal in parameters is expected to be a ndarray or None.")
    # print("the normal vector is: ",normal)
    ext_plane = get_external_plane(test_elements[0], normal, part)
    normal = ext_plane[1]
    ap['cross_section_normal'] = normal

    # pattern = get_cross_section_top_nodes_pattern(normal, 4000, part)
    # print("the pattern is:",pattern)
    cross_section_nodes = []
    for element_label in cross_section_set_data:
        element = part.elements[element_label]
        for node in element.nodes:
            if node_on_plane(node, ext_plane, part):
                cross_section_nodes.append(node)
            else:
                pass
    cross_section_nodes = list(set(cross_section_nodes))
    print("The length of cross section nodes is: ", len(cross_section_nodes))
    return cross_section_nodes, ext_plane


def get_boundary_elements(cross_section_set, ap,part):
    both_boundary_elements = []
    for element_label in cross_section_set.data:
        element = part.elements[element_label]
        if element.initially_active:
            len_neighbours = len(element.neighbours)
            for neighbour in element.neighbours:
                if part.elements[neighbour].initially_active:
                    pass
                else:
                    len_neighbours -= 1
                    # element.location += 'opted '
            # this judge can be used to distinguish the original boundary and created boundary
            if ap['element_layers'] == 'single_layer':
                if len_neighbours < 4:
                # it means the element has all neighbours, solid
                # element.location += 'inside '
                    both_boundary_elements.append(element.label)
            if ap['element_layers'] == 'multi_layer':
                if len_neighbours < 5:
                    both_boundary_elements.append(element.label)
    print("The length of boundary elements list is: ",len(both_boundary_elements))
    return both_boundary_elements


def get_boundary_nodes(cross_section_nodes, both_boundary, cross_section_set, part, ap):
    """
    input:  boundary class
    output: boundary nodes label list
    """

    boundary_nodes = []
    for element_label in both_boundary.elements:
        element = part.elements[element_label]
        for node_label in element.nodes:
            if node_label in cross_section_nodes:
                boundary_nodes.append(node_label)
    boundary_nodes = list(set(boundary_nodes))
    print("The length of both booundary nodes is: ",len(boundary_nodes))
    return boundary_nodes


def set_cross_section_node_neighbours(both_boundary, cross_section_nodes, ap, part):
    """ 
    3D boundary
    This should be run before boundary nodes tracing
    """
    print("setting neighbours for cross section nodes")
    for element_label in both_boundary.elements:
        element = part.elements[element_label]
        element_type = geometry.constants.element_label_map[ap['element_shape']]
        for joins in geometry.constants.nodeNeighboursList[element_type]:

            label0 = element.nodes[joins[0]] # int, node label for the joins pair
            label1 = element.nodes[joins[1]]
            if label0 in cross_section_nodes:
                if label1 in cross_section_nodes:
                    part.nodes[label0].neighbours.add(label1)
                    part.nodes[label1].neighbours.add(label0)
    print("Nodes neighbours for cross section are all set.")


def get_start_node(boundary_nodes_list,ap,part):
    # print("Getting start node.")
    normal = np.absolute(get_cross_section_normal(ap, part))
    nodes_coord_dict = {}
    for node_label in boundary_nodes_list:
        node = part.nodes[node_label]
        nodes_coord_dict[node] = node.coordinates.tolist()
    if normal[0] <= normal[2] and normal[1] <= normal[2]:
        start_node = min(nodes_coord_dict.items(), key = lambda x:x[1][0:2])[0]
        start_inverse_vector = np.array([0,-1,0])
    elif normal[0] <= normal[1] and normal[2] <= normal[1]:
        start_node = min(nodes_coord_dict.items(), key = lambda x:x[1][0:3:2])[0]
        start_inverse_vector = np.array([0,0,-1])
    elif normal[1] <= normal[0] and normal[2] <= normal[0]:
        start_node = min(nodes_coord_dict.items(), key = lambda x:x[1][1:3])[0]
        start_inverse_vector = np.array([0,0,-1])
    else:
        raise ValueError("Can't find the start node, cross section normal is {}".format(str(normal)))
    # print("the start node is:",start_node,start_inverse_vector)
    return (start_node,start_inverse_vector)


def scatter_plot(coords_list, plt_title):
    tr_list = list(np.transpose(coords_list))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tr_list[0],tr_list[1],tr_list[2], c='r', marker='.', s=[2, 2, 2])
    ax.set_title(plt_title)
    plt.show()


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def trace_boundary_nodes(boundary_nodes_list, side, ap, part):
    '''
    This function optput the most outside nodes
    '''
    print("Tracing boundary nodes.")
    # print("The boundary_nodes_list is: ", boundary_nodes_list)
    normal = get_cross_section_normal(ap, part)

    (start_node,start_inverse_vector) = get_start_node(boundary_nodes_list,ap,part)

    outside_nodes = {}
    outside_nodes[0]=start_node
    this_coordinate = start_node.coordinates
    this_node = start_node
    last_node = None
    last_inverse_vector = start_inverse_vector
    node_boundary_label = 1
    # Get boundary node one by one
    while True:
        neighbours_dict = {}
        for neighbour_label in this_node.neighbours:
            if neighbour_label in boundary_nodes_list:
                neighbour_node = part.nodes[neighbour_label]

                if neighbour_node == last_node:
                    #print('pass previous node')
                    pass
                else:
                    neighbour_coordinate = neighbour_node.coordinates
                    this_vector = neighbour_coordinate - this_coordinate
                    # angle = angle_between(last_inverse_vector, this_vector)
                    # this_length = np.linalg.norm(this_vector)
                    # last_length = np.linalg.norm(last_inverse_vector)
                    # dot_p = np.dot(last_inverse_vector, np.transpose(this_vector))
                    # det = last_inverse_vector[0]*this_vector[1]-last_inverse_vector[1]*this_vector[0]
                    det = np.dot(normal, np.cross(last_inverse_vector,this_vector))
                    # inner_angle = math.acos(dot_p/(this_length*last_length))
                    inner_angle = angle_between(last_inverse_vector, this_vector)
                    if side == "outside":
                        if det >= 0: # B is CCW of A
                            angle = inner_angle
                        if det < 0: # B is CW of A
                            angle = 2*math.pi - inner_angle
                    elif side == "inside":
                       if det >= 0: # B is CCW of A
                           angle = 2*math.pi - inner_angle
                       if det < 0: # B is CW of A
                           angle = inner_angle
                    else:
                        raise NameError("The side should be either outside or inside.")
                    neighbours_dict[neighbour_node]=angle
                    #print(neighbours_dict)
        # print("The boundary_nodes_list is: ", boundary_nodes_list)
        if neighbours_dict == {}:
            # boundary_nodes_list_plt = []
            # for node_label in boundary_nodes_list:
            #     boundary_nodes_list_plt.append(part.nodes[node_label])
            # scatter_plot(boundary_nodes_list_plt,"boundary_nodes_list_plt")
            raise ValueError("The neighbours of {} is not found on cross section. "
                             "The neighbours found are {}".format(this_node, this_node.neighbours))
        else:
            next_node = min(neighbours_dict, key = neighbours_dict.get)

                #node_x = [this_node.coordinates[0], next_node.coordinates[0]]
                #node_y = [this_node.coordinates[1], next_node.coordinates[1]]
                #plt.plot(node_x, node_y, 'ro')
                #plt.show()
        if next_node == start_node:
        # if next_node in outside_nodes.values():
            break
        else:
            outside_nodes[node_boundary_label] = next_node
            # print("Adding node to boundary: ", next_node)
            # print("The boundary size is: ", node_boundary_label)
            #for case in neighbours_dict:
            #    if case[1] == min(x[1] for x in neighbours_dict):
            #        next_list = case
            #    else:
            #        raise Exception("no minimum neighbour found")
            #outside_nodes[next_list[0].label]=next_list[0]

            # Procced to next node
            last_node = this_node
            this_node = next_node
            this_coordinate = next_node.coordinates
            last_inverse_vector = last_node.coordinates - this_node.coordinates
            node_boundary_label += 1
    return outside_nodes


def get_both_boundary(cross_section_set,cross_section_nodes, ap,part):
    """get both boundary for quasi 2d plate (unit element thickness)
    """
    both_boundary = geometry.flat_geometry.Boundary(label = 'both', elements=[], nodes=[])
    both_boundary.elements = get_boundary_elements(cross_section_set,ap,part)
    both_boundary.nodes = get_boundary_nodes(cross_section_nodes, both_boundary, cross_section_set, part, ap)
    return both_boundary


def get_outside_nodes(both_boundary, ap,part):
    boundary_nodes_list = both_boundary.nodes
    side = "outside"
    outside_nodes = trace_boundary_nodes(boundary_nodes_list, side,ap,part)
    # print(outside_nodes)
    print("The length of outside nodes list is: ",len(outside_nodes))
    return outside_nodes


def item_in_clusters(element_label, clusters):
    in_clusters = False
    cluster_no = None
    for cluster in clusters.items():
        if element_label in cluster[1][0]:
            # print("Element {} in cluster {}".format(element_label, cluster[1][0]))
            in_clusters = True
            cluster_no = cluster[0]
            break
        else:
            pass
    return (in_clusters, cluster_no)


def item_in_cluster_neighbours(element_label, clusters):
    in_cluster = False
    cluster_no = None
    for cluster in clusters.items():
        # print(cluster[1][1])
        if type(cluster[1][1]) == np.ndarray:
            # print("Checking element {} in neighbours {}".format(element_label, cluster[1][1]))
            if element_label in cluster[1][1]:
                in_cluster = True
                cluster_no = cluster[0]
                break
            # else
        else:
            raise Exception("The neighbour list is not right: ",cluster[1][1])
    return in_cluster, cluster_no

def node_on_cross_section_surface(node_to_check, normal, node_on_plane, part):
    node0 = part.nodes[node_on_plane].coordinates
    node_new = part.nodes[node_to_check].coordinates
    # plane function Ax+By+Cz+D=0, (A,B,C) is normal, (x,y,z) is point
    D = -np.dot(normal, node0)
    if np.isscalar(D):
        pass
    else:
        raise ValueError("The surface function is wrong. "
                         "The normal and surface node are: {}, {}.".format(normal, node0))
    dist = np.dot(normal, node_new)+D
    if dist < 0.001:
        return True
    else:
        return False

def add_void_elements(void_boundary_elements, part, void_elements, pattern, normal):
    element0 = void_boundary_elements[0]
    node0 = part.elements[element0].nodes[pattern[0]]
    for elmt_label in void_boundary_elements:
        element = part.elements[elmt_label]
        for neighbour_label in element.neighbours:
            if part.elements[neighbour_label].initially_active:
                pass
            elif not part.elements[neighbour_label].initially_active:
                node_to_check = part.elements[neighbour_label].nodes[pattern[0]]
                if node_on_cross_section_surface(node_to_check, normal, node0, part):
                    void_elements.append(neighbour_label)
                else:
                    pass
            else:
                raise ValueError("The element initially_active state is not set: {}.".format(neighbour_label))
    void_elements = list(set(void_elements))
    return void_elements


# def get_void_elements(void_boundary_elements, part, pattern, normal):
#     void_elements = []
#     void_elements = add_void_elements(void_boundary_elements, part, void_elements, pattern, normal)
# 
#     while True:
#         ve_temp = void_elements.copy()
#         void_elements = add_void_elements(ve_temp, part, void_elements, pattern, normal)
# 
#         if len(ve_temp) == len(void_elements):
#             break
#     print("Check 1.")
#     return void_elements


def get_elements_clusters(void_elements, part, ext_plane):
    # void_elements = []
    # void_elements = add_void_elements(void_boundary_elements, part, void_elements, pattern, normal)
    # while True:
    #     ve_temp = void_elements.copy()
    #     void_elements = add_void_elements(ve_temp, part, void_elements, pattern, normal)
    #     if len(ve_temp) == len(void_elements):
    #         break
    # print("Check 2.")
    clusters = []
    # print("The voids elements are:", void_elements)
    # quit()
    for ve_label in void_elements:
        ve_cluster = [ve_label]
        # ve_cluster = add_void_elements(ve_cluster.copy(), part, ve_cluster, pattern, normal)
        clusters.append(ve_cluster)
    clusters = list(eval(x) for x in set([str(x) for x in clusters]))
    # import copy
    clusters_temp = copy.deepcopy(clusters)
    # print("The initial clusters are: ",clusters)

    while True:
        no_update = True
        for cluster in clusters_temp:
            cluster_no = clusters_temp.index(cluster)
            if cluster_no + 1 < len(clusters_temp):
                if cluster in clusters:
                    if clusters_temp[cluster_no+1] in clusters:


                        # print("Checking cluster: ", cluster)
                        length0 = len(cluster)
                        length1 = len(clusters_temp[cluster_no + 1])
                        cluster_extend = list(set(cluster + clusters_temp[cluster_no+1]))
                        if length0 + length1 > len(cluster_extend):
                            no_update = False
                            clusters.remove(cluster)
                            clusters.remove(clusters_temp[cluster_no+1])
                            clusters.append(cluster_extend)
                        elif length0 + length1 == len(cluster_extend):
                            pass
                        else:
                            raise ValueError("The two lists are: {} and {}.".format(cluster, clusters_temp[cluster_no+1]))
                    else:
                        pass
                else:
                    pass
            else:
                pass
        clusters_temp = clusters.copy()
        if no_update:
            break
    print("Check 3.")
    # print(clusters)
    clusters_temp = clusters.copy()
    while True:
        no_update = True

        for cluster in clusters_temp:
            i = clusters_temp.index(cluster)
            while i+1 < len(clusters_temp):
                if clusters_temp[i+1] in clusters:
                    if cluster in clusters:
                        if not cluster == clusters_temp[i+1]:
                            length0 = len(cluster)
                            length1 = len(clusters_temp[i + 1])
                            cluster_extend = list(set(cluster + clusters_temp[i+1]))
                            if length0 + length1 > len(cluster_extend):
                                no_update = False
                                clusters.remove(cluster)
                                clusters.remove(clusters_temp[i+1])
                                clusters.append(cluster_extend)
                            elif length0 + length1 == len(cluster_extend):
                                print("Comparing clusters: {}, {}".format(cluster, clusters_temp[i+1]))
                                pass
                            else:
                                raise ValueError("The two lists are: {} and {}.".format(cluster, clusters_temp[i+1]))
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
                i = i + 1
        clusters_temp = clusters.copy()
        if no_update:
            break
    print("Check 4.")
    return clusters


def get_elements_clusters_v2(void_elements, part, ):
    queue = []
    ves = void_elements.copy()
    clusters = []
    tabu_list = []
    while len(ves) != 0:
        cluster = []
        queue.append(ves[0])
        while len(queue) != 0:
            element0 = queue.pop(0)
            cluster.append(element0)
            if element0 in ves:
                ves.remove(element0)
            tabu_list.append(element0)
            neighbours = list(part.elements[element0].neighbours)
            void_neighbours = neighbours.copy()
            for e in neighbours:
                if e in void_elements:
                    if e in tabu_list:
                        void_neighbours.remove(e)
                    else:
                        pass
                else:
                    void_neighbours.remove(e)
            queue.extend(void_neighbours)
        clusters.append(cluster)
    return clusters





def get_outside_boundary_elements(outside_boundary_nodes):
    outside_boundary_elements = []
    for node in outside_boundary_nodes.values():
        for element_label in node.elements:
            outside_boundary_elements.append(element_label)
    outside_boundary_elements = list(set(outside_boundary_elements))
    return outside_boundary_elements


def get_inside_boundary_elements(both_boundary, outside_boundary_elements):
    inside_boundary_elements = []
    for element_label in both_boundary.elements:
        if element_label not in outside_boundary_elements:
            inside_boundary_elements.append(element_label)
    # print("The length of inside boundary elements list is: ", len(inside_boundary_elements))
    return inside_boundary_elements


def get_void_elements(part, cross_section_nodes, ext_plane):
    """
    This is to get the void elements on the cross section
    """
    void_boundary_elements = []
    for element in part.elements.values():
        trigger = True
        if element.initially_active == False:
            for node in element.nodes:
                if node_on_plane(node, ext_plane, part):
                    trigger = False
                else:
                    pass
            if trigger == False:
                void_boundary_elements.append(element.label)
    void_boundary_elements = list(set(void_boundary_elements))
    return void_boundary_elements

def plot_polygon(poly):
    x,y = poly.exterior.xy
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(x, y, color='#6699cc', alpha=0.7,
    linewidth=3, solid_capstyle='round', zorder=2)
    ax.set_title('Polygon')
    plt.show()


def get_inside_nodes(cross_section_set, cross_section_nodes, ap, part, outside_boundary_nodes, pattern):
    """
    :param cross_section_set:
    :param cross_section_nodes:
    :param part:
    :param outside_boundary_nodes:
    :return: inside_boundary_nodes: [{1:node1,2:node2,...},{1:node1,...}]
    """
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    void_boundary_elements = get_void_elements(part, cross_section_nodes, pattern)
    outside_coords = []
    normal = get_cross_section_normal(ap, part)
    print("The cross section normal is: ",normal)
    normal_abs = np.absolute(normal)
    vbe = void_boundary_elements.copy()
    if normal_abs[0] == max(normal_abs):
        for node in outside_boundary_nodes.values():
            outside_coords.append(node.coordinates[1:3:1])
        poly = Polygon(outside_coords)
        # print("The polygon is: ",poly)
        # plot_polygon(poly)
        # quit()
        # vbe = void_boundary_elements.copy()
        for element_label in vbe:
            element = part.elements[element_label]
            pt_in_poly = True
            for index in pattern:
                coord = part.nodes[element.nodes[index]].coordinates[1:3:1]
                pt = Point(coord)
                if not pt.within(poly):
                    pt_in_poly = False
            if not pt_in_poly:
                void_boundary_elements.remove(element_label)
        # print("The void boundary elements are: ", void_boundary_elements)
    elif normal_abs[1] == max(normal_abs):
        for node in outside_boundary_nodes.values():
            outside_coords.append(node.coordinates[0:3:2])
        poly = Polygon(outside_coords)
        for element_label in vbe:
            element = part.elements[element_label]
            pt_in_poly = True
            for index in pattern:
                coord = part.nodes[element.nodes[index]].coordinates[0:3:2]
                pt = Point(coord)
                if not pt.within(poly):
                    pt_in_poly = False
            if not pt_in_poly:
                void_boundary_elements.remove(element_label)
    elif normal_abs[2] == max(normal_abs):
        for node in outside_boundary_nodes.values():
            outside_coords.append(node.coordinates[0:2:1])
        # print("Outside coords are: ",outside_coords)
        poly = Polygon(outside_coords)

        # # --------------------Following is test plot for polygon------
        # from matplotlib import pyplot as plt
        # x,y = poly.exterior.xy
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.scatter(x,y)
        # ax.set_title('Polygon')
        # plt.show()
        # import os
        # os.system("pause")
        # # ----------------------------------------------------------
        for element_label in vbe:
            element = part.elements[element_label]
            pt_in_poly = True
            for index in pattern:
                coord = part.nodes[element.nodes[index]].coordinates[0:2:1]
                pt = Point(coord)
                if not pt.within(poly):
                    pt_in_poly = False
            if not pt_in_poly:
                void_boundary_elements.remove(element_label)
    else:
        raise ValueError("Max of normal is not found. The normal is {}, the max is {}"
                         .format(normal,max(normal_abs)))
    # void_elements = get_void_elements(void_boundary_elements, part, pattern, normal)

    # print("The void elements are: ",void_elements)
    # print("The test void elements are: ",test_void_elements)

    clustered_elements = get_elements_clusters(void_boundary_elements, part, pattern, normal)
    # print("The clustered void elements are: ",clustered_elements)
    void_elements_nodes = []
    for cluster in clustered_elements:
        clustered_nodes = []
        for element_label in cluster:
            element = part.elements[element_label]
            for node_label in element.nodes:
                if node_label in cross_section_nodes:
                    clustered_nodes.append(node_label)
                else:
                    pass
        void_elements_nodes.append(list(set(clustered_nodes)))
    # print("The void elements nodes are: ",void_elements_nodes)
    inside_boundary_nodes = []
    for cluster in void_elements_nodes:
        boundary_nodes_list = []
        for node_label in cluster:
            boundary_nodes_list.append(node_label)
        boundary_nodes = trace_boundary_nodes(boundary_nodes_list, 'outside',ap,part)
        inside_boundary_nodes.append(boundary_nodes)
    print("The length of inside nodes list is: ",len(inside_boundary_nodes))
    return inside_boundary_nodes, normal


def trim_coord(coord, normal_index):
    if normal_index == 0:
        coord = coord[1:3:1]
    if normal_index == 1:
        coord = coord[0:3:2]
    if normal_index == 2:
        coord = coord[0:2:1]
    return coord


def get_inside_nodes_v2(cross_section_set, cross_section_nodes, ap, part, outside_boundary_nodes, ext_plane):
    """
    :param cross_section_set:
    :param cross_section_nodes:
    :param part:
    :param outside_boundary_nodes:
    :return: inside_boundary_nodes: [{1:node1,2:node2,...},{1:node1,...}]
    """
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    void_elements = get_void_elements(part, cross_section_nodes, ext_plane)
    outside_coords = []
    normal = ext_plane[1]
    # print("The cross section normal is: ",normal)
    normal_abs = np.absolute(normal)
    # import copy
    vbe = copy.deepcopy(void_elements)
    normal_index = list(normal_abs).index(max(normal_abs))
    for node in outside_boundary_nodes.values():
        outside_coords.append(trim_coord(node.coordinates, normal_index))
    poly = Polygon(outside_coords)
    for element_label in vbe:
        element = part.elements[element_label]
        pt_in_poly = True
        for node in element.nodes:
            if node_on_plane(node, ext_plane, part):
                coord = trim_coord(part.nodes[node].coordinates, normal_index)
                pt = Point(coord)
                if not pt.within(poly):
                    pt_in_poly = False
        if not pt_in_poly:
            void_elements.remove(element_label)

    clustered_elements = get_elements_clusters_v2(void_elements, part)
    # print("The clustered void elements are: ",clustered_elements)
    void_elements_nodes = []
    for cluster in clustered_elements:
        clustered_nodes = []
        for element_label in cluster:
            element = part.elements[element_label]
            for node_label in element.nodes:
                if node_label in cross_section_nodes:
                    clustered_nodes.append(node_label)
                else:
                    pass
        void_elements_nodes.append(list(set(clustered_nodes)))
    # print("The void elements nodes are: ",void_elements_nodes)
    inside_boundary_nodes = []
    for cluster in void_elements_nodes:
        boundary_nodes_list = []
        for node_label in cluster:
            boundary_nodes_list.append(node_label)
        boundary_nodes = trace_boundary_nodes(boundary_nodes_list, 'outside', ap, part)
        inside_boundary_nodes.append(boundary_nodes)
    print("The length of inside nodes list is: ", len(inside_boundary_nodes))
    return inside_boundary_nodes


def get_original_outside_nodes(outside_boundary_nodes, ap, part):
    """

    :param outside_boundary_nodes:
    :param part:
    :return:{index:node_instance, ...}
    """
    original_boundary_nodes = {}
    keep_set = get_keep_set(ap, part)
    for node_pair in outside_boundary_nodes.items():
        is_original_node = True
        if node_pair[-1].label in keep_set:
            pass
        else:
            for element_label in node_pair[-1].elements:
                element = part.elements[element_label]
                if element.initially_active:
                    pass
                else:
                    is_original_node = False
        if is_original_node:
            original_boundary_nodes[node_pair[0]] = node_pair[-1]
    print("The length of original outside nodes list is: ",len(original_boundary_nodes))
    return original_boundary_nodes


def adjust_outside_nodes(outside_boundary_nodes, original_outside_nodes):
    if len(original_outside_nodes) == 0:
        pass
    else:
        spline_start = []
        spline_end = []
        if outside_boundary_nodes[0] in original_outside_nodes.values():
            pass
        else:
            starting_spline = True
            node_index = 0
            while starting_spline:
                spline_start.append(node_index)
                node_index += 1
                if outside_boundary_nodes[node_index] in original_outside_nodes.values():
                    starting_spline = False
                else:
                    pass
            node_index = len(outside_boundary_nodes) - 1
            if outside_boundary_nodes[node_index] in original_outside_nodes.values():
                pass
            else:
                ending_spline = True
                while ending_spline:
                    spline_end.append(node_index)
                    node_index -= 1
                    if outside_boundary_nodes[node_index] in original_outside_nodes.values():
                        ending_spline = False
                    else:
                        pass
    new_out_bd_nodes = outside_boundary_nodes.copy()
    new_org_out_bd_nodes = original_outside_nodes.copy()
    if len(spline_start) != 0 and len(spline_end) != 0:
        for node_index in spline_start:
            del new_out_bd_nodes[node_index]
        for i in range(len(new_out_bd_nodes)):
            new_out_bd_nodes[i] = outside_boundary_nodes[i + len(spline_start)]
        len_out = len(new_out_bd_nodes)
        for ni in spline_start:
            new_out_bd_nodes[len_out + ni] = outside_boundary_nodes[ni]
        # check length
        if len(new_out_bd_nodes) == len(new_out_bd_nodes):
            pass
        else:
            raise ValueError("The new_out_bd_nodes has different length from outside_boundary_nodes", new_out_bd_nodes,
                             outside_boundary_nodes)

        org_node_indexies = list(new_org_out_bd_nodes.keys())
        # import copy
        org_node_indexies_copy = copy.deepcopy(org_node_indexies)
        for i in org_node_indexies_copy:
            new_org_out_bd_nodes[i - len(spline_start)] = original_outside_nodes[i]
    else:
        pass
    return new_out_bd_nodes, new_org_out_bd_nodes









def get_all_boundary_nodes(ap, part):
    cross_section_set = get_cross_section(ap,part)
    # print("cross section set is:",cross_section_set)

    cross_section_nodes, ext_plane = get_cross_section_nodes_v2(cross_section_set, ap, part)
    both_boundary = get_both_boundary(cross_section_set, cross_section_nodes, ap, part)

    set_cross_section_node_neighbours(both_boundary, cross_section_nodes, ap, part)


    outside_boundary_nodes = get_outside_nodes(both_boundary, ap,part)
    original_outside_nodes = get_original_outside_nodes(outside_boundary_nodes, ap, part)
    outside_boundary_nodes, original_outside_nodes = adjust_outside_nodes(outside_boundary_nodes,
                                                                          original_outside_nodes)
    inside_boundary_nodes = get_inside_nodes_v2(cross_section_set, cross_section_nodes,
                                                ap, part, outside_boundary_nodes, ext_plane)

    return (inside_boundary_nodes, outside_boundary_nodes, original_outside_nodes, ext_plane)


if __name__ == '__main__':
    print("read_inp_file is running")

    #utilities.log_setup.configure_loggers(parameters.dirname)
    #main_log = utilities.log_setup.get_main_logger()
    #main_log.info("Optimisation run using Version "+RevNumber)
    #main_log.debug("Optimisation parameters processed.")

    parsed_inp_file = utilities.abaqus.inp_reader_v2.parse_inp_file(parameters.inp_path)
    parts_list = utilities.abaqus.inp_tree_processor_v2.process_tree(parsed_inp_file)
    part = get_desired_part(parts_list)
    elements = part.elements
    nodes = part.nodes
    del parsed_inp_file
    del parts_list





    f = open("element_neighbours.txt","w+")
    for element in both_boundary_elements:
        f.write("%s\n" % element)




    #f = open("parts_list.txt","w+")
    #for item in parts_list:
    #    f.write("%s\n" % parts_list)

    ## Assume there is only one part in the parts_list

    


    


