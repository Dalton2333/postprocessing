import parameters
import math
import numpy as np
import matplotlib.pyplot as plt

def get_desired_part(parts_list):
	part_name_requested = parameters.part_name
	for part in parts_list:
		if part.label == part_name_requested:
			return part
	
	# This is reached if the requested part was never found.
	raise Exception("Could not find the part: " + part_name_requested +
				" as requested in the input file.")

def get_boundary_elements(part):
    both_boundary_elements = {}
    for element in part.elements.values():
        if element.initially_active:
            len_neighbours = len(element.neighbours)
            for neighbour in element.neighbours:
                if part.elements[neighbour].initially_active:
                    pass
                else:
                    len_neighbours -= 1
                    element.location += 'opted '
            # this judge can be used to distinguish the original boundary and created boundary
            if len_neighbours == 4:
                # it means the element has all neighbours, solid
                element.location += 'inside '
                pass
            else:
                both_boundary_elements[element.label]=element
                element.location += 'boundary '
                if not 'opted ' in element.location:
                    element.location += 'original '
        
            #if element.element_type == "C3D8R" or "C3D8":
            #    if parameters.part_thickness == 1:
                    #if len(element.neighbours) == 4:
                    #    # we can add up a new instance variable: location = external boundary / internal boundary / inside solid
                    #    element.location = "inside solid"
                    #else:
                    #    both_boundary_elements[element.label]=element
                #else:
                #    raise Exception("Part thickness is not 1.")
            #else:
            #    raise Exception("Non hexagonal element given.")
    #node_x = []
    #node_y = []
    #for element in both_boundary_elements.values():
    #    for node_label in element.nodes:
    #        node_x.append(part.nodes[node_label].coordinates[0])
    #        node_y.append(part.nodes[node_label].coordinates[1])
    #plt.plot(node_x, node_y, 'ro')
    #plt.show()
    return both_boundary_elements

from operator import attrgetter, methodcaller, itemgetter

def get_boundary_nodes(both_boundary,part):
    boundary_nodes = {}
    for element in both_boundary.elements.values():
        for node_label in element.nodes:
            
            if part.nodes[node_label].coordinates[parameters.thickness_direction] == parameters.thickness_starting_coord:
                # coordinate = part.nodes[node_label].coordinates.copy()
                boundary_nodes[node_label]=part.nodes[node_label]
            else:
                pass
    #node_x = []
    #node_y = []
    #for node in boundary_nodes.values():
    #    node_x.append(node.coordinates[0])
    #    node_y.append(node.coordinates[1])
    #plt.plot(node_x, node_y, 'ro')
    #plt.show()
    return boundary_nodes

def get_boundary_nodes_v2(both_boundary, part):
    boundary_nodes = []
    for element in both_boundary.elements.values():
        neighbouring_node_list = []
        for neighbour_label in element.neighbours:
            neighbour = part.elements[neighbour_label]
            neighbouring_node_list.append(neighbour.nodes)
        for node_label in element.nodes:
            repeats = 0
            # repeats of this node in neighbouring elements
            if repeats < 2:
                boundary_nodes.append(node_label)
    boundary_nodes = list(set(boundary_nodes))
    for node_label in boundary_nodes:
        if part.nodes[node_label].coordinates[parameters.thickness_direction] != 0:
            boundary_nodes.remove(node_label)
            # check how does remove works !!!!!!!!!!!!!!
        boundary_bodes_dict = {}
    for node_label in boundary_nodes:
        boundary_nodes_dict[nnode_label] = part.nodes[node_label]
    return boundary_nodes_dict

import geometry.constants
        
def set_node_neighbours(both_boundary, part):
    """ 2D boundary only
    """
    # setting neighbours for all boundary nodes
    for element in both_boundary.elements.values():
        for joins in geometry.constants.nodeNeighboursList[parameters.element_shape]:
            # this is for 2D only
            label0 = element.nodes[joins[0]] # int, node label for the joins pair
            label1 = element.nodes[joins[1]]
            if label0 in both_boundary.nodes:
                if label1 in both_boundary.nodes:
                    both_boundary.nodes[label0].neighbours[label1] = part.nodes[label1]
                    both_boundary.nodes[label1].neighbours[label0] = part.nodes[label0]

def get_outside_nodes_v2(both_boundary, part):
    boundary_nodes_dict = get_boundary_nodes_v2(both_boundary, part)
    sorted_boundary_nodes = sorted(boundary_nodes_dict, key = lambda x: (x.coordinates[1], x.coordinates[0]))

    start_node = sorted_boundary_nodes[0]
    outside_nodes = {}
    outside_nodes[0]=start_node
    this_coordinate = start_node.coordinates
    this_node = start_node
    last_node = None
    last_inverse_vector = np.array([-1,0,0])
    node_boundary_label = 1
    # Get boundary node one by one
    while True:
        neighbours_dict = {}
        for neighbour in this_node.neighbours.values():
            if neighbour == last_node:
                #print('pass previous node')
                pass
            else:
                neighbour_coordinate = neighbour.coordinates
                this_vector = neighbour_coordinate - this_coordinate
                this_length = np.linalg.norm(this_vector)
                last_length = np.linalg.norm(last_inverse_vector)
                dot_p = np.dot(last_inverse_vector, np.transpose(this_vector))
                det = last_inverse_vector[0]*this_vector[1]-last_inverse_vector[1]*this_vector[0]
                inner_angle = math.acos(dot_p/(this_length*last_length))
                if det >= 0: # B is CCW of A
                    angle = inner_angle
                if det < 0: # B is CW of A
                    angle = 2*math.pi - inner_angle
                neighbours_dict[neighbour]=angle
        next_node = min(neighbours_dict, key = neighbours_dict.get)
                #node_x = [this_node.coordinates[0], next_node.coordinates[0]]
                #node_y = [this_node.coordinates[1], next_node.coordinates[1]]
                #plt.plot(node_x, node_y, 'ro')
                #plt.show()
        if next_node == start_node:
            break
        outside_nodes[node_boundary_label] = next_node
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

def get_outside_nodes(both_boundary, part):
    set_node_neighbours(both_boundary, part)

    #node_x = []
    #node_y = []
    #for node in both_boundary.nodes.values():
    #    print(node.label)
    #    print(len(node.neighbours))
    #    for neighbour in node.neighbours.values():
    #        node_x.append(neighbour.coordinates[0])
    #        node_y.append(neighbour.coordinates[1])
    #plt.plot(node_x, node_y, 'ro')
    #plt.show()
    if parameters.thickness_direction == 0:
        pass # this is saved for later completion
    if parameters.thickness_direction == 1:
        pass # this is saved for later completion
    if parameters.thickness_direction == 2:
        sorted_boundary_nodes = sorted(both_boundary.nodes.values(), key = lambda x: (x.coordinates[1], x.coordinates[0]))

    start_node = sorted_boundary_nodes[0]
    outside_nodes = {}
    outside_nodes[0]=start_node
    this_coordinate = start_node.coordinates
    this_node = start_node
    last_node = None
    last_inverse_vector = np.array([-1,0,0])
    node_boundary_label = 1
    # Get boundary node one by one
    while True:
        neighbours_dict = {}
        for neighbour in this_node.neighbours.values():
            if neighbour == last_node:
                #print('pass previous node')
                pass
            else:
                neighbour_coordinate = neighbour.coordinates
                this_vector = neighbour_coordinate - this_coordinate
                this_length = np.linalg.norm(this_vector)
                last_length = np.linalg.norm(last_inverse_vector)
                dot_p = np.dot(last_inverse_vector, np.transpose(this_vector))
                det = last_inverse_vector[0]*this_vector[1]-last_inverse_vector[1]*this_vector[0]
                inner_angle = math.acos(dot_p/(this_length*last_length))
                if det >= 0: # B is CCW of A
                    angle = inner_angle
                if det < 0: # B is CW of A
                    angle = 2*math.pi - inner_angle
                neighbours_dict[neighbour]=angle
        next_node = min(neighbours_dict, key = neighbours_dict.get)
                #node_x = [this_node.coordinates[0], next_node.coordinates[0]]
                #node_y = [this_node.coordinates[1], next_node.coordinates[1]]
                #plt.plot(node_x, node_y, 'ro')
                #plt.show()
        if next_node == start_node:
            break
        outside_nodes[node_boundary_label] = next_node
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

def get_outside_elements(outside_boundary, part):
    outside_elements = {}
    for node in outside_boundary.nodes.values():
        for element_label in node.elements:
            outside_elements[element_label]=part.elements[element_label]
    
    return outside_elements

def get_outside_boundary(part):
    """ Get the outside boundary for quasi 2D plate (unit element thickness)
    """
    both_boundary = geometry.flat_geometry.Boundary(label = 'both', elements={}, nodes={})
    outside_boundary = geometry.flat_geometry.Boundary(label = 'outside', elements={}, nodes={})
    #inside_boundary = geometry.flat_geometry.Boundary(label = 'inside', elements=[], nodes=[])

    both_boundary.elements = get_boundary_elements(part)
    both_boundary.nodes = get_boundary_nodes(both_boundary,part)
    outside_boundary.nodes = get_outside_nodes(both_boundary, part)
    outside_boundary.elements = get_outside_elements(outside_boundary, part)
    return outside_boundary

def get_original_boundary_elements(boundary, part):
    original_boundary_elements = []
    for element in boundary.elements.values():
        # this is for 2d case only
        if 'original' in element.location:
            original_boundary_elements.append(element)
    return original_boundary_elements

def get_original_boundary_nodes(boundary, part):
    original_boundary_nodes = []
    for node in boundary.nodes.values():
        is_original_node = False
        for element_label in node.elements:
            element = part.elements[element_label]
            if 'original' in element.location:
                is_original_node = True
        if is_original_node:
            original_boundary_nodes.append(node)
    return original_boundary_nodes





    
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

    


    


