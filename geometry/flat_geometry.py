'''
Created on 09/10/2013
Updated on 09/01/2019 by Dedao
@author: Daniel Stojanov
'''
import collections
import numpy as np

import geometry.hex_volume
import geometry.constants

class FlatGeometryException(Exception):
    """ An exception raised when there is an error related specifically to the
    geometry of the model, or at least how it is being interpreted/manipulated.

    For example, when asking for the neighbour on the side of an element on which
    there is no neighbour.
    """
    pass

class Node(object):
    """ The node in the flat geometry format.
    """
    __slots__ = ["label", "coordinates", "elements", "neighbours", "boundary_label"]
    def __init__(self, label, coordinates, elements=None):
        """ Tested.
        """
        # Check types
        assert type(label) 			== int
        assert type(coordinates) 	== np.ndarray
        #assert type(elements)		== np.ndarray

        # Fix the formats of the items in the arrays.
        assert type(coordinates[0]) == np.float64
        #assert type(elements[0])	== np.int32 or np.int64
        #assert type(neighbours[0]) == node label


        # Assign values
        self.label 			= label
        self.coordinates 	= coordinates
        self.elements 		= []
        #self.neighbours 	= {}
        self.neighbours 	= set()
        self.boundary_label	= None

    def __repr__(self):
        return "Node: " + str(self.label) + " at: " + str(self.coordinates) + " with cross section neighbours: " + str(self.neighbours)

class Element(object):
    """ The element object in the flat geometry format.
    """
    __slots__ = ["label", "element_type", "nodes", "neighbours", "initially_active",
                 "_centroid_found", "_2centroid", "location"]
    def __init__(self, label, element_type, nodes, neighbours=None, initially_active=True):
        """ Tested.
        """
        # Check types
        assert type(label)			== int
        assert type(element_type)	== str
        assert type(nodes)			== np.ndarray
        #assert type(neighbours)		== np.ndarray

        # Assign values
        self.label 				= label
        self.element_type 		= element_type
        self.nodes 				= nodes
        self.neighbours 		= set()
        self.initially_active	= initially_active

        self._centroid_found = False
        self._2centroid = None
        self.location = ''
    # location = level 1: 'inside ', 'boundary '
    #            level 2: 'original ', 'opted '
    #            level 3: 'internal ', 'external '
    #            E.g. location = 'boundary ' + 'opted ' + 'external '

    def __repr__(self):
        return "Element: " + str(self.label) + " with nodes: " + str(self.nodes) # + " with neighbours " + str(self.neighbours)

class ItemSet(object):
    """ A set of labels in an elset or Node set.
    """
    __slots__ = ["label", "set_type", "data"]
    def __init__(self, label, set_type, data):
        # Check types.
        assert type(label)	== str
        assert set_type 	== "nset" or set_type == "elset"
        assert type(data)	== set

        self.label 		= label
        self.set_type 	= set_type
        self.data		= data

    def __repr__(self):
        return ("\nSet: " + self.label + " of type: " + self.set_type + "\n" +
                str(self.data))

# An optimisation on the element side is that any set of sides that
# includes only one side will reference the communal "side collection".
side_collection = [] # This is a list, since a set of sets is impossible.
# Include the 6 sides.
for i in range(1, 7, 1):
    side_collection.append({i})

def __find_optimised_side_set(side):
    global side_collection

    side_to_collect = None
    for side_set in side_collection:
        if side in side_set:
            side_to_collect = side_set

    # Make sure something was found.
    if side_to_collect == None:
        raise FlatGeometryException("The side: " + str(side) + " was requested from the "
                                                               "optimised side set, but no matching side set was "
                                                               "found.")
    return side_to_collect


def _update_side_set(new_side, current_sides):
    # Use one of the optimised side sets if the current side set is empty.
    if len(current_sides) == 0:
        new_side_set = __find_optimised_side_set(new_side)
    else:
        current_sides.add(new_side)
        new_side_set = current_sides

    return new_side_set


class Surface(object):
    """ A surface defined for part of a part. These are part of 'version 2'
    in which elset/surface pairs are being collected.

    To access the 'my surface' representation requires first building this
    using the .build_my_surface(part_geometry) method first.
    """
    def __init__(self, name, surface_parts):
        # Check set_surface_pairs
        # Make sure each element is a tuple of length 2.
        for pair in surface_parts:
            assert type(pair) == type(tuple())
            assert len(pair) == 2

        # Set values.
        self.name 			= name
        self.surface_parts 	= surface_parts
        # Alias
        self.data = surface_parts

        # My surface related
        self._built = False

    def build_my_surface(self, part_geometry):
        """ A 'my' surface is a slightly different representation of a surface
        that is used for this code_pack. It is a set and a dict which represent the
        surface. The set is the set of element labels that are referenced by
        the surface. The dict maps the element label to a smaller set of all
        the sides of that element included in this surface.

        Sets are just part_geometry.sets, but I felt this overly verbose
        interface might be better for testing.

        :param sets:	A list of the sets, as would be found in
                        part_geometry.sets.
        :type sets:		[set1, set2, ..., setn]
        :returns:		None (assigns a new surface to self.my_surface)
        """
        # The building of the my surface needs to only happen once and not more
        # times. This checks for this.
        if self._built:
            return

        sets = part_geometry.sets
        element_labels = set()
        element_sides = collections.defaultdict(set)

        # Update each component of the surface definition.
        for component in self.surface_parts:
            elset_name 	= component[0]
            side 		= component[1]
            # Collect the element_labels for the set.
            more_labels_set = geometry.flat_geometry.get_set(
                elset_name,
                "elset",
                part_geometry)
            # Add each individual element in the set for this component.
            for element_label in more_labels_set.data:
                element_labels.add(element_label)
                # Add the side for this element, in this component.
                updated_side_set = _update_side_set(
                    new_side		= side,
                    current_sides	= element_sides[element_label])
                # Add the updated side set to the element_sides collection.
                element_sides[element_label] = updated_side_set

        # Save the element labels and sides for this surface.
        self.my_surface = (element_labels, element_sides)
        # Flag that the 'my surface' has been built.
        self._built = True

    def get_my_surface(self):
        """ This will get a representation of this surface in the
        'my surface' format.
        """
        # First check that the my surface has been built.
        if not self._built:
            # Build the 'my surface' if it hasn't been done already.
            #if not self.built:
            #	self.build_my_surface(sets, part_geometry)
            raise FlatGeometryException("The my surface representation for this surface "
                                        "has not yet been built.")

        return self.my_surface

    def __repr__(self):
        return ("\nSurface: " + self.name + "contains: " + "\n" +
                str(self.surface_parts) + "\n")


class Part(object):
    """ A part in the flat geometry format.
    """
    def __init__(self, label, nodes, elements, sets):
        self.label 		= label
        self.nodes 		= nodes
        self.elements 	= elements
        self.sets		= sets



    def __repr__(self):
        return ("Part name: " + self.label + "\n" +
                "Elements: " + str(self.elements) + "\n" +
                "Nodes: " + str(self.nodes) + "\n")

class Boundary(object):
    """ A boundary of the part.
    """
    __slots__ = ["label", "nodes", "elements"]
    def __init__(self, label, nodes, elements):

        # Assign values
        self.label 			= label
        self.nodes  		= nodes
        self.elements 		= elements

    # E.g. nodes = {node_index:node_instance} ### This is not used anymore
    # E.g. nodes = [node_index, node_index]
    # E.g. elements = [element_index, element_index]

    def __repr__(self):
        return ("Boundary: " + self.label + "\n" +
                "Elements: " + str(self.elements) + "\n" +
                "Nodes: " + str(self.nodes) + "\n")




#########################
#                       #
#       Utilities       #
#                       #
#########################

def get_element_centroid(element, part_geometry):
    """ Given an element instance and the corresponding part_geometry, return
    the centroid position of the element.
    """
    # Return the centroid if it's known already.
    if element._centroid_found == True:
        return element._2centroid
    # Otherwise work it out.
    else:
        nodes_counted = 0
        centroid = None
        for node_label in element.nodes:
            # Use the first node's coordinates to start.
            if nodes_counted == 0:
                centroid = part_geometry.nodes[node_label].coordinates.copy()
            else:
                try:
                    centroid += part_geometry.nodes[node_label].coordinates
                except:
                    import pdb
                    pdb.set_trace()
            # Increment counter
            nodes_counted += 1
        # Divide to get the average
        centroid *= (1. / nodes_counted)

        # Store result for next time.
        element._centroid_found = True
        element._2centroid = centroid
        return centroid

def get_element_by_label(element_label,part_geometry):
    element_instance = part_geometry.elements[element_label]
    return element_instance

def get_element_centroid_by_label(element_label, part_geometry):
    element_instance = part_geometry.elements[element_label]
    centroid = get_element_centroid(element_instance, part_geometry)
    return centroid

def get_element_volume(element, part_geometry):
    is_C83DR 	= element.element_type == "C3D8R"
    is_C83D		= element.element_type == "C3D8"
    if is_C83DR or is_C83D:
        node_coordinates = []
        for node_label in element.nodes:
            node_coordinates.append(part_geometry.nodes[node_label].coordinates)
        return geometry.hex_volume.hex_volume(node_coordinates)
    else:
        return 1.0

def get_element_volume_by_label(element_label, part_geometry):
    """
    A convenient function that is equivalent to the above, but takes the
    element's label, rather than the instance.
    """
    element = part_geometry.elements[element_label]
    volume = get_element_volume(element, part_geometry)
    return volume

def get_set(label, set_type, part_geometry):
    for set in part_geometry.sets:
        name_match 		= label == set.label
        # set_type is elset or nset
        set_type_match 	= set_type == set.set_type
        if name_match and set_type_match:
            return set

    # Reached if set was not found.
    raise FlatGeometryException("Requested set: " + label + " from part: " + part_geometry.label + " not found.")

whole_part_volumes = {}

def get_whole_part_volume(part_geometry):
    global whole_part_volumes

    # Return volume if already calculated.
    if part_geometry.label in whole_part_volumes:
        return whole_part_volumes[part_geometry.label]

    total_volume = 0.
    # Calculate volume
    for element in part_geometry.elements.values():
        total_volume += get_element_volume(element, part_geometry)

    # Set and return
    whole_part_volumes[part_geometry.label] = total_volume
    return total_volume


#########################
#						#
#    Conversion code_pack.   #
#						#
#########################

def get_node_array_for_nodes(nodes):
    """ Returns an array containing the node labels, from a dictionary of node objects.

    Tested.

    :param nodes:	Dictionary of nodes.
    :type nodes:	{label: node,...}
    :return:		nodes_array
    :rtype:			np.array( [label,...] )
    """
    nodes_array = np.array([0]*len(nodes))

    for i, node in enumerate(nodes.values()):
        nodes_array[i] = node.label

    return nodes_array

def get_neighbour_array_for_element(element):
    """ Returns an array of the neighbour element labels for the given element.

    Tested.
    """
    neighbours_array = np.array([0]*len(element.neighbours))

    for i, neighbour in enumerate(element.neighbours.values()):
        neighbours_array[i] = neighbour.element.label

    return neighbours_array

def get_elements_from_part(part):
    """ Get a simple element geometry from an object part.
    """
    elements = [None] * (max(part.elements.keys())+1)

    for element in part.elements.values():
        # Necessary details
        label 		 = element.label
        element_type = element.elementType
        nodes 		 = get_node_array_for_nodes(element.nodes)
        neighbours	 = get_neighbour_array_for_element(element)


        elements[element.label] = Element(label, element_type, nodes, neighbours)
    return elements

def get_element_array_from_elements(elements):
    element_array = np.array([0]*len(elements))

    for i, element in enumerate(elements.values()):
        element_array[i] = element.label
    return element_array

def get_nodes_from_part(part):
    """ Get a simple list of nodes from an object part.
    """
    nodes = [None] * (max(part.nodes.keys())+1)

    for node in part.nodes.values():
        label = node.label
        coordinates = node.coordinates
        elements = get_element_array_from_elements(node.elements)

        nodes[node.label] = Node(label, coordinates, elements)

    return nodes


def get_flat_geometry_from_part(part):
    """ Returns a flattened, simpler geometry from a part that uses references.
    """
    elements = get_elements_from_part(part)
    nodes 	 = get_nodes_from_part(part)
    label 	 = part.label

    part = Part(label, nodes, elements)
    return part

#########################
#						#
#    Navigation code_pack.   #
#						#
#########################

def get_commonality_mask(element_nodes, common_nodes):
    mask = np.array([0]*len(element_nodes), dtype=bool)

    for i, node in enumerate(element_nodes):
        if node in common_nodes:
            mask[i] = True

    return mask

def get_neighbour_on_side(element_label, side, part_geometry):#, element_type="HEX"):
    element_neighbours = part_geometry.elements[element_label].neighbours
    desired_neighbour_label = None

    for neighbour_label in element_neighbours:
        # Finding common nodes.
        element_nodes 	= part_geometry.elements[element_label].nodes
        neighbour_nodes = part_geometry.elements[neighbour_label].nodes
        common_nodes = set(element_nodes) & set(neighbour_nodes)

        # Stop if this neighbour is the one on the side of interest.
        mask = get_commonality_mask(element_nodes, common_nodes)
        if (mask == geometry.constants.nodes_on_a_side_hex[side]).all():
            desired_neighbour_label = neighbour_label
            break
    else:
        # Raise an exception if there was no neighbour found.
        raise FlatGeometryException("The neighbour to element: " +
                                    str(element_label) + " on side: " + str(side)
                                    + " could not be found. It appears this "
                                      "element is on the edge of the part.")

    return desired_neighbour_label

def get_nodes_on_elements_side(node_labels, side):
    """ Currently assumes a hex element.
    """
    mask = geometry.constants.nodes_on_a_side_hex[side]
    return node_labels[mask]

def get_neighbour_sharing_nodes(element_label, node_labels, part_geometry):
    """ Given an element_label and node_labels, will return the element
    label of the element that also has those labels.
    """
    element_neighbours = part_geometry.elements[element_label].neighbours
    for neighbour_label in element_neighbours:
        neighbours_nodes = part_geometry.elements[neighbour_label].nodes
        if set(node_labels).issubset(set(neighbours_nodes)):
            return neighbour_label

    # Neighbour not found
    return None

def get_side_number_given_nodes(element_label, nodes, part_geometry):
    """ Given an element label and a list of nodes, will determine and return
    the side of the element on which these nodes lie.

    The nodes are a boolean array of flags of which of the eight nodes on
    this element are present on the side of interest.
    """
    # Generate the mask corresponding to the nodes list.
    all_elements_nodes = part_geometry.elements[element_label].nodes
    discovered_side_no = None
    for side_no, node_mask in geometry.constants.nodes_on_a_side_hex.items():
        # Test this mask to see if it is the correct side.
        side_nodes = all_elements_nodes[node_mask]
        # Check that the two sets of nodes match.
        if sorted(side_nodes) == sorted(nodes):
            discovered_side_no = side_no
            break
    else:
        raise FlatGeometryException("The node mask:\n" + str(nodes)
                                    + "\nfor element: " + str(element_label)
                                    + " has failed to find a neighbour and "
                                      "appears to not be a valid set of nodes"
                                      "for a side.")

    return discovered_side_no

def get_nodes_on_element_side(element_label, side, part_geometry):
    """ Given an element and a side, returns the nodes that are on that side
    of the element.
    """
    node_list = part_geometry.elements[element_label].nodes
    # Get the mask that corresponds to the selected side.
    mask = geometry.constants.nodes_on_a_side_hex[side]
    nodes = node_list[mask]
    return nodes

# # Current TODO.
# def _get_node_labels_given_mask_and_element(element_label, nodes_mask, part_geometry):
# 	""" Takes a boolean mask and the supplied element label, will return the
# 	nodel labels this element and node correspond to.
# 	
# 	:param element_label:	The element number.
# 	:type element_label:	int
# 	:param nodes:			A mask of the nodes on the side of the element.
# 							i.e. np.array([True, False, ..., True], dtype=bool)
# 	:type nodes:			np.array
# 	:param part_geometry:	The part of which the element and side are a part.
# 	:type part_geometry:	geometry.flat_geometry.Part
# 	
# 	:returns:				The element and its outer side, the point at which
# 							the part has been eroded.
# 	:rtype:					(int, int) --> (element_label, side_node)
# 	"""
# 	# Get the node list for the given element number.
# 	# node_list is an np.array
# 	node_list = part_geometry.elements[element_label].nodes 
# 	
# 	# Collect the nodes that are positive against the mask.
# 	selected_node_labels = node_list[nodes_mask]
# 	
# 	# Return collected nodes.
# 	return selected_node_labels


# Given a surface number, this returns the surface on the opposite side.
def get_elements_following_side(element_label, side, part_geometry):
    """ Given an element and a side, it will return a list of all the elements
    connected along that side. Quits when an element no longer has a neighbour
    on that side.

    Warning: Will infinitely loop for a ring-like structure.
    """
    label_list = [element_label]
    last_element_found = False
    while not last_element_found:
        # Get the side of the surface element, opposite to the part's outer
        # surface (i.e. the side of the element 1 unit in from the outer surface)
        try:
            # This will raise exception if there is not another element.
            next_neighbour_label = get_neighbour_on_side(	element_label,
                                                             side,
                                                             part_geometry)
            # This will only happen if the next neighbour was successfully found.
            label_list.append(next_neighbour_label)

        except FlatGeometryException:
            last_element_found = True

        # If the last element was found, you don't want to be looking for its
        # neighbours.
        if last_element_found:
            continue

        """ To get the side of the neighbour that is shared with the last element
        it is not sufficient to just assume that the elements are arranged such that
        the top side of the first element is the same as the top of the next.
        Instead, opposite sides of an element are those that share no common nodes
        between those two sides.
        """
        # Get the nodes the two elements shared.
        shared_nodes = get_nodes_on_element_side(element_label, side, part_geometry)
        # Get the side of the neighbour element that shared these nodes.
        side = get_side_number_given_nodes(next_neighbour_label, shared_nodes, part_geometry)
        # The next side will be the one on the opposite side of the one between
        # the two of this iteration.
        side = geometry.constants.opposite_surface[side]
        element_label = next_neighbour_label

    return label_list

#####################
# Geometry creation #
#####################





if __name__ == '__main__':
    pass
