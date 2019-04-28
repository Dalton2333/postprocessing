'''
Created on 09/10/2013

@author: Daniel Stojanov
'''

import numpy as np

#import utilities.log_setup
#main_log = utilities.log_setup.get_main_logger()

import geometry.flat_geometry

class MissingSubBlock(Exception):
	pass

def get_value(parameters, key_requested):
	for key, value in parameters:
		if key == key_requested:
			return value
			
	# This is only reached if a key is not found.
	raise Exception("Key: " + key_requested + " cannot be found.")

#########################
#         Nodes         #
#########################

def get_node_label(node_line):
	label = int(node_line[0])
	return label

def get_node_coordinates(node_line):
	coordinates = np.array([0.]*(len(node_line)-1))
	for i, coordinate in enumerate(node_line[1:]):
		coordinates[i] = float(coordinate)
	return coordinates

def process_nodes(nodes_block):
	nodes = {}
	for node in nodes_block.data:
		label = get_node_label(node)
		coordinates = get_node_coordinates(node)
		
		# Create new Node
		new_node = geometry.flat_geometry.Node(label, coordinates)
		nodes[label] = new_node
	return nodes
	
#########################
#       Elements        #
#########################

def get_element_type(element_block):
	element_type = get_value(element_block.parameters, "type")
	
	# Element type not found
	if element_type is None:
		raise Exception("Element type could not be found.")
	
	return element_type

def get_element_label_from_dataline(element_data_line):
	initially_active = True
	label = element_data_line[0]
	# Check for comments.
	if label.startswith("**"):
		# Strip comment from element label
		label = label[2:]
		initially_active = False
	# Cast to int
	label = int(label)
	return label, initially_active
	
def get_element_nodes_from_dataline(element_data_line):
	nodes_array = np.array([0]*(len(element_data_line)-1))
	# Process node labels
	for i, node in enumerate(element_data_line[1:]):
		nodes_array[i] = int(node)
	return nodes_array

def process_elements(elements_block):
	elements = {}
	
	# Assume no comments within the element body.
	element_type = get_element_type(elements_block)
	
	for element_data_line in elements_block.data:
		label, initially_active = get_element_label_from_dataline(element_data_line)
		nodes_array 			= get_element_nodes_from_dataline(element_data_line)
		
		# Create and add new Element.
		new_element = geometry.flat_geometry.Element(label, element_type, nodes=nodes_array, initially_active=initially_active)
		elements[label] = new_element
		
		# Get neighbours
	return elements

def find_block_indicies(inp_part, name):
	""" Returns a list, since sometimes a part will have multiple node sub_blocks.
	"""
	indicies = []
	for i, sub_block in enumerate(inp_part.sub_blocks):
		if sub_block.keyword == name:
			indicies.append(i)
	if len(indicies) == 0:
		raise MissingSubBlock("Could not find sub-block: " + name)
	return indicies

#########################
#         Sets          #
#########################

def _regular_set(dataline):
	start 	= int(dataline[0])
	end		= int(dataline[1])+1
	step	= int(dataline[2])
	return set(range(start, end, step))

def _long_form_set(data):
	contents = set()
	for line in data:
		for item in line:
			# Skip empty items (for when the dataline ends with a comma)
			if item == "":
				continue
			
			# Skip comments
			if item[:2] == "**":
				continue
			
			contents.add(int(item))
	return contents

def _process_set(inp_sub_block):
	parameter_keys = [pair[0] for pair in inp_sub_block.parameters]
	# Regular set
	if "generate" in parameter_keys:
		return _regular_set(inp_sub_block.data[0])
	# Long form set
	else:
		return _long_form_set(inp_sub_block.data)

def get_set_type_and_label(inp_sub_block):
	for key, value in inp_sub_block.parameters:
		if key == "nset":
			return key, value
		elif key == "elset":
			return key, value

def process_sets(inp_part):
	sets = []
	
	for block in inp_part.sub_blocks:
		block_is_nodeset 	= block.keyword == "Nset"
		block_is_elementset = block.keyword == "Elset"
		if block_is_nodeset or block_is_elementset:
			set_data = _process_set(block)
			set_type, set_label = get_set_type_and_label(block)
			sets.append(geometry.flat_geometry.ItemSet(	label=set_label,
														set_type=set_type,
														data=set_data))
			
	return sets

def block_has_parameters(block):
	if type(block.parameters) == list:
		return True
	else:
		return False

def block_is_element_surface(block):
	block_is_surface = block.keyword == "Surface"
	element_surface = False
	
	# Some blocks will have None as their block.parameters values.
	# This deals with exceptions
	try:
		for key, value in block.parameters:
			if key == "type" and value.lower() == "element":
				element_surface = True
	except TypeError:
		element_surface = False
	
	return block_is_surface and element_surface

def get_surface_side(surface_block):
	side_string = surface_block.data[0][1]
	return int(side_string[1])

def get_surface_name(surface_block):
	for key, value in surface_block.parameters:
		if key == "name":
			return value
	
	raise Exception("Unnamed surface encountered.")

def process_surfaces(inp_part):
	surfaces = {}
	
	for block in inp_part.sub_blocks:
		if block_is_element_surface(block):
			side = get_surface_side(block)
			surface_name = get_surface_name(block)
			surfaces[surface_name] = side
	
	return surfaces

def process_surfaces_2(inp_part):
	""" Processes the surfaces in the part a second time. The first time only
	took the surface side, but ignored the element set.
	"""
	surfaces = {}
	
	for block in inp_part.sub_blocks:
		if block_is_element_surface(block):
			surface_parts = []
			# For this surface, create the elset/side pairs.
			for line_of_surface in block.data:
				# The dataline might be a comment.
				if line_of_surface[0][:2] == "**":
					continue
				# Make sure that there are only elset/side pairs.
				try:
					assert len(line_of_surface) == 2
				except:
					import pdb
					pdb.set_trace()
				# Collect the elset and side.
				element_set_name = line_of_surface[0]
				side = line_of_surface[1]
				part_of_surface = (element_set_name, side)
				surface_parts.append(part_of_surface)
				
			# Generate the surfaces on this part.
			surface_name = get_surface_name(block)
			
			new_surface = geometry.flat_geometry.Surface(
										name			= surface_name,
										surface_parts	= surface_parts)
			# Append this new surface to the collection.
			surfaces[surface_name] = new_surface
			
	return surfaces

def collect_elements_for_nodes(elements, nodes):
	# Find all of the node's elements.
	for element in elements.values():

		for node_label in element.nodes:
			try:
				nodes[node_label].elements.append(element.label)
			except KeyError:
				import pdb
				pdb.set_trace()
	
	#main_log.debug("Setting neighbour lists to arrays.")

	# Set to arrays.
	for node in nodes.values():
		node.elements = np.array(node.elements)	
		

def set_element_neighbours(elements, nodes):
	for node in nodes.values():
		for element_label in node.elements:
			# This element is a neighbour with every other element referenced by this node.
			for neighbour_label in node.elements:
				# The pairing only need to be done once, and never for the same element
				element_is_smaller = element_label < neighbour_label
				if element_is_smaller:
					# check that they are connected by 4 nodes
					# made node lists for the two elements
					nlist1 = elements[element_label].nodes
					nlist2 = elements[neighbour_label].nodes
					no_of_overlapping_nodes = len(set(nlist1).intersection(set(nlist2)))
					# Only neighbours if the two elements share 4 nodes 
					if no_of_overlapping_nodes == 4:
						# Set both elements as neighbours
						elements[element_label].neighbours.add(neighbour_label)
						elements[neighbour_label].neighbours.add(element_label)		
	
	# Set to arrays.
	#main_log.debug("Converting neighbours to arrays.")
	for element in elements.values():
		element.neighbours = np.array(list(element.neighbours))	

def process_part(inp_part):
	"""
	"""
	# Part
	label = get_value(inp_part.parameters, "name")
	#main_log.debug("Processing part: " + str(label))
	
	# Process nodes
	#main_log.debug("Processing nodes.")
	try:
		node_subblock_indicies = find_block_indicies(inp_part, "Node")
	# Part without discreet geometry
	except MissingSubBlock:
		return geometry.flat_geometry.Part(label, nodes=[], elements=[], sets=[])

	node_dict = {}
	for index in node_subblock_indicies:
		temp_nodes_dict = process_nodes(inp_part.sub_blocks[index])
		node_dict.update(temp_nodes_dict)
	

	
	# Process elements
	#main_log.debug("Processing elements.")
	element_subblock_indicies = find_block_indicies(inp_part, "Element")
	
	element_dict = {}
	for index in element_subblock_indicies:
		temp_element_dict = process_elements(inp_part.sub_blocks[index])
		element_dict.update(temp_element_dict)
		
	#main_log.debug("Setting element neighbours.")
	
	# Setting relationships
	#main_log.debug("Finding elements to nodes.")
	collect_elements_for_nodes(element_dict, node_dict)
	#main_log.debug("Collecting neighbours.")
	set_element_neighbours(element_dict, node_dict)
	
	# Process sets and surfaces
	#main_log.debug("Processing sets and surfaces.")
	sets = process_sets(inp_part)
	surfaces = process_surfaces(inp_part)

	new_part = geometry.flat_geometry.Part(label, node_dict, element_dict, sets)
	new_part.surfaces = surfaces
	#main_log.debug("Processing complete.\n")
	
	# Surfaces (v1.0) did not include the associated element set that went
	# with the surface. Surfaces_2 includes this set.
	surfaces_2 = process_surfaces_2(inp_part)
	new_part.surfaces_2 = surfaces_2
	
	return new_part

def process_tree(inp_text_tree):
	""" Takes the processed .inp file tree and parses to create the part geometry.
	"""

	
	part_indicies = []
	# Find all parts
	for i, block in enumerate(inp_text_tree):
		if block.keyword == "Part":
			part_indicies.append(i)
	parts = []
	# Process all of the part sublocks
	for index in part_indicies:
		parts.append(process_part(inp_text_tree[index]))

	return parts
	

if __name__ == '__main__':
	pass