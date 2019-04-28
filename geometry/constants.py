'''
Created on 16/05/2013

@author: Daniel Stojanov
'''
import numpy as np

nodeNeighboursList = \
{
	#3D Shapes
	"HEX":
	[(0,1), (0,3), (0,4), (1,2), (1,5), (2,3),
	(2,6), (3,7), (4,5), (4,7), (5,6), (6,7)],
	
	# 2D Shapes
	"TRI":
	[(0,1), (0,2), (1,2)],

	"QUAD":
	[(0,1), (0,3), (1,2), (2,3)]
}

element_label_map = \
{
	"C3D8"	: "HEX",
	"C3D8R"	: "HEX",
	"R3D4"	: "QUAD",
	"R3D3"	: "TRI",
	"CPS4"	: "QUAD",
	"CPS4R"	: "QUAD",
	"CPS3"	: "TRI",
	"CAX4R"	: "QUAD"
}

# For backward compatibility.
elementLabelMap = element_label_map

# All mappings below this comment will correspond to that nomenclature

nodes_on_side =\
{
	# This is the mapping that comes from Section 27.1.4 of the ABAQUS User manual
	# sideNo: [nodes that are on the corner of that side]
	"HEX":
	{
		1: [1, 2, 3, 4],
		2: [5, 6, 7, 8],
		3: [1, 2, 5, 6],
		4: [2, 3, 6, 7],
		5: [3, 4, 7, 8],
		6: [5, 6, 7, 8]
	},
	
	"QUAD":
	{
		1: [1, 2],
		2: [2, 3],
		3: [3, 4],
		4: [4, 1]
	}
}

# # A neighbour on side n will have the same node label in the True positions of its node list.
# neighbour_matching = \
# {
# 
# 	# Side 1 meets side 2
# 	1: np.array([], dtype=bool)
# 
# }

# These Match Section 27.1.4 for an 8 node hexahedral element.
nodes_on_a_side_hex = {	1: np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
						2: np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool), 
						3: np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=bool), 
						4: np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=bool),
						5: np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=bool),
						6: np.array([1, 0, 0, 1, 1, 0, 0, 1], dtype=bool)}

opposite_surface = {1: 2,
					2: 1,
					3: 5,
					4: 6,
					5: 3,
					6: 4}



