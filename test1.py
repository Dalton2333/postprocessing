'''
11/12/2018
Author@Dedao
This is the main control function for post processing
'''
import sys
import os
para_path = os.getcwd()+'/../'
sys.path.insert(0, para_path)
import parameters
import utilities.logger
import utilities.abaqus.inp_reader_v2
import utilities.abaqus.inp_tree_processor_v2
import geometry.boundary_extractor
import matplotlib.pyplot as plt

def get_desired_part(parts_list,ap):
	part_name_requested = ap['part_name']
	for part in parts_list:
		if part.label == part_name_requested:
			return part
	# This is reached if the requested part was never found.
	raise Exception("Could not find the part: " + part_name_requested +
				" as requested in the input file.")

def scatter_plot(coords_list, plt_title):
    import numpy
    tr_list = list(numpy.transpose(list(coords_list)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(tr_list[0],tr_list[1],tr_list[2], c='r', marker='.', s=[1,1,1])
    fig.title(plt_title)
    plt.show()

if __name__ == '__main__':
    list1 = [(13.0, 0.0), (12.0, 0.0), (11.0, 0.0), (10.0, 0.0), (9.0, 0.0), (8.0, 0.0), (7.0, 0.0), (6.0, 0.0), (5.0, 0.0), (4.0, 0.0), (3.0, 0.0), (2.0, 0.0), (1.0, 0.0)]
    list1.append((136,212))
    list1.sort()
    print(list1)
    # main_log = utilities.logger.main_log
    # main_log.info("Post-processing starts.")
    # ap = parameters.ap
    #
    # parsed_inp_file = utilities.abaqus.inp_reader_v2.parse_inp_file(ap['inp_path'])
    # parts_list = utilities.abaqus.inp_tree_processor_v2.process_tree(parsed_inp_file)
    # part = get_desired_part(parts_list,ap)
    # ## delete parsed parts to release memory
    # del parsed_inp_file
    # del parts_list
    #
    # (inside_boundary,outside_boundary,original_outside_boundary) = geometry.boundary_extractor.get_all_boundary_nodes(ap,part)
    # print("inside: ",inside_boundary)
    # print("outside: ",outside_boundary)
    # print("original: ",original_outside_boundary)
    # scatter_plot(outside_boundary, 'outside nodes')


