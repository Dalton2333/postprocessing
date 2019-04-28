#!/usr/bin/env python3
'''
Created on 09/10/2013

@author: Daniel Stojanov
'''
print("STARTED\n")

import os
import json

analysis_parameters_location = "analysis_parameters.txt"
RevNumber = "4.4"

# Setup analysis parameters before anything else
import utilities.inputs
import utilities.log_setup

if __name__ == "__main__":
	# ap is a dictionary of parameters controlling the optimisation process
	ap = utilities.inputs.getAnalysisParameters(analysis_parameters_location)
	# Setup logger
	utilities.log_setup.configure_loggers(ap["model_dir"])
	main_log = utilities.log_setup.get_main_logger()
	main_log.info("Optimisation run using Version "+RevNumber)
	main_log.debug("Optimisation parameters processed.")
	
	try:
		k_history = float(ap["k_history"])
	except:
		main_log.info("k value for history based sensitivity not detected in configuration file.")
		ap["k_history"] = 0.0
		k_history = 0.0
	
	if k_history > 1.0:
		main_log.info("k value for history based sensitivity is: "+ap["k_history"]+", which is higher than the recommended maximum of 1.0")
	elif k_history > 0.0:
		main_log.info("k value for history based sensitivity is: "+ap["k_history"])
	elif k_history == 0.0:
		main_log.info("No history effect")
	else:
		raise Exception("Incompatible k value of ("+k_history+") for history effects on sensitivity function. ")

import utilities.abaqus.inp_file
import utilities.solver
import utilities.abaqus.odbExtraction
import utilities.abaqus.inp_file_gen
import optimisation.slice_method
import optimisation.strain_energy_sensitivity
import optimisation.sensitivity_by_energy_type
import optimisation.sensitivity_by_stress_value
import optimisation.sensitivity_filters
import optimisation.axial_rank
import optimisation.volume_manager_2
import optimisation.void_control
#import optimisation.surface_removal

optimisation_methods = [
						"slice_method"
						]
constraint_methods = [
					"a_star_method"
					]

root_code_dir = os.path.split(os.path.abspath(__file__))[0]

def launch_fem_analysis(ap, iteration_no):
	model_directory 	= ap["model_dir"]
	file_name_root		= ap["file_name_root"]
	double_precision 	= ap["double_precision"]
	platform			= ap["platform"]
	cpus				= ap["cpus"]
	
	# Create the new file_name
	file_name = file_name_root + "-" + str(iteration_no)
	
	utilities.solver.run_analysis(	model_directory,
									file_name,
									double_precision,
									platform,
									cpus,
									iteration_no)
	
def read_fem_output(ap, data_types_required, iteration_no):
	"""
	"""
	global root_code_dir
	odb_module = utilities.abaqus.odbExtraction
	# Determine the new .odb file path.
	odb_path = os.path.join(ap["model_dir"], ap["file_name_root"] + "-" + str(iteration_no) + ".odb")
	
	extraction_parameters = odb_module.get_options_for_extractor(
											ap,
											odb_path,
											root_code_dir,
											fields_required = data_types_required)
	
	output_iterator = odb_module.get_output_data(extraction_parameters, iteration_no)
	return output_iterator

def calculate_element_sensitivity_values(
									previous_design,
									fem_output_data,
									part_geometry,
									ap):
	# Calculate values.
	if ap["type_select_method"]:
		# check for invalid options
		if "energy_type" in ap and "sensitivity_parameter" in ap:
			raise Exception("Both energy_type and sensitivity_parameter have been selected")
		if "energy_type" in ap:
			sensitivity_array = optimisation.sensitivity_by_energy_type.get_element_sensitivity(part_geometry, fem_output_data, ap)
		elif "sensitivity_parameter" in ap:
			sensitivity_array = optimisation.sensitivity_by_stress_value.get_element_sensitivity(part_geometry, fem_output_data, ap)
		else:
			raise Exception("Invalid choice of sensitivity have been selected")
	elif ap["slice_method"]:
		# Default to strain energy.
		sensitivity_array = optimisation.strain_energy_sensitivity.get_element_sensitivity(part_geometry, fem_output_data, ap)
	else:
		raise Exception("No valid optimisation method was specified.")
	
	if ap["beso_filtering"] == True:
		# filter_sensitivity_values
		filtered_sensitivity = optimisation.sensitivity_filters.beso_filter(
								pre_sensitivities	= sensitivity_array,
								radius				= ap["beso_r_parameter"],
								previous_design		= previous_design,
								part_geometry		= part_geometry)
	else:
		# Just use raw values without BESO filtering. 
		vals = {}
		for label in part_geometry.elements.keys():
			vals[label] = sensitivity_array[label]
			
		filtered_sensitivity = vals 
	
	return filtered_sensitivity, sensitivity_array

def get_sensitivity_from_archive(**kwargs):
	iteration 	= kwargs["iteration_no"]
	ap			= kwargs["ap"]
	
	vr_dir = os.path.join(ap["model_dir"], "archive", "v_ratio")
	post_path = os.path.join(vr_dir, "post-sensitivity-"+str(iteration-1)+".json")
	
	with open(post_path, "r") as f:
		old_sensitivity = json.load(f)
	
	return old_sensitivity

def archive_iteration_results(**kwargs):
	iteration = kwargs["iteration_no"]
	
	# Create the directory that has archives, if it doesn't already exist.
	vr_dir = os.path.join(ap["model_dir"], "archive", "v_ratio")
	if not os.path.exists(vr_dir):
		os.makedirs(vr_dir)
	
	# Archive the volume ratio.
	volume = kwargs["part_vr"]
	with open(os.path.join(vr_dir, "vr.txt"), "a") as f:
		f.write(str(iteration) + ": " + str(volume))
		f.write("\n")
	
	# Archive the sensitivity values.
	# NOTE: pre_sensitivity is an array, element 0 is not used.
	#		element_sensitivity is a dictionary, indexed by element label.
	pre_filter_sensitivity = kwargs["pre_sensitivities"]
	post_filter_sensitivity = kwargs["element_sensitivity"]
	
	# Archive sensitivity numbers.
	pre_path = os.path.join(vr_dir, "pre-sensitivity-"+str(iteration)+".json")
	post_path = os.path.join(vr_dir, "post-sensitivity-"+str(iteration)+".json")
	group_sen_path = os.path.join(vr_dir, "group-sensitivity-"+str(iteration)+".json")
	ranked_group_path = os.path.join(vr_dir, "ranked_group-sensitivity-"+str(iteration)+".json")
	
	# Convert the sensitivity to a dict format.
	pre_sen = dict((label, sens) for label, sens in zip(range(len(pre_filter_sensitivity)), pre_filter_sensitivity))
	# Remove the 0 element placeholder.
	del(pre_sen[0])
	
	post_sen = post_filter_sensitivity

	with open(pre_path, "w") as f:
		f.write(json.dumps(pre_sen))
	
	with open(post_path, "w") as f:
		f.write(json.dumps(post_sen))
		
	with open(group_sen_path, "w") as f:
		f.write(json.dumps(kwargs["group_sensitivities"]))
	
	# The ranked groups are numpy integers, need to be converted to ints.
	new_ranked_groups = []
	for group in kwargs["ranked_groups"]:
		new_ranked_groups.append([int(number) for number in group])
	
	with open(ranked_group_path, "w") as f:
		f.write(json.dumps(new_ranked_groups))
		

def optimisation_loop(ap, part_geometry, previous_design, data_types_required, iteration_no):
	launch_fem_analysis(ap, iteration_no)
	# Collect FEM output data.
	fem_output_data = read_fem_output(ap, data_types_required, iteration_no)

	# Calculate sensitivity values
	element_sensitivity, pre_sensitivities = calculate_element_sensitivity_values(
									previous_design,
									fem_output_data,
									part_geometry,
									ap)
	# Here pre_sensitivities are the sensitivity values before filtering. The
	# array is addressed pre[i] for element labelled i.
	# element_sensitivity are the filtered values. A dictionary addressed just
	# like the pre array.

	##############################################
	# Archive individual sensitivity values here #
	##############################################
	# This function performed under archive_iteration_results() later in this function
	
	try:
		k_history = float(ap["k_history"])
	except:
		main_log.info("k value for history based sensitivity not detected in configuration file.")
		ap["k_history"] = 0.0
		k_history = 0.0
		
	if iteration_no > 0 and k_history > 0.0:
		# Add history effect
		main_log.info("Processing history effect...")
		old_sensitivity = get_sensitivity_from_archive(	ap				= ap, 
														iteration_no	= iteration_no)
		for element_label in element_sensitivity:
			element_sensitivity[element_label] = ( element_sensitivity[element_label] + k_history * old_sensitivity[str(element_label)] ) / (1 + k_history)
		
	
	# Rank elements
	(ranked_groups, group_sensitivities) = optimisation.axial_rank.simple_method(
									element_sensitivities	= element_sensitivity,
									part_geometry			= part_geometry,
									ap						= ap)
	########################
	# Post ranking options #
	########################
	###################################################################################
	# Place here any last minute algorithms about excluding elements or similar here. #
	###################################################################################

	""" The ranked_groups tell you which elements are in each group, so group
	3 has all of its elements in ranked_groups[3]. The group sensitivities are
	the average sensitivity value of that group. For example, the same group
	has a sensitivity accessed by group_sensitivities[3].
	"""
	
	""" The following are optional filters that can each be selected in the
	analysis parameters.
	"""
	if ap["void_control"]:
		# This is the old void control. Void control version 2 occurs in the
		# process of generating the new design.
		ranked_groups, group_sensitivities = optimisation.void_control.adjust_sensitivity_groups(
									ranked_groups		= ranked_groups,
									group_sensitivities	= group_sensitivities,
									previous_design		= previous_design,
									fem_output_data		= fem_output_data,
									part_geometry		= part_geometry,
									ap					= ap)
	
	# Surface removal forces the addition/removal of elements to occur along
	# a surface, thereby avoiding internal voids.
# 	if ap["surface_removal"]:
# 		ranked_groups, group_sensitivities = optimisation.surface_removal.enforce_surface_removal(
# 									ranked_groups		= ranked_groups,
# 									previous_design		= previous_design,
# 									group_sensitivities	= group_sensitivities,
# 									part_geometry		= part_geometry,
# 									ap					= ap)
	
	# Get the new design based on the new sensitivity numbers.
	# This includes connectivity control code_pack.
	part_design, part_vr = optimisation.volume_manager_2.get_new_design1(	
									ranked_groups,
									group_sensitivities,
									iteration_no,
									previous_design,
									fem_output_data,
									part_geometry,
									ap)
	
	# Clean up the output data
	fem_output_data.cleanUp()
	
	# Adjust the part design
	if ap["void_control"]:
		optimisation.void_control.force_void_connectivity(
									part_design			= part_design,
									part_vr				= part_vr,
									ranked_groups		= ranked_groups,
									group_sensitivities	= group_sensitivities,
									part_geometry		= part_geometry,
									ap					= ap,
									iteration_no		= iteration_no)
	
	

	# TODO: Archive design and vr
	archive_iteration_results(		part_design			= part_design,
									part_vr				= part_vr,
									pre_sensitivities	= pre_sensitivities,
									element_sensitivity	= element_sensitivity,
									ranked_groups		= ranked_groups,
									group_sensitivities	= group_sensitivities,
									iteration_no		= iteration_no)
	
	# Generate new .inp file
	utilities.abaqus.inp_file_gen.generate_inp_file(
									design 			= part_design,
									iteration_no 	= iteration_no+1,
									part_geometry 	= part_geometry,
									ap 				= ap)
	return part_design

def get_selected_methods(ap, method_list):
	"""
	Note that currently it is assumed that only one optimisation method
	is being requested. The return is still a list with one method.
	"""
	selected_method = None
	for method in method_list:
		if ap[method] == True:
			selected_method = method
	
	return [selected_method]


def get_initial_design(ap, part_geometry):
	# Create initial full design
	element_label_iter = part_geometry.elements.keys()
	initial_design = dict(zip(element_label_iter, [True]*len(element_label_iter)))
	
	if ap["continue_previous_analysis"]:
		for element in part_geometry.elements.values():
			initial_design[element.label] = element.initially_active
	
	if ap["initial_delete_set"]:
		delete_set_name = ap["delete_set_name"]
		initial_delete_set = part_geometry.sets[delete_set_name].data
		
		# Set elements off
		update_tuples = dict(zip(initial_delete_set, [False]*len(initial_delete_set)))
		# Update dict
		initial_design.update(update_tuples)
	
	return initial_design

def get_desired_part(parts_list, ap):
	part_name_requested = ap["part_name"]
	for part in parts_list:
		if part.label == part_name_requested:
			return part
	
	# This is reached if the requested part was never found.
	raise Exception("Could not find the part: " + part_name_requested +
				" as requested in the input file.")
	
def _sanity_checks(part_geometry, ap):
	"""
	These are checks that are run on the part *after* it has been passed, but
	before the optimisation loop has begun.
	
	These functions must raise their own Exceptions if they find a problem.
	"""
	
	optimisation.volume_manager_2.__model_check__(part_geometry, ap)
	if ap['void_control'] or ap['void_control_v2'] or ap['void_connectivity']:
		optimisation.void_control_v2.__model_check__(part_geometry, ap)
	

if __name__ == '__main__':
	main_log.info("Optimisation analysis started for .inp file at: " + ap["inp_path"] +
				" for part named: " + ap["part_name"])
	main_log.debug("Parsing ABAQUS input file: Building .inp file syntax tree.")
	parsed_inp_file = utilities.abaqus.inp_file.parse_inp_file(ap["inp_path"])
	main_log.debug("Processing the syntax tree.")
	print("")
	parts_list = utilities.abaqus.inp_file.process_parsed_inp_tree(parsed_inp_file)
	# Delete the syntax tree. It doesn't get used again and can sometimes
	# use a lot of memory.
	del parsed_inp_file
	part_geometry = get_desired_part(parts_list, ap)
	
	# Run model sanity checks.
	_sanity_checks(part_geometry, ap)
	
	# Determine optimisation and constraint modules selected by the user.
	selected_methods 		= get_selected_methods(ap, optimisation_methods)
	selected_constraints 	= get_selected_methods(ap, constraint_methods)
	
	# Poll methods for required data
	data_types_required = []
	# This may be updated later to allow a user to select from a range of options, then
	# collect the data types based on the modules that will end up being used.
	if ap["slice_method"]:
		data_types_required += optimisation.slice_method.data_types_required(ap)
		# print(data_types_required)
	elif ap["type_select_method"]:
		if "energy_type" in ap and "sensitivity_parameter" in ap:
			raise Exception("Both energy_type and sensitivity_parameter have been selected")
		if "energy_type" in ap:
			data_types_required += optimisation.sensitivity_by_energy_type.data_types_required(ap)
			# print(data_types_required)
		elif "sensitivity_parameter" in ap:
			#===================================================================
			# import pdb
			# pdb.set_trace()
			#===================================================================
			data_types_required += optimisation.sensitivity_by_stress_value.data_types_required(ap)
			# print(data_types_required)
		else:
			raise Exception("Invalid choice of sensitivity have been selected")
	else:
		raise Exception("No valid optimisation method selected.")
	
	#===========================================================================
	# # check for STATUS for use in connectivity
	# if ap["connectivity_constraint"]:
	# 	if ["STATUS"] not in data_types_required:
	# 		data_types_required += ["STATUS"]
	# 		# print(data_types_required)
	# 	else:
	# 		pass
	#===========================================================================
	if ap["stress_constraint"]:
		if ["S"] not in data_types_required:
			data_types_required += ["S"]
			# print(data_types_required)
		else:
			pass
	
	#===========================================================================
	# import pdb
	# pdb.set_trace()
	#===========================================================================
	
	main_log.debug("Initialising optimisation module. These may take some time.")
	# Initialise the optimisation method
	optimisation.slice_method.initialise()
	# Initial pre-processing for sensitivity filtering.
	optimisation.sensitivity_filters.initialise(part_geometry, ap["beso_r_parameter"])

	# Initialise the input file for first run.
	initial_design = get_initial_design(ap, part_geometry)
	initial_volume_ratio = optimisation.volume_manager_2.get_volume_ratio_of_design(
								design			= initial_design,
								part_geometry	= part_geometry)
	
	optimisation.volume_manager_2.set_current_volume_ratio(ratio=initial_volume_ratio)
	# Create the first .inp file.
	utilities.abaqus.inp_file_gen.generate_inp_file(
								design			= initial_design,
								iteration_no	= 0,
								part_geometry	= part_geometry,
								ap				= ap)
	
	total_iterations = ap["no_of_iterations"]

	previous_design = initial_design
	# Loop through the optimisation
	for iteration_no in range(total_iterations):
		previous_design = optimisation_loop(
								iteration_no		= iteration_no,
								data_types_required	= data_types_required,
								previous_design		= previous_design,
								part_geometry		= part_geometry,
								ap					= ap)
	



