'''
This module will process the parameters
'''

import os
import math

mandatory = [   'inp_name',
                'initial_model_name',
                'part_name',
                'step_name',
                'instance_name',
                'hc_iterations',]

float_variables = ['penalty_factor',
                   'min_step_length',
                   'void_dst',]

int_variables = ['hc_group_size',
                 ]

def parse_parameters(file_text):
    """
    param: file_text:       Text of parameters file.
    type: file_text:        str
    returns:                parameters processed in a dictionary.
    rtype:                  dict
    """
    parameters = {}

    # Processing the parameters file's lines
    for i, line in enumerate(file_text.splitlines()):
        try:
            # strip comments
            # only line fragment before # is kept
            try:
                cleaned_line = line.split("#")[0]
            except:
                import pdb
                pdb.set_trace()

            # skip blank lines
            if cleaned_line == "":
                continue

            # split line between variable name and values, then strip
            # 
            cleaned_line = [a.strip().strip('"') for a in cleaned_line.split('=')]

            # parameters[name] = value
            parameters[cleaned_line[0]] = cleaned_line[1]

        # Exception is raised when error
        except IndexError:
            print("\nError in line:", line, "\n")
            print("Line NO.", i+1)
            raise

    return parameters

def get_missing_parameters(parameters):
    missing_parameters = []
    # Check each missing option
    for option in mandatory:
        if option not in parameters:
            missing_parameters.append(option)
    return missing_parameters

def mandatory_parameters_are_present(parameters):
    """ Confirms that values have been given for the mandatory parameters.
    """
    missing_parameters = get_missing_parameters(parameters)
    # Raise exception for any missing parameters.
    if len(missing_parameters) > 0:
        return False
    # Return if all is good.
    else:
        return True

def derive_from_default_parameters(parameters):
    """ Adds some values to the parameters dictionary that are derived from the mandatory parameters.
    
    :param parameters:			The input parameters.
    :type parameters:			dict
    :returns:					new_parameters --> A new dictionary with
                                the derived parameters included.
    :rtype:						dict
    """
    new_parameters = parameters.copy()
    # Get the folder from the inpFilePath
    new_parameters["model_dir"] = os.path.dirname(new_parameters["inp_path"])
    
    # Get file name root
    fullFileName = os.path.split(new_parameters["inp_path"])[1]
    fileNameRoot = os.path.splitext(fullFileName)[0]
    new_parameters["file_name_root"] = fileNameRoot
    # The first .odb file path
    new_parameters["odb_path"] = os.path.abspath(os.path.join(	new_parameters["model_dir"],
                                                new_parameters["file_name_root"] + ".odb"))
    return new_parameters
    
def convert_string_bools_to_bool(parameters):
    """ Change string bool values to bool()s.
    """
    new_parameters = parameters.copy()
    # Convert all boolean options to proper booleans
    for parameterKey in new_parameters:
        if (new_parameters[parameterKey]).lower() == "true":
            new_parameters[parameterKey] = True
        elif (new_parameters[parameterKey]).lower() == "false":
            new_parameters[parameterKey] = False
    return new_parameters

def convert_floats(parameters):
    """ Convert strings of numbers to floats.
    """
    new_parameters = parameters.copy()
    for label in float_variables:
        # Skip fixed_vr_target_value since it's not mandatory. # Temp fix.
        if label == "fixed_vr_target_value" and "fixed_vr_target_value" not in new_parameters:
            continue
        new_parameters[label] = float(new_parameters[label])
    return new_parameters

def convert_ints(parameters):
    """ Convert strings of numbers to ints.
    """
    new_parameters = parameters.copy()
    for label in int_variables:
        if label in new_parameters:
            new_parameters[label] = int(new_parameters[label])
    return new_parameters

def set_default_value(parameters, label, default_value=None, data_type="float"):
    """ Convert a default value from string, otherwise specify one.
    """
    new_parameters = parameters.copy()
    
    data_types = {	"float"	: float,
                    "int"	: int}
    
    if label not in new_parameters:
        # The default shouldn't be None, reject it.
        if default_value == None:
            raise Exception("No default value specified for a parameter.")
        # Set a default value
        new_parameters[label] = default_value
    # Convert a string to a data type
    elif data_type in data_types:
        new_parameters[label] = data_types[data_type](new_parameters[label])
    else:
        raise Exception("An incorrect data_type was specified for the default value.")
    
    return new_parameters

def set_default_false_parameters(parameters):
    new_parameters = parameters.copy()
    # Here, enter default values for parameters that are optional and have not been specified.
    for parameter in default_false:
        if parameter not in new_parameters:
            new_parameters[parameter] = False
    return new_parameters

def derive_optional_parameters(parameters):
    """ Give a default value for parameters that are optional.
    """
    new_parameters = parameters.copy()
    
    # ER
    label 		= "ER"
    default_val = 0.01
    data_type 	= "float"
    new_parameters = set_default_value(new_parameters, label, default_val, data_type)
    
    # AR
    label 		= "AR"
    default_val = 0.01
    data_type 	= "float"
    new_parameters = set_default_value(new_parameters, label, default_val, data_type)
    
    return new_parameters

    
def print_mandatory_input_parameters(parameters):
    """ Prints all of the mandatory parameters.
    """
    # Find the longest mandatory label
    max_length = max([len(parameter) for parameter in parameters])
    all_parameters = list(parameters.keys())
    # This ensures the output is alphabetic.
    all_parameters.sort()
    
    print("Input options:")
    for parameter in all_parameters:
        size_difference = max_length - len(parameter)
        print(parameter + ": " + " "*(size_difference+1) + str(parameters[parameter]))
    print("")

def deduce_constraint_types(parameters):
    """ Check the selected constraint checking method to use.
    """
    new_parameters = parameters.copy()
    
    # Give a default false to unspecified constraint checking methods.
    for constraint in constraint_types:
        if constraint not in new_parameters:
            new_parameters[constraint] = False
    
    # From here, it's possible to assume that every parameter has been specified.
    
    a_constraint_type_found = False
    # Make sure that if constraints are selected, at least one constraint method is desired
    if new_parameters["opTarget"] == "constraint":
        for constraint in constraint_types:
            # Check that the parameters were properly specified.
            if type(new_parameters[constraint]) != bool:
                raise Exception("Bad input used to select the constraint type: " + constraint)
            constraint_was_selected = new_parameters[constraint] == True
            if constraint_was_selected:
                a_constraint_type_found = True
        # Complain if no constraints were found
        if a_constraint_type_found == False:
            raise Exception("Constraint analysis selected, but no constraint type specified.")
    
    return new_parameters

def check_for_multiple_load_cases(parameters):
    new_parameters = parameters.copy()
    if new_parameters["multiple_load_cases"]:
        if "load_cases" not in new_parameters:
            raise Exception("Multiple load cases analysis selected, but load cases not specified.")
        # Evaluate the input
        new_parameters["load_cases"] = eval(new_parameters["load_cases"])
    
    return new_parameters

def set_non_standard_default_parameters(parameters):
    new_parameters = parameters.copy()
    
    for parameter, value in non_false_default.items():
        if parameter not in new_parameters:
            new_parameters[parameter] = value
    return new_parameters

def check_mandatory_dependent_variables(parameters):
    new_parameters = parameters.copy()
    
    for parent in mandatory_dependent_variables.keys():
        # Run tests if parameter is present and active (True)
        variable_present = parent in new_parameters
        variable_active = new_parameters[parent]
        if variable_present and variable_active:
            at_least_one_parameter_present = None
            not_all_parameters_are_present = None
            for child_parameter in mandatory_dependent_variables[parent][0]:
                if child_parameter in new_parameters:
                    at_least_one_parameter_present = True
                else:
                    not_all_parameters_are_present = True
            
            test_type = mandatory_dependent_variables[parent][1]
            any_test = test_type == "any"
            all_test = test_type == "all"
            
            any_test_fail = any_test and at_least_one_parameter_present != True
            all_test_fail = all_test and not_all_parameters_are_present
            if any_test_fail or all_test_fail:
                if test_type == "any":
                    number_to_specify = "at least one"
                else:
                    number_to_specify = "all" 
                raise Exception("The parameter: " + parent + " was specified, but it requires "
                                "that further parameters be also specified. Please add values for "
                                + number_to_specify + " of the parameters from the following "
                                "list: " + str(mandatory_dependent_variables[parent][0]))
    return new_parameters
            
def check_path(parameters):
    """
    Perform checks on the folder path provided.
    """
    # The path to the model folder must not contain spaces.
    if " " in parameters["inp_path"]:
        raise Exception("The input file path: " + parameters["inp_path"] +
                        " is unacceptable. The path cannot contain spaces.")
    
def process_parameters(parameters):
    new_parameters = parameters.copy()
    # An error was found for a relative inp_path. Converting to abspath here.
    new_parameters["inp_path"] = os.path.abspath(new_parameters["inp_path"])
    new_parameters = convert_string_bools_to_bool(new_parameters)
    new_parameters = convert_floats(new_parameters)
    new_parameters = convert_ints(new_parameters)

    # Derive derived parameters
    new_parameters = derive_from_default_parameters(new_parameters)

    new_parameters = set_default_false_parameters(new_parameters)
    new_parameters = derive_optional_parameters(new_parameters)
    new_parameters = deduce_constraint_types(new_parameters)
    new_parameters = check_for_multiple_load_cases(new_parameters)
    
    new_parameters = set_non_standard_default_parameters(new_parameters)
    new_parameters = check_mandatory_dependent_variables(new_parameters)
    return new_parameters

def getAnalysisParameters(path_to_analysis_parameters):
    """ Returns a properly processed parameters dictionary given a path to a file.
    
    :param path:			A string to the path of the analysis parameters.
    :type path:				str
    :returns:				parameters --> A dictionary of parameters from the input file.
    :rtype:					dict
    :raises:				Exception (Mandatory parameters missing in the input file)
    """
    # Open the file and split into lines of strings
    with open(path_to_analysis_parameters, "r") as f:
        parametersFile = f.read()
    
    parameters = parseConfigFile(parametersFile)
    # Raise exception if some of the mandatory parameters are missing.
    if mandatory_parameters_are_present(parameters) is False:
        print("A mandatory parameter is missing from the configuration file.\n"
            "Missing parameters are listed:\n")
        # List the missing parameters
        for parameter in get_missing_parameters(parameters):
            print(parameter)
        print("")
        raise Exception("Mandatory parameter(s) missing.")

    parameters = process_parameters(parameters)

    # Print the parameters for purposes of debugging
    if parameters["debug"] == True:
        print_mandatory_input_parameters(parameters)

    return parameters



