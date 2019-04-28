'''
Created on 27/09/2013

@author: Daniel Stojanov
'''
import shlex

import utilities.logger
main_log = utilities.logger.main_log

# These are blocks that end with a "*End something" command.
super_blocks = ["Part",
				"Instance",
				"Assembly",
				"Step"]


newline_token = "`"

class inp_file(object):
	pass

class inp(object):
	def __init__(self):
		self.parameters = {}
		
	def __repr__(self):
		return "'" + str(self.keyword) + ": " + str(self.parameters) + "'"
		
class inp_super_block(inp):
	pass
class inp_block(inp):
	pass

#########################
# Line type determiners #
#########################

def strip_start_of_file(stream):
	""" Strip tokens until the first * is found.
	
	:param stream:	shlex token stream.
	:type stream:	shlex.shlex
	:returns:		new_stream
	:rtype:			shlex.shlex
	"""
	for token in stream:
		if token == "*":
			stream.push_token(token)
			break
	# Add the newline token
	stream.push_token(newline_token)
	return stream

def next_line_is_comment(stream):
	""" Given a stream, determine if the next bit of content is a comment.
	
	Assumes that the stream begins with the newline_token.
	
	:param stream:	shlex token stream.
	:type stream:	shlex.shlex
	:returns:		is_comment, stream
	:rtype:			(bool, shlex.shlex)
	"""
	is_comment = None
	# Pop newline token
	next_token_0 = stream.get_token()
	next_token_1 = stream.get_token()
	# Empty line
	if next_token_1 != "*":
		is_comment = False
	else:
		# Try the second token.
		next_token_2 = stream.get_token()
		# Is a comment
		if next_token_2 == "*":
			is_comment = True
		else:
			is_comment = False
		# Push token 2 back.
		stream.push_token(next_token_2)
	# Push token 1 back
	stream.push_token(next_token_1)
	# Push back the newline_token
	stream.push_token(next_token_0)
	
	# Just to be sure, a comment should be confirmed one way or another.
	if is_comment == None:
		raise Exception("There was an error determining whether tokens were in a comment.")

	return is_comment, stream

def prepend_newline_tokens(file_body):
	""" Prepends the newline_token to every line in the text file. 
	
	:param file_body:	Text string
	:type file_body:	str
	:returns:			Text string
	:rtype:				str
	"""
	file_lines = file_body.splitlines()
	new_file_lines = [None]*len(file_lines)
	for i, line in enumerate(file_lines):
		new_file_lines[i] = newline_token+line
	return "\n".join(new_file_lines)

def trim_empty_lines(file_body):
	file_lines = file_body.splitlines()
	while file_lines[-1] == "`":
		file_lines.pop(-1)
	return "\n".join(file_lines)

def remove_comment(stream):
	""" Remove the tokens until the end of the comment is reached.
	
	Assumes the comment *is* there.
	
	:param stream:	A token stream.
	:type stream:	shlex.shlex
	:returns:		new_stream
	:rtype:			shlex.shlex
	"""
	stream.get_token()
	for token in stream:
		if token == "`":
			stream.push_token(token)
			break
		# End of file
		elif token == "":
			break
	return stream

def remove_comments_until_content(stream):
	""" Removes all comments until actual file content is encountered.
	
	:param stream:	A token stream.
	:type stream:	shlex.shlex
	:returns:		new_stream
	:rtype:			shlex.shlex
	"""
	more_comments, stream = next_line_is_comment(stream)
	while more_comments:
		stream = remove_comment(stream)
		more_comments, stream = next_line_is_comment(stream)
	
	return stream
	

def get_grouped_parameters(parameter_tokens):
	""" Takes a stream of tokens and separates them by the comma separator.
	
	:param parameter_tokens:	A token stream.
	:type parameter_tokens:		list
	:returns:					grouped_tokens
	:rtype:						list
	"""
	grouped_tokens = [[]]
	first_run = True
	for token in parameter_tokens:
		# The first run edge case must get rid of the empty internal list if
		# the first token is a comma
		if first_run == True and token == ",":
			grouped_tokens.pop()
		first_run = False
		
		if token == ",":
			grouped_tokens.append([])
		else:
			grouped_tokens[-1].append(token)
	return grouped_tokens

def get_key_value_from_tokens(tokens):
	""" Converts a list of tokens into a single key/value pair.
	
	:param tokens:	The tokens, as strings.
	:type tokens:	[str]
	:returns:		(key, value)
	:rtype:			(string, string)
	"""
	key_tokens 			= []
	value_tokens 		= []
	found_equals_sign 	= False
	
	for token in tokens:
		# Mark and skip the equals sign.
		if token == "=":
			found_equals_sign = True
			continue
		if not found_equals_sign:
			key_tokens.append(token)
		else:
			value_tokens.append(token)
	
	# Combine the tokens into a string
	if len(key_tokens) == 0:
		key = None
	else:
		key	= "".join(key_tokens)
		
	if len(value_tokens) == 0:
		value = None
	else:
		value = "".join(value_tokens)
	return (key, value)  
	
def get_line_of_tokens(stream):
	""" Returns all of the tokens until the end of the line.
	
	Assumes that the newline token indicates the start of the next line.
	The next line will begin after the newline_token.
	
	(redo tests)
	
	:param stream: 	A token stream.
	:type stream:	shlex.shlex
	:returns:		Tokens from the first line, and the rest of the stream.
	:rtype:			(tokens, stream)
	"""
	token = stream.get_token()
	tokens = []
	
	break_loop = False
	while break_loop == False:
		tokens.append(token)
		token = stream.get_token()
		
		# It needs to be like this an *not* a while loop because of some moronic bug.
		if token == "`":
			break_loop = True
		elif token == "":
			break_loop = True
	

	return tokens, stream

def process_parameters(stream):
	""" Process the parameters that are inline with the block heading
	
	:param stream:	A token stream.
	:type stream:	shlex.shlex
	:returns:		processed_parameters, stream
	:rtype:			[(key, value),...], stream
	"""
	parameter_tokens, stream = get_line_of_tokens(stream)

	grouped_parameters = get_grouped_parameters(parameter_tokens)
	
	# Generate key/value tuples
	processed_parameters = []
	for parameter in grouped_parameters:
		processed_parameters.append(get_key_value_from_tokens(parameter))
	
	return processed_parameters, stream

def does_block_have_parameters(stream):
	""" Given a stream, do parameters follow.
	
	Assumes the stream comes just after the keyword and is on the same line
	
	Will return a stream ready for parameter processing or body/new block processing.
	"""
	parameters_present = None
	
	# The next token should be either a comma or the next line.
	token_1 = stream.get_token()
	
	# Has popping the token caused the lexer to move to the next line.
	if token_1 == newline_token:
		parameters_present = False
	# At EOF
	elif token_1 == "":
		parameters_present = False
	elif token_1 != ",":
		import pdb
		pdb.set_trace()
		raise Exception("Unexpected input found on line: " + str(stream.lineno))
	else:
		# Another token on the same line indicates parameters.
		token_2 = stream.get_token()
		if token_2 != newline_token:
			parameters_present = True
		else:
			parameters_present = False 
		stream.push_token(token_2)
	return parameters_present, stream

def next_line_is_data(stream):
	""" Determine if the next line in the stream is data.
	
	First token must be first after newline token.
	"""
	is_data = None
	# Next token
	token_1 = stream.get_token()
	# At EOF
	if token_1 == "":
		return False, stream
	# Maybe not data
	elif token_1 == "*":
		# The data could be a comment or heading.
		token_2 = stream.get_token()
		# Comment
		if token_2 == "*":
			is_data = True
		else:
			is_data = False
		stream.push_token(token_2)
	else:
		is_data = True
		
	stream.push_token(token_1)
		
	return is_data, stream
	
	


# def does_block_have_body(stream):
# 	""" Given a stream, determine if the block body has data.
# 	
# 	Comments count as data
# 	"""
# 	current_line = stream.lineno
# 	content = True
# 	next_token_1 = stream.get_token()
# 	
# 	# Star indicates comment or block
# 	if next_token_1 == "*":
# 		content = False
# 	return content, stream

# def data_content_type(stream):
# 	""" Types: empty, data, comment, block
# 	"""
# 	content_type = None
# 	
# 	token_1 = stream.get_token()
# 	# End of file
# 	if token_1 == "":
# 		return "empty"
# 	# New block or comment
# 	elif token_1 == "*":
# 		token_2 = stream.get_token()
# 		# Content is a line of *
# 		if token_2 == "`":
# 			content_type = "data"
# 		else:
# 			token_3 = stream.get_token()
# 			# Comment
# 			if token_3 == "*":
# 				content_type = "comment"
# 			else:
# 				content_type = "block"
# 			stream.push_token(token_3)
# 		stream.push_token(token_2)
# 	stream.push_token(token_1)
# 	
# 	return content_type, stream

def group_data_line_tokens(data_tokens):
	token_lists = [[]]
	
	for token in data_tokens:
		if token == ",":
			token_lists.append([])
		else:
			token_lists[-1].append(token)
	return token_lists
			
def process_data_line(data_tokens):
	token_lists = group_data_line_tokens(data_tokens)
	
	data_items = []
	for tokens in token_lists:
		data_items.append("".join(tokens))
	return data_items

def process_data(stream):
	""" Process the data in a block. It is OK to send a block without any data.
	
	Comments are processed as regular data, split by commas.
	"""
	block_data = []
	next_line_data, stream = next_line_is_data(stream)
	
	while next_line_data:
		token_line, stream = get_line_of_tokens(stream)
		# Process line
		line = process_data_line(token_line)
		block_data.append(line)
		
		next_line_data, stream = next_line_is_data(stream)

	return block_data, stream 

def generate_block(keyword, parameters, data, sub_blocks=None):
	"""
	"""
	block = inp_block()
	block.keyword = keyword
	block.parameters = parameters
	block.data = data
	return block
	

def process_block(stream):
	""" Processes the block.
	
	Assumption: The first token is the block's keyword.
	:param stream:	A token stream.
	:type stream:	shlex.shlex
	:returns:		
	:rtype:			
	"""

	# Newline token
	stream.get_token()
	# Star
	stream.get_token()
	
	block_keyword = stream.get_token()
	print("Block:", block_keyword)

	# Check for double word keywords
	at_end_of_keyword = False
	while not at_end_of_keyword:
		
		next_token = stream.get_token()
	
		# Test cases for end of keyword
		following_comma = next_token == ","
		end_of_line 	= next_token == newline_token
		end_of_file 	= next_token == ""
		if not following_comma and not end_of_line and not end_of_file:
			# Append the next token to the block keyword
			block_keyword += " " + next_token
		else:
			# Put back
			stream.push_token(next_token)
			at_end_of_keyword = True
	
	#main_log.debug(block_keyword)
	#if block_keyword == "Contact output":
	#	import pdb
	#	pdb.set_trace()
	
	parameters_present, stream = does_block_have_parameters(stream)
	

	
	if parameters_present:
		block_parameters, stream = process_parameters(stream)
	else:
		block_parameters = None
	##.
	block_data, stream = process_data(stream) 
	
	# Append newline, unless at EOF after completing data processing
	next_token = stream.get_token()
	if next_token != "":
		stream.push_token(next_token)
		stream.push_token(newline_token)
	
	block = generate_block(keyword=block_keyword,
						parameters=block_parameters,
						data=block_data)

	return block, stream
	
	
	
	
def reached_eof(stream):
	"""
	"""
	token = stream.get_token()
	if token == "":
		return True
	else:
		stream.push_token(token)
		return False

def group_superblocks(stream):
	pass

def parse_abaqus_stream(stream):
	""" Takes the stream of tokens at the lowest level of the file.
	"""
	stream = strip_start_of_file(stream)

	# Remove comments before block
	stream = remove_comments_until_content(stream)
	
	at_eof = reached_eof(stream)
	blocks = []
	# Process all blocks until EOF
	while not at_eof:
		block, stream = process_block(stream)
		blocks.append(block)
		at_eof = reached_eof(stream)
	
	return blocks

def get_inp_file_blocks(file_location):
	# Open and read the file
	
	with open(file_location, "r") as f:
		file_bytes = f.read()
	file_body = file_bytes
	#file_body = file_bytes.decode("ascii")
	
	# Strip commas at end of line
	file_body = prepend_newline_tokens(file_body)
	# Trim empty lines near EOF
	file_body = trim_empty_lines(file_body)
	
	stream = shlex.shlex(file_body)

	processed_inp_file = parse_abaqus_stream(stream)
	return processed_inp_file

def process_super_blocks(block_list, block_iterator):
	"""
	"""
	blocks = []
	
	for block in block_iterator:
		# Push superblock keywords.
		if block.keyword in super_blocks:
			blocks.append(block)
			sub_blocks = process_super_blocks(block_list, block_iterator)
			blocks[-1].sub_blocks = sub_blocks
		# Pop from stack
		elif block.keyword[:3] == "End":
			break
		else:
			blocks.append(block)

	return blocks

def parse_inp_file(location):
	block_list = get_inp_file_blocks(location)
	block_iter = block_list.__iter__()
	sorted_blocks = process_super_blocks(block_list, block_iter)
	return sorted_blocks
	
if __name__ == "__main__":
	file_location = "C:/abs/code_pack/beso/testData/modelTesting-old.inp"
	
	block_list = get_inp_file_blocks(file_location)
	block_iter = block_list.__iter__()
	sorted_blocks = process_super_blocks(block_list, block_iter)
	
	main_log.debug("done")
	import pdb
	pdb.set_trace()

