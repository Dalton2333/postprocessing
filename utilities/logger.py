import logging
import sys
import os

main_path = os.getcwd()+'/../'
        
class Logger():
    def __init__(self, logname, loglevel, logger):
        '''
        Indicate logging file path, level
        Save log into the file
        '''
        
        # create a logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)
        
        # create a handler for writing log
        log_path = main_path + logname
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        
        # create a handler for output to controller
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.DEBUG)
        
        # define output format of handler
        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s')
        #log level 
        format_dict = {
           1 : logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
           2 : logging.Formatter('%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s'),
           3 : logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
           4 : logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
           5 : logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        }
        formatter = format_dict[int(loglevel)]
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        
        # add up handler to logger
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        
    
    def getlog(self):
        return self.logger

def create_main_logger():
    global main_log
    main_log = Logger(logname='main.log', loglevel=1, logger="main_log").getlog()
    return main_log

def create_abaqus_logger():
    global abaqus_log
    abaqus_log = Logger(logname='abaqus.log', loglevel=1, logger="abaqus_log").getlog()

main_log = create_main_logger()
#create_abaqus_logger()

# Calling example
#logger = Logger(logname='log.txt', loglevel=1, logger="fox").getlog()
#logger.info('fool') 
#logging.warning('%s is %d years old.', 'Tom', 10)