import sys
import logging

def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info() # exc_tb it contains information about the line number where the error occurred
    
    file_name = exc_tb.tb_frame.f_code.co_filename # it will give the file name where the error occurred

    error_message = "Error Occured in Python Script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, 
        exc_tb.tb_lineno, 
        str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail) ## After raise CustomException("Error Occured", sys) 
        
        #this will call the error_message_details function and pass the error message and sys module to it. It will return the error message with file name, line number and error message.

    def __str__(self):
        return self.error_message
    
    """
    Now when you do:

    After raise CustomException(e, sys)
    
    print(e)

    Python automatically calls:

    e.__str__()

    """

        