import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logger import logging

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message='Error occured in python script name [{0}] line number [{1}] error message'
    file_name, exc_tb.tb_lineno, str(error)
    line_number = exc_tb.tb_lineno

    return f"Error occurred in {file_name} at line {line_number}: {error_message}"

class CustomException(Exception):

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
# if __name__=='__main__':
#     try:
#         a=1/0
            
#     except Exception as e:
#         logging.error('This is a error message: ' + str(e))
#         raise CustomException(str(e), sys)
#         # print(e)