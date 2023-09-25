from enum import Enum

# Define an enumeration called 'QuestionType'
class QuestionType(Enum):
    ATPG_General = 0 # General questions related to ATPG
    Tessent_Commands = 1 # Questions related to Tessent command usage
    Tessent_DRC = 2 # Questions related to Tessent DRC
    # ----
    Unknown = 99
