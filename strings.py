# strings.py
# Author: bmalbusca
# Year: 2024
# License: GNU General Public License v3.0 (GPL-3.0)
# License URL: https://github.com/bmalbusca/montlySavings/blob/master/LICENSE
# Description: This library file provides utility functions for working with strings (str)

def convert_inner_lists_of_list_to_strings(transfer_types_data):
    """Convert inner arrays to single strings in the original dictionary."""
    for key, value in transfer_types_data.items():
        converted_values = [' '.join(inner_array) for inner_array in value]
        transfer_types_data[key] = converted_values

    return transfer_types_data


def levenshtein_distance(str1, str2):
    m, n = len(str1), len(str2)
    
    # Initialize the matrix
    matrix = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill in the first row and first column
    for i in range(m + 1):
        matrix[i][0] = i
    for j in range(n + 1):
        matrix[0][j] = j
    
    # Fill in the rest of the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,      # Deletion
                              matrix[i][j - 1] + 1,      # Insertion
                              matrix[i - 1][j - 1] + cost)  # Substitution
    
    # The Levenshtein distance is the value in the bottom-right cell
    return matrix[m][n]

def str2num(x: str) -> float:
    """
    Converts a string to a floating-point number.

    Parameters:
    - x (str): Input string.

    Returns:
    - float: Converted floating-point number.
    """
    try:
        return float(x.split()[0].replace('.', '').replace(',', '.'))
    except:
        # Handle the exception by returning the original string
        return x