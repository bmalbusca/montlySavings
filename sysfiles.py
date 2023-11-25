# sysfiles.py
# Author: bmalbusca
# Year: 2023
# License: GNU General Public License v3.0 (GPL-3.0)
# License URL: https://github.com/bmalbusca/montlySavings/blob/master/LICENSE
# Description: This library file provides utility functions for working with files and caching data.

import os

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

def getLocalFiles(sufix="pdf", debug=False) -> dict:
    """
    Gets a list of local files with a specific suffix.

    Parameters:
    - sufix (str): The suffix of the files to look for (default is "pdf").
    - debug (bool): If True, print debug information.

    Returns:
    - dict: A dictionary with the current working directory as the key and a list of matching filenames as the value.
    """
    existing_files_list = []

    # Get the current working directory
    cwd = os.getcwd()
    dir_list = os.listdir(cwd)

    if debug:
        print(dir_list)

    for filename in dir_list:
        suffixes = filename.split(".")
        
        if debug:
            print(suffixes)

        # Check if the file has the specified suffix
        if suffixes[-1] == sufix:
            existing_files_list.append(filename)

    if debug:
        print(existing_files_list)

    return {cwd: existing_files_list}

# Arguments example: collect_cache_func=pandas.read_pickle, bool_assert_cache_func=(lambda dataframe: dataframe.empty)
def collectCacheData(collect_cache_func, bool_assert_cache_func, cache_name='cached_dataframe.pkl', debug=False):
    """
    Collects cached data from a file.

    Parameters:
    - collect_cache_func (function): The function used to collect the cache (e.g., pandas.read_pickle).
    - bool_assert_cache_func (function): A function that asserts whether the cache is valid (e.g., lambda dataframe: dataframe.empty).
    - cache_name (str): The name of the cache file (default is 'cached_dataframe.pkl').
    - debug (bool): If True, print debug information.

    Returns:
    - DataFrame or None: The cached data if valid; otherwise, None.
    """
    try:
        # Read the DataFrame from the cache
        read_cache = collect_cache_func(cache_name)

        if debug:
            print("[LOAD]: Loading cache data", read_cache)

        # Check if the cache is valid using the provided assertion function
        if not bool_assert_cache_func(read_cache):
            return read_cache
    except Exception as e:
        if debug:
            print("[ALERT]: Error loading cache data:", e)

    return None
