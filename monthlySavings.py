# monthlySavings.py
# Author: bmalbusca
# Year: 2023
# License: GNU General Public License v3.0 (GPL-3.0)
# License URL: https://github.com/bmalbusca/montlySavings/blob/master/LICENSE

import sys
import datetime
import pandas
import tabula
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from sysfiles import getLocalFiles, collectCacheData
from plotdata import plotOverview_EI, plotSimple_EI
from configstructures import categories_hierarchy, pdf_columns, df_columns_translator
from strings import str2num, convert_inner_lists_of_list_to_strings, levenshtein_distance

def processDirectoryFiles(file_list, read_file_func):
    """
    Process files in the specified directories.

    Args:
        file_list (dict): A dictionary containing directory names as keys and lists of file names as values.
        read_file_func (function): A function to read and process a file.

    Returns:
        pandas.DataFrame: Concatenated DataFrame of processed files.
        int: Total count of processed files.
    """
    df = pandas.DataFrame()
    count = 0
    for fdir in list(file_list.keys()):
        for filename in file_list[fdir]:
            complete_dir = fdir + '/' + filename
            s = read_file_func(complete_dir)[0]
            df = pandas.concat([df, s])
            count += 1
    return df, count

def parseDataframe(df, columns={"Balance":"Saldo","Value":"Valor.1", "Date":"Mov", "Info":"Descritivo do Movimento", "Group":"Valor", "Filename":"Filename"}):
    """
    Parse the DataFrame to prepare it for further analysis.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        columns (dict): Dictionary mapping column names.

    Returns:
        pandas.DataFrame: Processed DataFrame.
    """
    try:
        # Drop rows with NaN in 'Saldo' and 'Valor.1' columns
        df = df.dropna(subset=[columns["Balance"], columns["Value"]]).reset_index(drop=True)
        # Apply str2num function to 'Saldo' and 'Valor.1' columns
        df[[columns["Balance"], columns["Value"]]] = df[[columns["Balance"], columns["Value"]]].applymap(str2num)
        # Combine 'Mov' with the predefined year and convert to datetime
        df[columns["Date"]] = pandas.to_datetime(df[columns["Date"]].astype(str) + '-'+ df[columns["Group"]].astype(str) , errors='coerce', format='%d-%m-%Y')
        # Drop duplicates, sort by 'Mov', and reset index
        df = df.drop_duplicates().sort_values(by=columns["Date"]).reset_index(drop=True)
    except:
        df = None
        print("Alert - Failed to parse at parseDataframe() \n\n")
    
    return df

def existsNaNDataframe(df, column='Saldo'):
    """
    Check if NaN values exist in the specified column of the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        column (str): Column name.

    Returns:
        None
    """
    print("Nan Values:", df[column].isnull().values.any())  # There is nan
    print("Nan Values Indexes:", list(df.loc[pandas.isna(df[column]), :].index))  # Return the indexes
    print(df.loc[pandas.isna(df[column]), :])  # Return the rows with nan

def checkDataframeValid(df, columns=df_columns_translator):
    """
    Check the validity of a DataFrame based on specific criteria.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be validated.
    - columns: dict, optional
        A dictionary representing column translations. Default is df_columns_translator.

    Returns:
    bool
        True if the DataFrame is valid, False otherwise.
    """
    dateformat_valid_bool = pandas.to_datetime(df[df_columns_translator['Date']], format='%d-%m-%Y', errors='coerce').notnull().all()
    balance_valid_bool = df[df_columns_translator['Balance']].notnull().values.any()
    value_valid_bool = df[df_columns_translator['Value']].notnull().values.any()
    columns_names = set(df_columns_translator.values()).issubset(set(df.columns))
    return dateformat_valid_bool & balance_valid_bool & value_valid_bool & columns_names


def readPDF(file_dir="~/Downloads/extracto.pdf", debug=False, filter_set_column=pdf_columns, date_ref={'year':'2023', 'column': 'Valor'}, store_incache=None, cache_name='cached_dataframe.pkl'):
    """
    Read tables from a PDF file, filter based on specified columns, and add a reference to the DataFrame.

    Parameters:
    - file_dir: str, optional
        The file path of the PDF. Default is "~/Downloads/extracto.pdf".
    - debug: bool, optional
        If True, print debug information. Default is False.
    - filter_set_column: set, optional
        The set of columns used for filtering tables. Default is pdf_columns.
    - date_ref: dict, optional
        A dictionary containing year and column information for adding a reference to the DataFrame. Default is {'year':'2023', 'column': 'Valor'}.
    - store_incache: function, optional
        A function for storing the DataFrame in cache. Default is None. Example: df.to_pickle('cached_dataframe.pkl')
    - cache_name: str, optional
        The name of the cache file. Default is 'cached_dataframe.pkl'.

    Returns:
    - df: pandas DataFrame
        The resulting DataFrame.
    - non_idenfied_tables: list
        A list of tables that were not identified based on filtering.
    """
    tables_list = tabula.read_pdf(file_dir, pages="all")

    if debug:
        print(tables_list)

    filtered_tables=[]
    filtered_table_length=0
    non_idenfied_tables=[]

    # Filter tables based on specified columns
    for table in tables_list:
        if filter_set_column.issubset(table.columns):
                filtered_tables.append(table)
                filtered_table_length+=1

    # If no tables are identified, store column names for manual identification
    if filtered_table_length==0:
        for table in tables_list:
            filtered_tables.append(table.columns)
    
    df=filtered_tables[0].copy()

    if debug:
        print(len(filtered_tables[0]), len(filtered_tables[1]),len(filtered_tables[0])+ len(filtered_tables[1]) )
    
    # Concatenate identified tables
    for i in range(filtered_table_length-1):
        if debug:
            print("table[",i,"]")
        if list(filtered_tables[0].columns) == list(filtered_tables[i+1].columns):
            df=pandas.concat([df, filtered_tables[i+1]], ignore_index=True)
    
    df['Filename']=file_dir
    df['Category']=None

    if debug:
        print(len(filtered_tables[0]),len(df))
        print(df)
    if date_ref is not None:
        try:
            df[date_ref['column']]=date_ref['year']
        except Exception as e:
            print("Alert - Not able to add date reference",e)
            pass
        
    # Store your DataFrame
    if store_incache is not None:    
        store_incache(cache_name) # will be stored in the current directory store_incache->df.to_pickle
    
    return df, non_idenfied_tables


def filterDate(df, start, end_date=False, column='Mov', debug=False):
    """
    Filter a DataFrame based on a date range.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be filtered.
    - start: str
        The start date for filtering.
    - end_date: str, optional
        The end date for filtering. Default is False.
    - column: str, optional
        The column containing date information. Default is 'Mov'.
    - debug: bool, optional
        If True, print debug information. Default is False.

    Returns:
    pandas DataFrame
        The filtered DataFrame.
    """
    if end_date:
        mask = (df[column] >= start) & (df[column] <= end_date)
    else:
        mask = (df[column] >= start)

    if debug:
        print(mask.to_string())
        print(df[mask])
    
    return df[mask]


def StoreInCache(df, cache_name):
    """
    Store a DataFrame in cache.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be stored.
    - cache_name: str
        The name of the cache file.
    """
    if isinstance(df, pandas.DataFrame):
        df.to_pickle(cache_name)


#Example for Santander PDF files the df_columns = {"Balance":"Saldo","Value":"Valor.1", "Date":"Mov", "Info":"Descritivo do Movimento", "Group":"Valor", "Filename":"Filename"}
class FinancialDataProcessor:
    def __init__(self, df_columns, input_df=None, _put_store_data=None, _get_store_data=None, _data_processor=parseDataframe):
        self.df = None
        self.columns = df_columns
        self.categories = categories_hierarchy
        self.labels= {} # Example {'Transaction Name': 'Shopping''}
        self.put_store_data= _put_store_data
        self.get_store_data = _get_store_data
        self.data_processor = _data_processor

        if input_df is not None:
            self.df = ParseNewData(input_df, self._data_processor)

    def putStoreData(self, df=None, data_store_method=None, data_dir='cached_dataframe.pkl'):
        if df is None:
            df = self.df
        else:
            try:
                if not (df.empty):
                    self.df = df
            except:
                print ("*Alert* the dataframe is empty")
                pass

        if data_store_method is None:
            data_store_method = self.put_store_data

        if data_store_method  is not None and df is not None and data_dir is not None:
            data_store_method(self.df, data_dir)
            return True
        return False

    def getStoreData(self, data_dir='cached_dataframe.pkl'):
        if self.get_store_data is not None:
            return self.get_store_data(dir)
        return None

    def ParseNewData(self, df, parsing_func=None, data_store_method=None, data_dir='cached_dataframe.pkl', debug=False):
        # Call the parsing function
        # Set the default parsing function if None is provided
        if parsing_func is None:
            parsing_func = self.data_processor
        df = parsing_func(df,self.columns)
        if debug:
            print(df.to_string())

        # Store the DataFrame to pickle file if data_store_method is provided
        if data_store_method is not None and data_dir is not None:
            data_store_method(df, data_dir)
        
        if debug:
            print("\n\n after parsing_func ParseNewData call:\n", df.to_string())

        self.df = df

        return df


    def balanceAmount(self, df, column=None):
        if column is None:
            column=self.columns

        value0 = float(df[column['Balance']].iloc[0]) - float(df[column['Value']].iloc[0])
        valuet = float(df[column['Balance']].iloc[-1])
        diffvalue = float(valuet) - float(value0)
        result = {
            'Inicio': (df[column['Date']].iloc[0], value0),
            'Fim': (df[column['Date']].iloc[-1], valuet),
            'Resultado': diffvalue
        }
        print("Inicio:", df[column['Date']].iloc[0],value0,"€   Fim: ", df[column['Date']].iloc[-1], valuet, "€  Resultado:", diffvalue,"€")
        return result

    # column={ 'Info': 'Descritivo do Movimento','Balance':'Saldo','Value':'Valor.1', 'Date':'Mov'}
    def expensesBiggest(self, df, threshold=10,column=None):
        if column is None:
            column=self.columns
        print("======= " + str(threshold) + " Maiores Despesas ========")
        result = df.nsmallest(threshold, column['Value'], keep='all')[[column['Date'], column['Info'], column['Value']]]
        #print(result)
        print(" ")
        return result

    #column={ 'Info': 'Descritivo do Movimento','Balance':'Saldo','Value':'Valor.1', 'Date':'Mov'}
    def expensesAbove(self, df, threshold=10, column=None):
        if column is None:
            column=self.columns
        print("======= Compras acima de " + str(threshold) + " euros ========")
        result = df[df[column['Value']] < -threshold][[column['Date'], column['Info'], column['Value']]]
        #print(result)
        print(" ")
        return result

    def expensesReceiver(self, df, threshold=10,column=None):
        if column is None:
            column=self.columns
        print("======= " + str(threshold) + " Maiores Recebedores =======")
        receivers = df.groupby(column['Info'])[column['Value']].sum()
        result = receivers.nsmallest(threshold).reset_index()
        #print(result)
        print(" ")
        return result

    def expensesRecurring(self, df, threshold=10, column=None):
        if column is None:
            column=self.columns
        print("======= " + str(threshold) + " Despesas Mais Recorrentes =======")
        recurring_exp = df[(df[column['Value']] < 0)].groupby(column['Info']).size().to_frame(name='size').reset_index().sort_values(by='size', ascending=False)
        rec_merge = pd.merge(receivers, recurring_exp, on=column['Info'], how='inner')
        result = rec_merge.sort_values(by='size', ascending=False).rename(columns={column['Value']: 'Sum', 'size': 'Occurrences'}).head(threshold)

        return result

    def expensesAnalytics(self,df, column=None):
        if column is None:
            column=self.columns

        total_expense=(df[df[column['Value']]<0][column['Value']].sum())
        expense_avg=(df[df[column['Value']]<0][column['Value']].mean())
        expense_max=(df[df[column['Value']]<0][column['Value']].max())
        expense_min=(df[df[column['Value']]<0][column['Value']].min())
        
        duration= (df[column['Date']].iloc[-1] - df[column['Date']].iloc[0])
        expense_avg_day=total_expense/float(duration.days)

        return [total_expense,expense_avg, expense_avg_day, expense_min, expense_max]

    def earningsAnalytics(self,df, column=None):
        if column is None:
            column=self.columns

        total_earning=(df[df[column['Value']]>0][column['Value']].sum())
        earning_avg=(df[df[column['Value']]>0][column['Value']].mean())
        earning_max=(df[df[column['Value']]>0][column['Value']].max())
        earning_min=(df[df[column['Value']]>0][column['Value']].min())
        
        duration= (df[column['Date']].iloc[-1] - df[column['Date']].iloc[0])
        earning_avg_day=total_earning/float(duration.days)

        return [total_earning, earning_avg, earning_avg_day, earning_max, earning_min]

    def getDataMonth(self,df, column=None):
        if column is None:
            column=self.columns
        return (df[column['Date']].dt.month.unique())


    def getExpensesOverview(self,df=None, column=None):
        if column is None:
            column=self.columns
        if df is None:
            df=self.df
        expense_values = df[df[column['Value']]<0].groupby([column['Date']]).agg("sum")
        idx_expense_values= expense_values.index
        acc_expense_values=  expense_values[column['Value']].cumsum()
        return [idx_expense_values,-expense_values[column['Value']], -acc_expense_values]

    def getMarginOverview(self,df=None,column=None):
        if column is None:
            column=self.columns
        if df is None:
            df=self.df
        margin_values = df.groupby([column['Date']]).agg("sum")
        idx_margin_values = margin_values.index
        acc_margin_values = margin_values[column['Value']].cumsum()

        return [idx_margin_values,margin_values[column['Value']], acc_margin_values]

    def getIncomeOverwiew(self,df=None,column=None):
        if column is None:
            column=self.columns
        if df is None:
            df=self.df
        income_values = df[df[column['Value']]>0].groupby([column['Date']]).agg("sum")
        idx_income_values  = income_values.index
        acc_income_values = income_values[column['Value']].cumsum()
        
        return [idx_income_values, income_values[column['Value']], acc_income_values]

    def getExpensesMonthly(self,df=None,column=None):
        if column is None:
            column=self.columns
        if df is None:
            df=self.df
        # Perform the groupby operation and store the result in a variable
        grouped_expense_by_month = df[df[column['Value']] < 0].groupby([df[column['Date']].dt.year, df[column['Date']].dt.month]).sum()

        ##By month
        x_grouplist=[]
        for i in grouped_expense_by_month.index:
                x_grouplist.append(datetime.datetime.strptime(str(i[0])+'-'+str(i[1]), '%Y-%m'))

        sdf=grouped_expense_by_month.reset_index(drop=True) #expenses_by_month
        sdf[column['Date']]=x_grouplist
        return [sdf[column['Date']],sdf[column['Value']]]

    def getIncomeMonthly(self, df=None,column=None):
        if column is None:
            column=self.columns
        if df is None:
            df=self.df
        grouped_income_by_month = df[df[column['Value']] > 0].groupby([df[column['Date']].dt.year, df[column['Date']].dt.month]).sum()
        
        x_grouplist2=[]
        for i in grouped_income_by_month.index:
                x_grouplist2.append(datetime.datetime.strptime(str(i[0])+'-'+str(i[1]), '%Y-%m'))

        sdf2=grouped_income_by_month.reset_index(drop=True) #expenses_by_month
        sdf2[column['Date']]=x_grouplist2
        return [sdf2[column['Date']],sdf2[column['Value']]]

    def getMarginMonthly(self, df=None, column=None):
        if column is None:
            column=self.columns
        if df is None:
            df=self.df
        grouped_by_month = df.groupby([df[column['Date']].dt.year, df[column['Date']].dt.month]).sum()
        x_grouplist3=[]
        for i in grouped_by_month.index:
                x_grouplist3.append(datetime.datetime.strptime(str(i[0])+'-'+str(i[1]), '%Y-%m'))
        #print(x_grouplist)
        sdf3=grouped_by_month.reset_index(drop=True) #by_month
        sdf3[column['Date']]=x_grouplist3

        return [sdf3[column['Date']],sdf3[column['Value']]]

    def getIncomeYear(self, df=None, column=None):
        if column is None:
            column=self.columns
        if df is None:
            df=self.df

        # Access a specific group by using the .get_group() method on the GroupBy object
        #group=grouped_expense_by_month.loc[2023]
        # Display the result
        #print(grouped_expense_by_month.loc[grouped_expense_by_month.index[0]], grouped_expense_by_month.index, group.index,group.loc[group.index[0]]

        expense_by_year= df[df[column['Value']]<0].groupby([df[column['Date']].dt.year]).sum()
        x = expense_by_year.index
        y= -expense_by_year[column['Value']]
        return [x,y]

    def getExpensesYear(self, df=None, column=None):
        if column is None:
            column=self.columns
        if df is None:
            df=self.df
        income_by_year= df[df[column['Value']]>0].groupby([df[column['Date']].dt.year]).sum()
        x2 = income_by_year.index
        y2=  income_by_year[column['Value']]
        return [x2,y2]

    def getExpensesWeekly(self, df, column=None, filter_year=2023):
        if column is None:
            column=self.columns

        # Filter the DataFrame
        df_filtered = df[df[column['Value']] < 0]
        df_filtered= df_filtered.drop(columns=column['Balance']) #only drops on df_filtered and not df
        
        df_filtered=(df_filtered.groupby([df[column['Date']].dt.year, df[column['Date']].dt.strftime('%U')]).sum())
        #print(df_filtered.loc[test.index[0]],test.rename_axis(index=["Year", "Week"]).sort_index(level="Week")) # renaing multi-index 
        t=(df_filtered[df_filtered.index.get_level_values(0).isin([filter_year])].droplevel(0)) #level 0 - Year, level 1 - Week
        return [t.index.astype(int),t[column['Value']]]
   

    def plotOverviewDF(self,df,fig,ax,columns=None,label=False):
        if columns is None:
            columns=self.columns

        expense_values = df[df[columns['Value']]<0].groupby([columns['Date']]).agg("sum")
        x1 = expense_values.index
        y1 = -expense_values[columns['Value']]
        y3 = -expense_values[columns['Value']].cumsum()

        margin_values = df.groupby([columns['Date']]).agg("sum")
        x5= margin_values.index
        y5= margin_values[columns['Value']].cumsum()


        income_values = df[df[columns['Value']]>0].groupby([columns['Date']]).agg("sum")
        x2 = income_values.index
        y2 = income_values[columns['Value']]
        y4 = income_values[columns['Value']].cumsum()

        # plot
        width = 2

        acumulative1=ax.stackplot(x1, y3,alpha=0.2, color='red')
        acumulative2=ax.stackplot(x2, y4,alpha=0.1, color='green')

        bar1=ax.bar(x1, y1, color='red',width=0.5*width)
        bar2=ax.bar(x2,y2,color='green',alpha=0.5)
        line1=ax.plot(x5,y5,color='k', alpha=0.1) 
        
        #fig.suptitle('This is a somewhat long figure title', fontsize=16)
        ax.set(ylabel='Value (€)', xlabel='Time', title='Expenses x Income during time')
        plt.xticks(rotation=30,ha='right')
        if label:
            ax.bar_label(bar1, rotation=30)
            ax.bar_label(bar2,rotation=30)
    
    # Example of usage: df=processor.assingCategory(attr, df,debug=True)
    ## Where attr must be attr = {'COMPRA *6081 A9 LOURES':'Toll Tax','COMPRA *6081 PRIO ENERGY ALFORNELOS':'Fuel', "COMPRA *6081 PD RAMADA ODIVELAS":'Supermarket','COMPRA *0080 CP AMADORA 2700-349-AMAD':'Train'}
    def assingCategory(self,  categories:dict, df=None,debug=False):
        if df is None:
            df=self.df
        if debug:
            print(df.head(10).to_string())

        df[self.columns['Category']] = df[self.columns['Info']].map(categories)
        if debug:
            print(df.head(10).to_string())

        return df

    def getTransferNotCategorised(self, df=None):
        if df is None:
            df=self.df_cache
        unique_labels_filtered = df[[self.columns['Info'],self.columns['Category']]]
        return unique_labels_filtered[pandas.isna(unique_labels_filtered[self.columns['Category']])][self.columns['Info']]

    def getUniqueTransfers(self, df=None):
        if df is None:
            df=self.df
        unique_labels = df[self.columns['Info']].unique()
        return unique_labels

    def getSimilarTransfers(self, df=None):
        if df is None:
            df=self.df
        unique_labels = df[self.columns['Info']].unique()
        transfer_types_data = create_transfer_types_data(unique_labels)
        guess_dict = get_similiar_transfer_types(transfer_types_data, levenshtein_distance)
        return guess_dict

    def getTransfersByType(self,df=None):
        if df is None:
            df=self.df
        unique_labels = df[self.columns['Info']].unique()
        transfer_types_data = create_transfer_types_data(unique_labels)
        return convert_inner_lists_of_list_to_strings(transfer_types_data)



def create_transfer_types_data (unique_labels:dict):
    """
    Create a dictionary to store information about transfer types.

    Parameters:
    - unique_labels: dict
        Dictionary containing unique labels.

    Returns:
    dict
        Dictionary with transfer types and their corresponding details.
    """
    # Dictionary to store information about transfer types
    transfer_types_data = {}

    # Iterate through unique labels to create a dictionary with transfer types and their corresponding details
    for label in unique_labels:
        words = label.split(" ")  # Split the label into words
        word_count = len(words)
        
        # Depending on the number of words in the label, create a nested structure in the dictionary
        if word_count > 2:
            try:
                transfer_types_data[str(words[0]) + str(words[1])].append(words[2:])
            except KeyError:
                transfer_types_data[str(words[0]) + str(words[1])] = [words[2:]]
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    return transfer_types_data

def get_similiar_transfer_types(transfer_types_data: dict, levenshtein_distance):
    """
    Find similar transfer types using Levenshtein distance.

    Parameters:
    - transfer_types_data: dict
        Dictionary with transfer types and their details.
    - levenshtein_distance: function
        Levenshtein distance function.

    Returns:
    dict
        Dictionary containing similar transfer types.
    """
    # Extract keys from the dictionary
    transfer_type_keys = list(transfer_types_data.keys())  
    number_of_keys = len(transfer_type_keys)
    guess_dict = {}

    # Iterate through the keys to find similar transfer types using Levenshtein distance
    for i in range(number_of_keys - 1):
        for j in range(i + 1, number_of_keys - 1):
            key_i, key_j = transfer_type_keys[i], transfer_type_keys[j]
            distance = levenshtein_distance(key_i, key_j)

            # Check if Levenshtein distance is less than 3
            if distance < 3:
                transfer_labels_i, transfer_labels_j = transfer_types_data[key_i], transfer_types_data[key_j]
                # Iterate through the transfer labels of the current keys
                for transfer_label_i in transfer_labels_i:
                    for transfer_label_j in transfer_labels_j:
                        # Compare details of transfer types
                        if transfer_label_i[0] == transfer_label_j[0]:
                            fullstring = f"{key_i} {' '.join(transfer_label_j)}"
                            fullstring2 = f"{key_j} {' '.join(transfer_label_i)}"

                            # Update guess_dict
                            try:
                                guess_dict.setdefault(fullstring2, []).append(fullstring)
                            except Exception as e:
                                print(f"An unexpected error occurred: {e}")
    return guess_dict



def checkUniqueFilesInCache(df_cache, file_list):
    """
    Check for unique files in the cache.

    Parameters:
    - df_cache: pandas DataFrame
        Cached DataFrame.
    - file_list: list
        List of file names.

    Returns:
    dict
        Dictionary containing unique files.
    """
    if isinstance(df_cache, type(None)) == False: #does not existe data

        cached_file_list= df_cache['Filename'].unique()

        filtered_file=[]
        dir_file_list={}
        for keys in file_list:
            for cached_filename in cached_file_list:
                if cached_filename.split("/")[-1] not in file_list[keys]: #get filename without the path
                    filtered_file.append(file_list[keys])

            dir_file_list[keys]=filtered_file
    else:
        print("line 352-Alert: Does not exists stored data (cache)\n")
        dir_file_list =file_list
    return dir_file_list

def getLocalStoreDataFrame():
    """
    Get the local stored DataFrame in cache.

    Returns:
    tuple
        Tuple containing DataFrame, cached DataFrame, and count.
    """
    read_cache = None
    df=None
    debug=False
  
    df_cache=collectCacheData(debug=False,collect_cache_func=pandas.read_pickle, bool_assert_cache_func=(lambda dataframe: dataframe.empty))
    file_list =getLocalFiles()
    dir_file_list = checkUniqueFilesInCache(df_cache, file_list)
    df, count = processDirectoryFiles(dir_file_list, read_file_func=readPDF)

    return df, df_cache, count


if __name__ == "__main__":
    read_cache = None
    df=None
    debug=False
    processor =FinancialDataProcessor(df_columns_translator)
    
    # add  the files processing to larger fucntion; the ideia is  having multiple ways to get data, inclusive from a API  
    df, df_cache, count = getLocalStoreDataFrame()
    # processor should only have store method and not passing datastore_method; need to be adaptable to store both in cloud and cache
    if count:
        df = processor.ParseNewData(df)
        if isinstance(df_cache, type(None)) == False:
            #df = processor.ParseNewData(df)
            if checkDataframeValid(df_cache):
                df = pandas.concat([df,df_cache])

        status = processor.putStoreData(df,data_store_method= StoreInCache, data_dir='cached_dataframe.pkl')
    elif isinstance(df_cache, type(None)) == False and checkDataframeValid(df_cache):
            df = df_cache
    else:
        print("line-369 No Data Available.")
        sys.exit(0)
    
    # ====== Start: Example of usage ======
    # print(processor.balanceAmount(df))
    # print("processor.expensesAnalytics(df)")
    # print(processor.expensesAnalytics(df))
    # print("processor.earningsAnalytics(df)")
    # print(processor.earningsAnalytics(df))
    # print(processor.expensesAbove(df,100))
    # print(processor.expensesBiggest(df))

    # print("Months Available: processor.getDataMonth(df) \n")
    # print(processor.getDataMonth(df))

    # print("getTransferNotCategorised() \n",processor.getTransferNotCategorised(df), "\n\n\n")
    # print("getUniqueTransfers() \n",processor.getUniqueTransfers(df),"\n\n\n")
    # print("processor.getSimilarTransfers \n",processor.getSimilarTransfers(df),"\n\n\n")
    # print("processor.getTransfersByType \n",processor.getTransfersByType(df),"\n\n\n")
    # ====== End: Example of usage ======

    attr = {'COMPRA *6081 A9 LOURES':'Toll Tax','COMPRA *6081 PRIO ENERGY ALFORNELOS':'Fuel', "COMPRA *6081 PD RAMADA ODIVELAS":'Supermarket','COMPRA *0080 CP AMADORA 2700-349-AMAD':'Train'}
    df=processor.assingCategory(attr, df)

    # plot
    fig=plt.figure()
    ax=fig.add_subplot()# 121 for 3 plot at same time

    string_input = """\nWrite 1,2,3 or 4 to select the example:\n
    1 - Weekly Espenses View\n
    2 - Year View 1\n
    3 - Year View 2\n
    4 - Montlhy Expenses versus Income View\n
    Other - Exit()\n
    > """
    while True:
        
        choice = input(string_input)

        fig.clf()
        plt.close() 
        fig=plt.figure()
        ax=fig.add_subplot()# 121 for 3 plot at same time

        if(choice =='1' ):
            ## Example 1
            data=processor.getExpensesWeekly(df)
            ax.bar(data[0],-data[1])
        elif(choice =='2'):
            ## Example 2
            processor.plotOverviewDF(df,fig,ax)
        elif(choice =='3'):
            ## Example 3
            plotOverview_EI(processor.getExpensesOverview(df),processor.getIncomeOverwiew(df),processor.getMarginOverview(df),fig,ax)
        elif(choice =='4'):
            ## Example 4
            plotSimple_EI([processor.getExpensesMonthly(df)[0],-processor.getExpensesMonthly(df)[1]],processor.getIncomeMonthly(df),fig, ax)
            line0=ax.stackplot(processor.getMarginMonthly(df)[0],processor.getMarginMonthly(df)[1],color='k', alpha=0.1)
        else:
            print("Exit(); Bye.")
            sys.exit(0)

        plt.show()

