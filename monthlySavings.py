import sys
import datetime
import pandas
import tabula
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from sysfiles import str2num, getLocalFiles, collectCacheData
from plotdata import plotOverview_EI, plotSimple_EI

def processDirectoryFiles(file_list, read_file_func):
    df = pandas.DataFrame()
    count = 0
    for fdir in list(file_list.keys()):
        for filename in file_list[fdir]:
            complete_dir = fdir+'/'+filename
            s = read_file_func(complete_dir)
            df = pandas.concat([df, s])
            count += 1
    return df, count


def parseDataframe(df, columns={"Balance":"Saldo","Value":"Valor.1", "Date":"Mov", "Info":"Descritivo do Movimento", "Group":"Valor", "Filename":"Filename"}):
    try:
        # Drop rows with NaN in 'Saldo' and 'Valor.1' columns
        df = df.dropna(subset=[columns["Balance"], columns["Value"]]).reset_index(drop=True)
        # Apply str2num function to 'Saldo' and 'Valor.1' columns
        df[[columns["Balance"], columns["Value"]]] = df[[columns["Balance"], columns["Value"]]].applymap(str2num)
        # Combine 'Mov' with the predefined year and convert to datetime
        df[columns["Date"]] = pandas.to_datetime(df[columns["Date"]].astype(str) + '-2023', errors='coerce', format='%d-%m-%Y')
        # Drop duplicates, sort by 'Mov', and reset index
        df = df.drop_duplicates().sort_values(by=columns["Date"]).reset_index(drop=True)
    except:
        df=None

    return df

def readPDF(file_dir="~/Downloads/extracto.pdf", debug=False, filter_set_column={'Mov', 'Valor', 'Descritivo do Movimento','Valor.1' ,'Saldo'}, store_incache=None, cache_name='cached_dataframe.pkl'):

    tables = tabula.read_pdf(file_dir, pages="all")

    if debug:
        print(tables)

    filtered_tables=[]
    for table in tables:
        if filter_set_column.issubset(table.columns):
                filtered_tables.append(table)
    
    #change this for future; manual find out is not the correct way, maybe should pass as function or each for each version known
    #year= tables[-2]['Valor'][0].split('-')[0]
    year='2023'
    df=filtered_tables[0].copy()

    if debug:
        print(len(filtered_tables[0]), len(filtered_tables[1]),len(filtered_tables[0])+ len(filtered_tables[1]) )
    
    for i in range(len(filtered_tables)-1):
        if debug:
            print("table[",i,"]")
        if list(filtered_tables[0].columns) == list(filtered_tables[i+1].columns):
            df=pandas.concat([df, filtered_tables[i+1]], ignore_index = True)
    df['Filename']=file_dir
    if debug:
        print(len(filtered_tables[0]),len(df))
        print(df)
    df['Valor']=year
    # Store your DataFrame
    if store_incache is not None:
        #df.to_pickle('cached_dataframe.pkl') # will be stored in current directory store_incache->df.to_pickle
        store_incache(cache_name)
    
    return df

def existsNaNDataframe(df, column='Saldo'):
    print("Nan Values:",df[column].isnull().values.any()) #There is nan
    print("Nan Values Indexes:",list(df.loc[pandas.isna(df[column]), :].index)) #return the indexes
    print(df.loc[pandas.isna(df[column]), :]) # return the rows with nan

## example: filterDate(df,'2023-07-11','2023-07-13' ))
def filterDate(df, start, end_date=False,column='Mov', debug=False):
    if end_date:
        mask = (df[column] >= start) & (df[column] <= end_date)
    else:
        mask = (df[column] >= start)

    if debug:
        print(mask.to_string())
        print(df[mask])
    
    return df[mask]


class FinancialDataProcessor:
    def __init__(self, input_df=None):
        self.df = None
        self.columns = {"Balance":"Saldo","Value":"Valor.1", "Date":"Mov", "Info":"Descritivo do Movimento", "Group":"Valor", "Filename":"Filename"}
        self.categories = [
    {'label': 'Housing', 'value': 0, 'sub': [
        {'label': 'Mortgage/Rent', 'value': 0},
        {'label': 'House Insurance', 'value': 0},
        {'label': 'Repairs', 'value': 0},
        {'label': 'Utilities', 'value': 0, 'sub': [
            {'label': 'Telcom', 'value': 0},
            {'label': 'Electricity', 'value': 0},
            {'label': 'Water', 'value': 0},
            {'label': 'Gas', 'value': 0}
        ]}
    ]},
    {'label': 'Transport', 'value': 0, 'sub': [
        {'label': 'Car Loan', 'value': 0},
        {'label': 'Car Insurance', 'value': 0},
        {'label': 'Public Transport', 'value': 0, 'sub': [
            {'label': 'Train', 'value': 0},
            {'label': 'Metro', 'value': 0},
            {'label': 'Bus', 'value': 0}
        ]},
        {'label': 'Maintenance', 'value': 0},
        {'label': 'Fuel', 'value': 0},
        {'label': 'Flight', 'value': 0}
    ]},
    {'label': 'Food', 'value': 0, 'sub': [
        {'label': 'Groceries', 'value': 0, 'sub': [
            {'label': 'Super Market', 'value': 0},
            {'label': 'Local Market', 'value': 0}
        ]},
        {'label': 'Take Away', 'value': 0}
    ]},
    {'label': 'Health', 'value': 0, 'sub': [
        {'label': 'Doctor Visits', 'value': 0},
        {'label': 'Medications', 'value': 0},
        {'label': 'Supplements', 'value': 0},
        {'label': 'Health Exams', 'value': 0},
        {'label': 'Dentist', 'value': 0},
        {'label': 'Treatments', 'value': 0},
        {'label': 'Health Insurance', 'value': 0},
        {'label': 'Life Insurance', 'value': 0}
    ]},
    {'label': 'Personal Care', 'value': 0, 'sub': [
        {'label': 'Gym', 'value': 0},
        {'label': 'Haircut', 'value': 0},
        {'label': 'Toiletries', 'value': 0},
        {'label': 'SPA', 'value': 0}
    ]},
    {'label': 'Education', 'value': 0, 'sub': [
        {'label': 'Tuition Fees', 'value': 0},
        {'label': 'Learning Materials', 'value': 0},
        {'label': 'School Fees', 'value': 0},
        {'label': 'Childcare', 'value': 0}
    ]},
    {'label': 'Leisure', 'value': 0, 'sub': [
        {'label': 'Cinema', 'value': 0},
        {'label': 'Theater/Concert', 'value': 0},
        {'label': 'Festivals', 'value': 0},
        {'label': 'Cafe', 'value': 0},
        {'label': 'Restaurant', 'value': 0},
        {'label': 'Books', 'value': 0},
        {'label': 'Hobbies', 'value': 0},
        {'label': 'Streaming Services', 'value': 0},
        {'label': 'Bars', 'value': 0},
        {'label': 'Hotels', 'value': 0},
        {'label': 'Subscriptions', 'value': 0},
        {'label': 'Trip Experiences', 'value': 0}
    ]},
    {'label': 'Shopping', 'value': 0, 'sub': [
        {'label': 'Clothing', 'value': 0},
        {'label': 'Shoes', 'value': 0},
        {'label': 'Accessories', 'value': 0},
        {'label': 'Gifts', 'value': 0}
    ]},
    {'label': 'Savings', 'value': 0, 'sub': [
        {'label': 'Emergency Fund', 'value': 0},
        {'label': 'Retirement', 'value': 0},
        {'label': 'Term Deposits', 'value': 0}
    ]}
    ]
        self.labels= {} # Example {'Transaction Name': 'Shopping''}

        if input_df is not None:
            self.df = ParseNewData(input_df, parseDataframe)
     
    def ParseNewData(self, df, parsing_func=parseDataframe, store_incache=None, cache_name='cached_dataframe.pkl', debug=False):
        # Call the parsing function
        df = parsing_func(df,self.columns)

        if debug:
            print(df.to_string())

        # Store the DataFrame to pickle file if store_incache is provided
        if store_incache is not None and cache_name is not None:
            store_incache(cache_name)
        
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
        #print(result)
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
        #print(idx_expense_values, expense_values)
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
        # print(grouped_expense_by_month,grouped_expense_by_month[column['Value']],grouped_expense_by_month.index[0][1])

        ## By month
        x_grouplist=[]
        for i in grouped_expense_by_month.index:
                x_grouplist.append(datetime.datetime.strptime(str(i[0])+'-'+str(i[1]), '%Y-%m'))
        
        #print(x_grouplist)
        sdf=grouped_expense_by_month.reset_index(drop=True) #expenses_by_month
        sdf[column['Date']]=x_grouplist
        
        #print(sdf.head())
        #print(sdf[column['Date']],sdf[column['Value']])

        return [sdf[column['Date']],sdf[column['Value']]]

    def getIncomeMonthly(self, df=None,column=None):
        if column is None:
            column=self.columns
        if df is None:
            df=self.df
        grouped_income_by_month = df[df[column['Value']] > 0].groupby([df[column['Date']].dt.year, df[column['Date']].dt.month]).sum()
        # print(grouped_expense_by_month,grouped_expense_by_month[column['Value']],grouped_expense_by_month.index[0][1])

        x_grouplist2=[]
        for i in grouped_income_by_month.index:
                x_grouplist2.append(datetime.datetime.strptime(str(i[0])+'-'+str(i[1]), '%Y-%m'))
        #print(x_grouplist)
        sdf2=grouped_income_by_month.reset_index(drop=True) #expenses_by_month
        sdf2[column['Date']]=x_grouplist2
        #print(sdf.head())
        #print(sdf2[column['Date']],sdf2[column['Value']])
        
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


def create_transfer_types_data (unique_labels:dict):
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

def convert_inner_lists_of_list_to_strings(transfer_types_data):
    """Convert inner arrays to single strings in the original dictionary."""
    for key, value in transfer_types_data.items():
        converted_values = [' '.join(inner_array) for inner_array in value]
        transfer_types_data[key] = converted_values

def checkUniqueFilesInCache(df_cache, file_list):
    if isinstance(df_cache, type(None)) == False: #does not existe data

        cached_file_list= df_cache['Filename'].unique()

        filtered_file=[]
        dir_file_list={}
        for keys in file_list:
            for cached_filename in cached_file_list:
                if cached_filename.split("/")[-1] not in file_list[keys]:
                    print(cached_filename.split("/")[-1]," is a new file to explore\n")
                    filtered_file.append(file_list[keys])

            dir_file_list[keys]=filtered_file
    else:
        print("line 352-Alert: Does not exists stored data (cache)\n")
        dir_file_list =file_list
    return dir_file_list

def getLocalStoreDataFrame():
    read_cache = None
    df=None
    debug=False
    ## add  the files processing to larger fucntion; the ideia is  having multiple ways to get data, inclusive from a API    
    df_cache=collectCacheData(debug=False,collect_cache_func=pandas.read_pickle, bool_assert_cache_func=(lambda dataframe: dataframe.empty))
    file_list =getLocalFiles()

    dir_file_list = checkUniqueFilesInCache(df_cache, file_list)

    df, count = processDirectoryFiles(dir_file_list, read_file_func=readPDF)

    if isinstance(df_cache, type(None)) == False:
        df = pandas.concat([df,df_cache])
    elif not(count):
        print("line-369 No Data Available.")
        sys.exit(0)

    return df, count

if __name__ == "__main__":
    read_cache = None
    df=None
    debug=False
    processor =FinancialDataProcessor()
    df, count = getLocalStoreDataFrame()

    ## processor should only have store method and not passing store_incache; need to be adaptable to store both in cloud and cache
    df = processor.ParseNewData(df,store_incache= df.to_pickle)
   
   
    print(processor.balanceAmount(df))
    print("processor.expensesAnalytics(df)")
    print(processor.expensesAnalytics(df))
    print("processor.earningsAnalytics(df)")
    print(processor.earningsAnalytics(df))
    print(processor.expensesAbove(df,100))
    print(processor.expensesBiggest(df))

    print("Months Available: processor.getDataMonth(df) \n")
    print(processor.getDataMonth(df))


    # Extract unique values from the "Descritivo do Movimento" column in the DataFrame
    unique_labels = df["Descritivo do Movimento"].unique()
    transfer_types_data = create_transfer_types_data(unique_labels)
    guess_dict = get_similiar_transfer_types(transfer_types_data, levenshtein_distance)
    

    # Update the original dictionary to have inner arrays converted to single strings
    # convert_inner_lists_of_list_to_strings(transfer_types_data)

    # Print the result
    print("These movements are the same category? \n\n", guess_dict, "\n\n", unique_labels,"\n\n")
    # now, create a pandas dataframe where column "Descritivo do Movimento" is unique_labels, column "label" will be the category; Then  or each unique_label that is a key on guess_dict should have the same label that are the key value of that key 


    print("\n\n Prepare data for plot \n\n")
  
    
    # plot
    fig=plt.figure()
    #plt.figure(1).clear()
    ax=fig.add_subplot(111)# 121 for 3 plot at same time
    #ax2=fig.add_subplot(224, label="Spend")
    #ax2.stem(x,y)
    #ax1=fig.add_subplot(222, label="montly Spend")
    #ax1.bar(sdf['Mov'].values,-sdf['Valor.1'])
    #ax.bar(t.index.astype(int),t['Valor.1'])
    data= processor.getExpensesWeekly(df)
    #ax.bar(data[0],data[1])
  
    
    #processor.plotOverviewDF(df,fig,ax)
    #plotOverview_EI(processor.getExpensesOverview(df),processor.getIncomeOverwiew(df),processor.getMarginOverview(df),fig,ax)
    #plotSimple_EI([x,y],[x2,y2],fig, ax)
    #plotSimple_EI([sdf['Mov'],-sdf['Valor.1']],[sdf2['Mov'],sdf2['Valor.1']],fig, ax)
    #line0=ax.plot(sdf3['Mov'],sdf3['Valor.1'],color='k', alpha=0.1)
    plotSimple_EI([processor.getExpensesMonthly(df)[0],-processor.getExpensesMonthly(df)[1]],processor.getIncomeMonthly(df),fig, ax)
    line0=ax.stackplot(processor.getMarginMonthly(df)[0],processor.getMarginMonthly(df)[1],color='k', alpha=0.1)

    plt.show()

