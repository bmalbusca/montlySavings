import os
import pandas
import tabula
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

def str2num(x: str) -> float:
    try:
        return float(x.split()[0].replace('.','').replace(',','.'))
    except:
        #print(x)
        return x


def getLocalFiles(debug = False) -> list:
    existing_files_list = []

    cwd = os.getcwd()
    dir_list = os.listdir(cwd)
    if debug:
        print(dir_list)

    for filename in dir_list:
        suffixes = filename.split(".")
        if debug:
            print(suffixes)
        if suffixes[-1]=="pdf":
            existing_files_list.append(filename)
    if debug:
        print(existing_files_list)
    return {cwd : existing_files_list}



def collectCacheData(cache_name='cached_dataframe.pkl',debug=False):
    # Read your DataFrame
    try:
        read_cache = pandas.read_pickle(cache_name) # read from current directory
        if debug:
            print("[LOAD]: Loading cache data", read_cache)

        if not read_cache.empty :
            return read_cache
    except:
        if debug:
            print("[ALERT]: No cached data")
    return None

def readPDFSantander(file_dir="~/Downloads/extracto.pdf", debug=False, filter_set_column={'Mov', 'Valor', 'Descritivo do Movimento','Valor.1' ,'Saldo'}, store_incache=True):

    tables = tabula.read_pdf(file_dir, pages="all")
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
    if store_incache:
        df.to_pickle('cached_dataframe.pkl') # will be stored in current directory
    
    return df

def existsNaNDataframe(df, column='Saldo'):
    print("Nan Values:",df[column].isnull().values.any()) #There is nan
    print("Nan Values Indexes:",list(df.loc[pandas.isna(df[column]), :].index)) #return the indexes
    print(df.loc[pandas.isna(df[column]), :]) # return the rows with nan

def parseDataframe(df):
    df=df.dropna(subset=['Saldo']).reset_index(drop=True)
    df['Saldo']=df['Saldo'].apply(str2num)
    df= df.dropna(subset=['Valor.1']).reset_index(drop=True)
    df['Valor.1'] =df['Valor.1'].apply(str2num)
    # change this read the value of Valor or gettign the year by intput
    df['Mov']=df['Mov'].apply(lambda x: str(x)+'-'+str(2023))
    df['Mov'] = pandas.to_datetime(df['Mov'],errors='coerce', format='%d-%m-%Y')
    df.drop_duplicates(inplace=True)
    df.sort_values(by='Mov',inplace=True)
    df.reset_index(inplace=True,drop=True)
    return df
 
def balanceAmount(df, column='Saldo'):
    value0= df[column].iloc[0]-df['Valor.1'].iloc[0]
    valuet= df[column].iloc[-1]
    diffvalue=float(valuet)-float(value0)
    print("Inicio:", df['Mov'].iloc[0],value0,"€   Fim: ", df['Mov'].iloc[-1], valuet, "€  Resultado:", diffvalue,"€")

## example: filterDate(df,'2023-07-11','2023-07-13' ))
def filterDate(df, start, end_date=False, debug=False):
    if end_date:
        mask = (df['Mov'] >= start) & (df['Mov'] <= end_date)
    else:
        mask = (df['Mov'] >= start)

    if debug:
        print(mask.to_string())
        print(df[mask])
    
    return df[mask]



## Despesas e ganhos por mes, selecionar mes e valor medio de gastos e valores maximo e minimos 
def expensesBiggest(df,threshold=10):
    print("======= 10 Maiores Despesas ========")
    print(df.nsmallest(threshold,'Valor.1', keep='all')[['Mov','Descritivo do Movimento', 'Valor.1']])
    print(" ")

def expensesAbove(df,threshold=10):
    print("======= Compras acima de 10 euros ========")
    print(df[df['Valor.1']<-threshold][['Mov','Descritivo do Movimento', 'Valor.1']])
    print(" ")

def expensesReceiver(df,threshold=10):
    print("======= 10 Maiores Recebedores =======")
    receivers= df.groupby("Descritivo do Movimento")["Valor.1"].sum()
    print(receivers.nsmallest(threshold).reset_index())
    print(" ")

def expensesRecurring(df,threshold=10):
    print("======= 10 Despesas Mais Recorrentes =======")
    recurring_exp = df[(df['Valor.1'] < 0)].groupby("Descritivo do Movimento").size().to_frame(name = 'size').reset_index().sort_values(by='size', ascending=False)
#print(recurring_exp.head(10))
    rec_merge = pandas.merge( receivers, recurring_exp, on="Descritivo do Movimento", how='inner') 
    print(rec_merge.sort_values(by='size', ascending=False).rename(columns = {'Valor.1':'Soma', 'size':'Ocurrencias'}).head(threshold))

def expensesAnalytics(df, column='Valor.1'):
    total_expense=(df[df[column]<0][column].sum())
    expense_avg=(df[df[column]<0][column].mean())
    expense_max=(df[df[column]<0][column].max())
    expense_min=(df[df[column]<0][column].min())
    
    duration= (df['Mov'].iloc[-1] - df['Mov'].iloc[0])
    expense_avg_day=total_expense/float(duration.days)

    return [total_expense,expense_avg, expense_avg_day, expense_min, expense_max]

def earningsAnalytics(df, column='Valor.1'):
    total_earning=(df[df[column]>0][column].sum())
    earning_avg=(df[df[column]>0][column].mean())
    earning_max=(df[df[column]>0][column].max())
    earning_min=(df[df[column]>0][column].min())
    
    duration= (df['Mov'].iloc[-1] - df['Mov'].iloc[0])
    earning_avg_day=total_earning/float(duration.days)

    return [total_earning, earning_avg, earning_avg_day, earning_max, earning_min]

def getDataMonth(df, column='Mov'):
    return (df[column].dt.month.unique())


if __name__ == "__main__":
    read_cache = None
    df=None
    debug=False


    file_list =getLocalFiles()
    df=collectCacheData(debug=False)

    if isinstance(df, type(None)) == True: #does not existe data
        df = pandas.DataFrame()
        new_file=False
        for fdir in list(file_list.keys()):
            for filename in file_list[fdir]:
                complete_dir = fdir+'/'+filename
                s = readPDFSantander(file_dir=complete_dir,store_incache=False)
                #print(s)
                new_file=True
                df=pandas.concat([df,s])
        if new_file:
            df=parseDataframe(df)
            if debug:
                print(df.to_string())
            df.to_pickle('cached_dataframe.pkl') # will be stored in current directory

    else: #see this else routine
        cache_file_list= df['Filename'].unique()
        new_file=False
        for fdir in list(file_list.keys()):
            for filename in file_list[fdir]:
                complete_dir = fdir+'/'+filename
                if complete_dir  not in cache_file_list:
                    s = readPDFSantander(file_dir=complete_dir,store_incache=False)
                    s = parseDataframe(s)
                    df=pandas.concat([df,s])
                    new_file=True
        if new_file:
            df=parseDataframe(df)
            if debug:
                print(df.to_string())
            df.to_pickle('cached_dataframe.pkl') # will be stored in current directory


    print(expensesAnalytics(df))
    print(earningsAnalytics(df))
    #expensesAbove(df,30)
    #expensesBiggest(df)
    print(getDataMonth(df))
    balanceAmount(df)
    
    # Perform the groupby operation and store the result in a variable
    grouped = df[df['Valor.1'] < 0].groupby([df['Mov'].dt.year, df['Mov'].dt.month]).sum()

    # Access a specific group by using the .get_group() method on the GroupBy object
    group=grouped.loc[2023]
    # Display the result
    print(grouped.loc[grouped.index[0]], grouped.index, group.index,group.loc[group.index[0]])

    df[df['Valor.1']<0].groupby([df['Mov'].dt.year]).sum()
    


    #converter o plot num funcao e fazer pequenas funcoes para criar o dataframe final: sum dia, mes, ano; e filtro com filterDate
    expense_values = df[df['Valor.1']<0].groupby(['Mov']).agg("sum")
    x1 = expense_values.index
    y1 = -expense_values['Valor.1']
    y3 = -expense_values['Valor.1'].cumsum()
    
    margin_values = df.groupby(['Mov']).agg("sum")
    x5= margin_values.index
    y5= margin_values['Valor.1'].cumsum()
 
 
    income_values = df[df['Valor.1']>0].groupby(['Mov']).agg("sum")
    x2 = income_values.index
    y2 = income_values['Valor.1']
    y4 = income_values['Valor.1'].cumsum()

     # plot
    fig=plt.figure()
    plt.figure(1).clear()
    width = 2
    ax=fig.add_subplot(111, label="Spend")
    
    acumulative1=ax.stackplot(x1, y3,alpha=0.2, color='red')
    acumulative2=ax.stackplot(x2, y4,alpha=0.1, color='green')

    bar1=ax.bar(x1, y1, color='red',width=0.5*width)
    bar2=ax.bar(x2,y2,color='green',alpha=0.5)
    line1=ax.plot(x5,y5,color='k', alpha=0.1) 
    #fig.suptitle('This is a somewhat long figure title', fontsize=16)
    ax.set(ylabel='Value (€)', xlabel='Time', title='Expenses x Income during time')
    plt.xticks(rotation=30,ha='right')
    #ax.bar_label(bar1, rotation=30)
    #ax.bar_label(bar2,rotation=30)

    plt.show()

