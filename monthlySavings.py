import os, types
import pandas
import tabula

def str2num(x: str) -> float:
    try:
        return float(x.split()[0].replace('.','').replace(',','.'))
    except:
        #print(x)
        return x

read_cache = None
df=None

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


test =getLocalFiles()
print(test)

def collectCacheData(cache_name='cached_dataframe.pkl',debug=False):
    # Read your DataFrame
    read_cache = pandas.read_pickle(cache_name) # read from current directory
    if debug:
        print("read_cache", read_cache)

    if not read_cache.empty :
        return read_cache
    return None

def readPDF(file_dir="~/Downloads/extracto.pdf", debug=False, filter_set_column={'Mov', 'Valor', 'Saldo'}, store_incache=True):

    tables = tabula.read_pdf(file_dir, pages="all")
    #print(tables[0], type(tables[0]), len(tables[0]))

    filtered_tables=[]
    for table in tables:
        if filter_set_column.issubset(table.columns):
                filtered_tables.append(table)

    df=filtered_tables[0].copy()
    if debug:
        print(len(filtered_tables[0]), len(filtered_tables[1]),len(filtered_tables[0])+ len(filtered_tables[1]) )
    
    for i in range(len(filtered_tables)-1):
        if debug:
            print("table[",i,"]")
        if list(filtered_tables[0].columns) == list(filtered_tables[i+1].columns):
            df=pandas.concat([df, filtered_tables[i+1]], ignore_index = True)
    if debug:
        print(len(filtered_tables[0]),len(df))
        print(df)
    
    # Store your DataFrame
    if store_incache:
        df.to_pickle('cached_dataframe.pkl') # will be stored in current directory
    
    return df

df=collectCacheData(debug=True)
if isinstance(df, type(None)) == True:
    df=readPDF()

def existsNaNDataframe(df, column='Saldo'):
    print("Nan Values:",df[column].isnull().values.any()) #There is nan
    print("Nan Values Indexes:",list(df.loc[pandas.isna(df[column]), :].index)) #return the indexes
    print(df.loc[pandas.isna(df[column]), :]) # return the rows with nan

def parseDataframe(df):
    df=df.dropna(subset=['Saldo']).reset_index(drop=True)
    df['Saldo']=df['Saldo'].apply(str2num)
    df= df.dropna(subset=['Valor.1']).reset_index(drop=True)
    df['Valor.1'] =df['Valor.1'].apply(str2num)
    return df

#print(valor.to_string())
df=parseDataframe(df)
def balanceAmount(df, column='Saldo'):
    value0= df[column][0]
    valuet= df[column].iloc[-1]
    diffvalue=float(valuet)-float(value0)
    print("Inicio:", df['Valor'][1],value0,"€   Fim: ", df['Valor'].iloc[-1], valuet, "€  Resultado:", diffvalue,"€")
#print(df.dtypes)

balanceAmount(df)

#df= df.dropna(subset=['Valor.1']).reset_index(drop=True)
#df['Valor.1'] =df['Valor.1'].apply(str2num)

#print(df)

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


expensesAbove(df,30)
expensesBiggest(df)
## Falta fazer menu, guardar por data, fazer grafico 
