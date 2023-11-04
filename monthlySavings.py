import os, types
import pandas
import tabula

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


#expensesAbove(df,30)
#expensesBiggest(df)

if __name__ == "__main__":
    read_cache = None
    df=None
    debug=False


    file_list =getLocalFiles()
    df=collectCacheData(debug=False)

    if isinstance(df, type(None)) == True:
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

    else:
        cache_file_list= df['Filename'].unique()
        new_file=False
        for fdir in list(file_list.keys()):
            for filename in file_list[fdir]:
                complete_dir = fdir+'/'+filename
                if complete_dir  not in cache_file_list:
                    s = readPDFSantander(file_dir=complete_dir,store_incache=False)
                    df=pandas.concat([df,s])
                    new_file=True
        if new_file:
            df=parseDataframe(df)
            if debug:
                print(df.to_string())
            df.to_pickle('cached_dataframe.pkl') # will be stored in current directory


    mask = (df['Mov'] >= '2023-07-11') 
    #print(mask.to_string())
    #print(df[mask])
    balanceAmount(df[mask])



## Falta fazer menu, guardar por data, fazer grafico 
