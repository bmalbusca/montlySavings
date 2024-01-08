import matplotlib.pyplot as plt

# Usage examples: #plotOverview_EI(processor.getExpensesOverview(df),processor.getIncomeOverwiew(df),processor.getMarginOverview(df),fig,ax)
def plotOverview_EI(expense_values,income_values,margin_values, fig,ax,label=True,accumulative=False):

        x1 = expense_values[0]
        y1 = expense_values[1]

        x2 = income_values[0]
        y2 = income_values[1]

        if(accumulative):
            y3 = expense_values[2]
            y4 = income_values[2]
            x5= margin_values[0]
            y5= margin_values[2]

            line0=ax.plot(x5,y5,color='k', alpha=0.1)
            acumulative1=ax.stackplot(x1, y3,alpha=0.2, color='red')
            acumulative2=ax.stackplot(x2, y4,alpha=0.1, color='green')

        # plot
        width = 2
        bar1=ax.bar(x1, y1, color='red',width=0.5*width)
        bar2=ax.bar(x2,y2,color='green',alpha=0.5)
        
        #fig.suptitle('This is a somewhat long figure title', fontsize=16)
        ax.set(ylabel='Value (€)', xlabel='Time', title='Expenses x Income during time')
        plt.xticks(rotation=30,ha='right')
        if label:
            ax.bar_label(bar1, rotation=30)
            ax.bar_label(bar2,rotation=30)
 
# usages examples 
#plotSimple_EI([x,y],[x2,y2],fig, ax)
#plotSimple_EI([sdf['Mov'],-sdf['Valor.1']],[sdf2['Mov'],sdf2['Valor.1']],fig, ax)
#plotSimple_EI([processor.getExpensesMonthly(df)[0],-processor.getExpensesMonthly(df)[1]],processor.getIncomeMonthly(df),fig, ax)  
def plotSimple_EI(expense_values,income_values,fig,ax,label=True):

        x1 = expense_values[0]
        y1 = expense_values[1]

        x2 = income_values[0]
        y2 = income_values[1]

        # plot
        width = 2
        bar1=ax.bar(x1, y1, color='red',width=0.5*width)
        bar2=ax.bar(x2,y2,color='green',alpha=0.5)
        
        #fig.suptitle('This is a somewhat long figure title', fontsize=16)
        ax.set(ylabel='Value (€)', xlabel='Time', title='Expenses x Income during time')
        plt.xticks(rotation=30,ha='right')
        if label:
            ax.bar_label(bar1, rotation=30)
            ax.bar_label(bar2,rotation=30)


