# plotdata.py
# Author: bmalbusca
# Year: 2024
# License: GNU General Public License v3.0 (GPL-3.0)
# License URL: https://github.com/bmalbusca/montlySavings/blob/master/LICENSE
# Description: Description: This module contains functions for plotting financial data using matplotlib.

import matplotlib.pyplot as plt

# Usage example:
# plotOverview_EI(processor.getExpensesOverview(df),processor.getIncomeOverwiew(df),processor.getMarginOverview(df),fig,ax)
def plotOverview_EI(expense_values,income_values,margin_values, fig,ax,label=True,accumulative=False):
    """
    Plot overview of expenses, income, and margin over time.

    Parameters:
    - expense_values (list): List containing expense data, pos-0 time,pos-1 inverted values, pos-2 accumulated values.
    - income_values (list): List containing income data, pos-0 time, pos-1 values,pos-2 accumulated values.
    - margin_values (list): List containing margin data - sum of all transation value.
    - fig (matplotlib.figure.Figure): Figure object.
    - ax (matplotlib.axes._axes.Axes): Axes object.
    - label (bool): Whether to display labels on the bars (default is True).
    - accumulative (bool): Whether to show accumulative values (default is False).
    """
    # Function implementation...
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

# Usages examples: 
# plotSimple_EI([x,y],[x2,y2],fig, ax)
# plotSimple_EI([sdf['Mov'],-sdf['Valor.1']],[sdf2['Mov'],sdf2['Valor.1']],fig, ax)
# plotSimple_EI([processor.getExpensesMonthly(df)[0],-processor.getExpensesMonthly(df)[1]],processor.getIncomeMonthly(df),fig, ax)  
def plotSimple_EI(expense_values,income_values,fig,ax,label=True):
    """
    Plot simple expenses and income over time.

    Parameters:
    - expense_values (list): List containing expense data, pos-0 time,pos-1 values
    - income_values (list): List containing income data,pos-0 time,pos-1 values
    - fig (matplotlib.figure.Figure): Figure object.
    - ax (matplotlib.axes._axes.Axes): Axes object.
    - label (bool): Whether to display labels on the bars (default is True).
    """

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


