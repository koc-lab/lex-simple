import pandas as pd
import matplotlib as plt
import re
import numpy as np
import math

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [13, 5.6]

f = open('handwritten_complex_words.txt','r',encoding='ascii',errors="ignore")
additional_words = f.read()
additional_words = re.sub(', ', ',', additional_words)
additional_words_list = additional_words.split(',')


df_complex = pd.read_excel('edited_complex_words_zipf.xlsx')
df_common = pd.read_excel('edited_common_words_zipf.xlsx')

df_complex.sort_values(by='Subtlex Zipf Value', ascending = False, inplace=True)

"""
plt.plot(np.arange(df_complex.shape[0]),df_complex['Law Zipf Value'],'y.',label='Law')
plt.plot(np.arange(df_complex.shape[0]),df_complex['Subtlex Zipf Value'],'b.',label='Subtlex')
plt.legend()
plt.title("Zipf Value Comparison")
plt.ylabel("Zipf Values")
plt.xlabel("Words")
"""


std_legal = np.std(df_common["Law Zipf Value"].to_numpy())
std_subtlex = np.std(df_common["Subtlex Zipf Value"].to_numpy())
mean_legal = np.mean(df_common["Law Zipf Value"].to_numpy())
mean_subtlex = np.mean(df_common["Subtlex Zipf Value"].to_numpy())
arr_zipf_diff=df_common["Zipf Values Difference"].to_numpy()

print(mean_subtlex)
std_diff = np.std(arr_zipf_diff)
print(2*std_diff)
print(np.mean(arr_zipf_diff))

"""
plt.plot(df_common['Subtlex Zipf Value'],df_common['Law Zipf Value'],'b.',linewidth = 0.1)
x=[1,8]
dikme3_1 = [4-std_subtlex,4-std_subtlex]
dikme3_2 = [1,8]
plt.plot(x,x+2*std_diff,'r-',label='2 std difference')
plt.plot(x,x,'y-',label = "equal scores")
plt.plot(dikme3_1,dikme3_2,'r-',linewidth=1.5,label = "4-std(subtlex_values)")
plt.title("Zipf Value Comparison")
plt.ylabel("Law Values")
plt.xlabel("Subtlex Values")
plt.legend()
plt.show()



df_complex.sort_values(by='Subtlex Zipf Value', ascending = False, inplace=True)
plt.plot(list(df_complex['Subtlex Zipf Value']),list(df_complex['Law Zipf Value']),'b.',linewidth=0.1,markeredgewidth=5,markersize=0.5)
plt.plot([1,8],[1,8],'g-')
plt.title("Zipf Value Comparison")
plt.ylabel("Law Values")
plt.xlabel("Subtlex Values")
plt.show()
"""

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Zipf Scale Analysis')

ax1.plot(df_common['Subtlex Zipf Value'],df_common['Law Zipf Value'],'b.',linewidth = 0.1,markeredgewidth=5,markersize=0.5)
ax1.set_xlim([1, 8])
ax1.set_ylim([1, 8])
x=[1,8]
dikme3_1 = [4-std_subtlex,4-std_subtlex]
dikme3_2 = [1,8]
ax1.plot(x,x+2*std_diff,'r-',label='2 std difference')
ax1.plot(x,x,'g-',label = "equal scores")
ax1.plot(dikme3_1,dikme3_2,'y-',linewidth=1.5,label = "4-std(subtlex_values)")
#ax1.legend(loc='upper left')

ax2.plot(list(df_complex['Subtlex Zipf Value']),list(df_complex['Law Zipf Value']),'b.',linewidth=0.1,markeredgewidth=5,markersize=0.5)
ax2.set_xlim([1, 8])
ax2.set_ylim([1, 8])
ax2.plot([1,8],[1,8],'g-')
#ax2.legend(loc='upper left')

plt.show()
