import pandas as pd 
import numpy as np   

data_train = pd.read_csv("data/train.csv")


#此代码是为了显示图表中的中文
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


# 各个属性与Survived 的关系  
import matplotlib.pyplot as plt

fig = plt.figure()
fig.set(alpha=0.3)              # 设定图表颜色alpha参数 值越大 颜色越深 


# 统计获救情况
plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图 
data_train.Survived.value_counts().plot(kind='bar')  # 柱状图 plots a bar graph of those who surived vs those who did not. 
plt.title(u"获救情况 (1为获救)")    # puts a title on our graph
plt.ylabel(u"人数")  

# 统计各等舱获救比例
plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

#按照年龄看获救情况
plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")                         # sets the y axis lable
plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
plt.title(u"按年龄看获救分布 (1为获救)")

# 各等级的乘客年龄分布
plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   #密度图  plots a kernel desnsity estimate of the subset of the 1st class passanges's age
data_train.Age[data_train.Pclass == 2].plot(kind='kde')   #密度图
data_train.Age[data_train.Pclass == 3].plot(kind='kde')   #密度图
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度") 
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.

# 各登船口岸上船人数
plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")  
plt.show()


# 图表的信息显示：
# 被救的人300多点，不到半数；
# 3等舱乘客灰常多；遇难和获救的人年龄似乎跨度都很广；
# 3个不同的舱年龄总体趋势似乎也一致，2/3等舱乘客20岁多点的人最多，
# 1等舱40岁左右的最多(→_→似乎符合财富和年龄的分配哈，咳咳，别理我，我瞎扯的)；
# 登船港口人数按照S、C、Q递减，而且S远多于另外俩港口。

# 这个时候我们可能会有一些想法了：
# 不同舱位/乘客等级可能和财富/地位有关系，最后获救概率可能会不一样
# 年龄对获救概率也一定是有影响的，毕竟前面说了，副船长还说『小孩和女士先走』呢
# 和登船港口是不是有关系呢？也许登船港口不同，人的出身地位不同？