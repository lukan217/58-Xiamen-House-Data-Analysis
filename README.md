# 爬取厦门58同城二手房数据进行数据分析（二）

<a name="EtBdk"></a>
# 一、前言
书接上文：[爬取厦门58同城二手房数据进行数据分析（一）](https://zhuanlan.zhihu.com/p/329185040)<br />这一篇主要对上一篇文章爬取下来的数据进行一些探索性分析和可视化，并且建立一个简单的预测模型进行房价预测。
<a name="MULWk"></a>
# 二、数据分析及可视化
<a name="PNypY"></a>
## 2.1 数据预处理
首先导包，由于`seaborn`画图不支持中文显示，因此还需要加几行代码：
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题
```
读入数据，删除不需要分析的字段，以及删除存在缺失值的数据：
```python
data = pd.read_csv('data.csv')
data = data.drop(columns=['Unnamed: 0', 'title', 'url', '产权年限', 'location2'])
data = data[data['location1'] != '厦门周边'] # 删除厦门周边的数据
data = data.dropna()
data
```
最终得到的数据像这样子，去除缺失值后一共749行*16列：<br />![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607620860545-327f70ad-5f36-4bb7-bb43-d88b1d34f9b1.png#align=left&display=inline&height=170&margin=%5Bobject%20Object%5D&name=image.png&originHeight=320&originWidth=1421&size=73922&status=done&style=none&width=755)<br />
<br />为了方便后续的工作，我们在将数据做一些简单的处理：
```python
data['室'] = data['房屋户型'].apply(lambda x: int(x[0]))
data['厅'] = data['房屋户型'].apply(lambda x: int(x[2]))
data['卫'] = data['房屋户型'].apply(lambda x: int(x[4]))
data['均价'] = data['均价'].apply(lambda x: float(x.split('元')[0]))
data['房本面积'] = data['房本面积'].apply(lambda x: float(x[:-1]))
data['建筑年代'] = data['建筑年代'].apply(lambda x: int(x[:-1]))
data['总楼层'] = data['所在楼层'].apply(lambda x: int(x[4:-2]))
data['所在楼层'] = data['所在楼层'].apply(lambda x: x[0])
data['小区均价'] = data['小区均价'].apply(
    lambda x: float(x.split('元')[0]))
data['物业费'] = data['物业费'].apply(
    lambda x: float(x.split('元')[0]))
data['绿化率'] = data['绿化率'].apply(float)
data['车位信息'] = data['车位信息'].apply(int)
```
<a name="5U1Ug"></a>
## 2.1 单变量可视化
**价格分布**<br />厦门市的房价总体来说还是非常贵的，一平方米平均要四万多，一套下来得四百多万，买不起买不起
```python
sns.distplot(data['均价'])
data['均价'].mean()
sns.distplot(data['总价'])
data['总价'].mean()
```
![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607617912033-306d3e0a-325a-4ae7-bbff-1d91d715aa58.png#align=left&display=inline&height=230&margin=%5Bobject%20Object%5D&name=image.png&originHeight=459&originWidth=630&size=33390&status=done&style=none&width=315)<br />![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607617933393-ebd56488-b0a6-49f9-a770-c3f1982fe140.png#align=left&display=inline&height=217&margin=%5Bobject%20Object%5D&name=image.png&originHeight=435&originWidth=650&size=30509&status=done&style=none&width=325)<br />**房屋区域分布**<br />有将近一半的二手房都在岛内（思明和湖里）
```python
data['位置1'].value_counts().plot.pie(autopct='%.2f%%')
```
**![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607619988435-7f18f6b6-fa96-4fa9-b889-e04cedf4e344.png#align=left&display=inline&height=244&margin=%5Bobject%20Object%5D&name=image.png&originHeight=380&originWidth=414&size=26951&status=done&style=none&width=266)**<br />**房屋朝向分布**<br />选取前五种最受欢迎的房屋朝向，可以看出，有2/3的房子都是南北朝向：
```python
data['房屋朝向'].value_counts().head(5).plot.pie(autopct='%.2f%%')
```
**![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607620480819-0821a116-4e07-45f8-90cb-dcbe6a31ce44.png#align=left&display=inline&height=227&margin=%5Bobject%20Object%5D&name=image.png&originHeight=342&originWidth=392&size=21546&status=done&style=none&width=260)**<br />**房屋户型分布**<br />同样选取前五种最受欢迎的房屋朝向，可以发现3室2厅2卫的户型最受欢迎：
```python
data['房屋户型'].value_counts().head(5).plot.pie(autopct='%.2f%%')
```
**![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607620697989-2b958314-3805-4be7-8094-2e0237bc6f5d.png#align=left&display=inline&height=226&margin=%5Bobject%20Object%5D&name=image.png&originHeight=371&originWidth=473&size=25943&status=done&style=none&width=288)**<br />**装修情况分布**<br />二手房基本上都是装修好了的，只有不到10%的是毛坯（为啥二手房还有毛坯的？）
```python
data['装修情况'].value_counts().plot.pie(autopct='%.2f%%')
```
**![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607621101526-c6d1f02d-3bc1-4008-a9f0-7dbc1dbba179.png#align=left&display=inline&height=226&margin=%5Bobject%20Object%5D&name=image.png&originHeight=366&originWidth=451&size=24867&status=done&style=none&width=279)**<br />**
<a name="U23ut"></a>
## 2.2 多变量间关系及可视化
**地域与房价**<br />画出各个区域的每平方米价格的箱型图，果然，岛内的房价更可怕了，思明区接近6万/平米，更有12万/平米的天价房，湖里区也接近5万/平米，就算在同安和翔安这两个鸟不拉屎的地方一平米也要两万多了
```python
sns.boxplot(data=data, x='位置1', y='均价')
```
![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607621274809-4304e87f-f8bf-4720-b874-7cf6192373e5.png#align=left&display=inline&height=224&margin=%5Bobject%20Object%5D&name=image.png&originHeight=429&originWidth=662&size=25464&status=done&style=none&width=346)<br />地域与其他变量<br />将数据做一个聚合，取平均，可以发现，岛内的房子都比较老，大概都在2000年上下（因为没地方可建了吧），而岛外基本上都在2010年左右，而且岛内的房子就只有十三四层，而岛外的房子有二十层左右，面积也相对来说比岛内的小一点
```python
data.groupby(by=['位置1'])['总价','房本面积','建筑年代','总楼层'].mean()
```
![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607621696738-bc35f9ce-7418-460b-a873-a5bf5001c76c.png#align=left&display=inline&height=179&margin=%5Bobject%20Object%5D&name=image.png&originHeight=278&originWidth=503&size=25370&status=done&style=none&width=324)<br />**<br />**<br />**建筑年代与房价**<br />看上去好像越老的房子越贵，上世纪末建的房子最值钱，而最近几年建的房子都不怎么值钱，当然这也跟我们之前分析的区域有关，因为最近建的房子基本都在岛外，所以当然不怎么值钱
```python
data.groupby(by='建筑年代')['均价'].mean().plot()
```
![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607623479793-fe610cab-0954-4669-afff-3824eabb0d12.png#align=left&display=inline&height=235&margin=%5Bobject%20Object%5D&name=image.png&originHeight=438&originWidth=640&size=35780&status=done&style=none&width=343)<br />**所在楼层与房价**<br />一般来说，大家都不太喜欢低楼层的房子，因为太吵了，当然太高也不行，这种关系，也反映在房价中：
```python
sns.barplot(x='所在楼层', y='均价', data=data)
```
![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607622112702-1862764d-a020-4303-8938-244724f22518.png#align=left&display=inline&height=224&margin=%5Bobject%20Object%5D&name=image.png&originHeight=447&originWidth=663&size=20120&status=done&style=none&width=331.5)<br />**<br />再来看看厦门哪个小区的房子最贵吧，这里选取小区均价最高的15个小区：
```python
data.groupby(by='小区名')['小区均价'].mean().sort_values(ascending=False).head(15).plot(kind='barh')
```
![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607651743576-9563d729-5c92-4f5c-a10d-1d85574a67fe.png#align=left&display=inline&height=236&margin=%5Bobject%20Object%5D&name=image.png&originHeight=410&originWidth=850&size=51836&status=done&style=none&width=489)
<a name="4zrss"></a>
## 2.3 地理可视化
前阵子刚好接触到百度地图的API，非常强大，就顺手做个地图可视化吧！<br />首先需要去百度地图开发者官网（ [http://lbsyun.baidu.com/](https://link.jianshu.com/?t=http://lbsyun.baidu.com/)）注册一个密钥，然后创建两个应用，一个是服务端的，用来使用Python获取小区坐标，一个是浏览器端的，用来通过修改html源代码创建热力图，具体实现可以参考这篇文章：[Python使用百度地图API实现地点信息转换及房价指数热力地图](https://blog.csdn.net/ebzxw/article/details/80265796)<br />最后生成的效果如下图所示，可以看出，厦门市最贵的地段基本上就在火车站周围那一块：<br />![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607655308085-a39295fa-8bad-495b-80b8-8fedada0be79.png#align=left&display=inline&height=433&margin=%5Bobject%20Object%5D&name=image.png&originHeight=866&originWidth=1089&size=947685&status=done&style=none&width=544.5)<br />_ps: 这里可视化原本想使用 folium，但是存在 folium包存在两个问题，一个是热力图存在 bug，没有渐变效果，另外一个是因为我坐标采用的是百度的坐标，百度的坐标是经过加密的，用在 folium上会存在坐标偏移的情况，故弃用_
<a name="myLOD"></a>
# 三、预测模型
以每平方米价格为因变量，其余变量为自变量，并将分类变量使用 LabelEncoder 编码，将测试集与训练集以2：8的比例分割：
```python
x=data.drop(columns=['小区均价','总价','均价','房屋户型','小区名'])
y=data['均价']
for col in ['位置1','房屋朝向','一手房源','所在楼层','装修情况']:
    le = LabelEncoder()
    x[col]=le.fit_transform(x[col])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
由于自变量中存在很多分类变量，因此考虑使用树模型进行预测，由于树模型本身就有着特征选择的功能，因此，不做特征选择，直接跑模型：<br />**决策树**
```python
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
print(f'决策树绝对值误差：{mean_absolute_error(dt.predict(x_test),y_test)}')
```
**随机森林**
```python
rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
rf.fit(x_train, y_train)
print(f'随机森林绝对值误差：{mean_absolute_error(rf.predict(x_test),y_test)}')
```
**Catboost**
```python
cb=CatBoostRegressor()
cb.fit(x_train, y_train)
print(f'Catboost绝对值误差：{mean_absolute_error(cb.predict(x_test),y_test)}')
```
**结果对比**

|  | 决策树 | 随机森林 | catboost |
| --- | --- | --- | --- |
| 绝对值误差 | 2885.81 | 2286.76 | 2347.04 |

**特征重要性**<br />用随机森林输出特征重要性看看：
```python
fi = pd.DataFrame(
    {'x': x.columns, 'feature_importance': rf.feature_importances_})
fi = fi.sort_values(by='feature_importance',ascending=False)
sns.barplot(x='feature_importance', y='x', data=fi)
```
![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607667452772-a2ae6636-935b-411c-b9fa-864472bf30a5.png#align=left&display=inline&height=210&margin=%5Bobject%20Object%5D&name=image.png&originHeight=421&originWidth=686&size=33317&status=done&style=none&width=343)<br />啊这，小区均价一枝独秀，解释力度太大了，把其他特征的信息都全部吃下去了，为了更好的解释其他特征与每平方米价格的关系，我们考虑把它排除在外，再输出一次特征重要性：<br />![image.png](https://cdn.nlark.com/yuque/0/2020/png/764062/1607668343722-a06abc5d-4305-4e67-a2ea-ee97a3a68a0b.png#align=left&display=inline&height=215&margin=%5Bobject%20Object%5D&name=image.png&originHeight=429&originWidth=687&size=34008&status=done&style=none&width=343.5)<br />这次就好点了，预测的绝对值误差虽然变成了四千，预测效果变差了，但是解释力度提高了，对房价影响最大的前五个特征为：位置1（区域）、物业费（反映小区的质量）、容积率（反映小区的居住的舒适度）、总楼层、建筑年代，而房屋朝向、所在楼层和装修情况这些特征居然没有想象中的那么重要，看来在厦门，**决定一套房子价格的是房子所在小区的属性，而不是你这套房子本身的属性**。
<a name="xHnNs"></a>
# 四、小结
好了，又一篇文章水完了，这篇文章还是花了我不少时间的，尤其是在研究怎么画图上，看来可视化这方面还是得继续学习一下啊！这个月总体来说还是比较忙的，希望能够坚持每周写一篇吧，下周可能会开始写一些算法的学习笔记。<br />

