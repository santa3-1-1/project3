import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_iris

# 加载Iris数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# 查看数据集的一部分
print(df[50:100])

# 数据预处理
df = df.dropna()  # 删除缺失值
df['species'] = df['species'].astype('category').cat.codes  # 将类别型数据转化为数值型

# 创建多个子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # 创建2行2列的子图

# 第一个子图：sepal_length 的箱线图
sns.boxplot(x='species', y='sepal length (cm)', data=df, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Sepal Length by Species')

# 第二个子图：sepal_width 的箱线图
sns.boxplot(x='species', y='sepal width (cm)', data=df, ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Sepal Width by Species')

# 第三个子图：petal_length 的箱线图
sns.boxplot(x='species', y='petal length (cm)', data=df, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Petal Length by Species')

# 第四个子图：petal_width 的箱线图
sns.boxplot(x='species', y='petal width (cm)', data=df, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Petal Width by Species')

# 调整布局，使得子图之间不重叠
plt.tight_layout()

# 保存Matplotlib图像
plt.savefig('box_plots.png', dpi=300)  # 高分辨率保存

# 显示图像
plt.show()

# 使用Plotly绘制交互式散点图：绘制每一对特征之间的散点图
fig1 = px.scatter(df, x='sepal length (cm)', y='sepal width (cm)', color='species', title="Sepal Length vs Sepal Width", template='plotly_dark')
fig2 = px.scatter(df, x='sepal length (cm)', y='petal length (cm)', color='species', title="Sepal Length vs Petal Length", template='plotly_dark')
fig3 = px.scatter(df, x='sepal length (cm)', y='petal width (cm)', color='species', title="Sepal Length vs Petal Width", template='plotly_dark')
fig4 = px.scatter(df, x='sepal width (cm)', y='petal length (cm)', color='species', title="Sepal Width vs Petal Length", template='plotly_dark')
fig5 = px.scatter(df, x='sepal width (cm)', y='petal width (cm)', color='species', title="Sepal Width vs Petal Width", template='plotly_dark')
fig6 = px.scatter(df, x='petal length (cm)', y='petal width (cm)', color='species', title="Petal Length vs Petal Width", template='plotly_dark')

# 保存Plotly图像
fig1.write_image('scatter_plot_sepal_length_vs_sepal_width.png')
fig2.write_image('scatter_plot_sepal_length_vs_petal_length.png')
fig3.write_image('scatter_plot_sepal_length_vs_petal_width.png')
fig4.write_image('scatter_plot_sepal_width_vs_petal_length.png')
fig5.write_image('scatter_plot_sepal_width_vs_petal_width.png')
fig6.write_image('scatter_plot_petal_length_vs_petal_width.png')

# 显示交互式图表
fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()
fig6.show()
