# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def plot_3d_decision_boundary(X, y, feature_names, classifier, classifier_name):
    """
    3D决策边界可视化（原有函数，基本可用）
    """
    # 选择三个特征
    selected_features_3d = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
    feature_indices_3d = [feature_names.index(f) for f in selected_features_3d]
    X_3d = X[:, feature_indices_3d]
    y_3d = y

    # 创建3D画布
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制样本点
    colors = ['royalblue', 'orange']
    labels = ['Setosa', 'Versicolor']
    for i in range(2):
        mask = y_3d == i
        ax.scatter(
            X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
            c=colors[i], label=labels[i], s=80, edgecolors='black', alpha=0.8
        )

    # 创建网格，用于绘制决策平面
    x_min, x_max = X_3d[:, 0].min() - 1, X_3d[:, 0].max() + 1
    y_min, y_max = X_3d[:, 1].min() - 1, X_3d[:, 1].max() + 1
    z_min, z_max = X_3d[:, 2].min() - 1, X_3d[:, 2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    zz = np.zeros_like(xx)

    # 使用分类器的决策边界公式来计算z值
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = -(classifier.intercept_[0] + classifier.coef_[0][0] * xx[i, j] +
                         classifier.coef_[0][1] * yy[i, j]) / classifier.coef_[0][2]

    # 绘制决策平面
    ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='coolwarm', edgecolors='none')

    # 设置轴标签
    ax.set_xlabel(selected_features_3d[0], fontsize=12)
    ax.set_ylabel(selected_features_3d[1], fontsize=12)
    ax.set_zlabel(selected_features_3d[2], fontsize=12)
    ax.set_title(f'{classifier_name} 3D Decision Boundary', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'decision_boundary_3d_{classifier_name.replace(" ", "_")}.png', dpi=300)
    plt.show()


def plot_3d_probability_map_fixed(X, y, classifier, feature_names, classifier_name):
    """
    修复版的3D概率图可视化（任务三）
    """
    # 选择三个特征
    selected_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
    feature_indices = [feature_names.index(f) for f in selected_features]
    X_3d = X[:, feature_indices]

    # 创建2x2的子图布局
    fig = plt.figure(figsize=(16, 12))

    # 1. 3D散点图（概率着色）
    ax1 = fig.add_subplot(221, projection='3d')

    # 预测每个数据点的概率
    proba = classifier.predict_proba(X_3d)

    # 为Setosa类别着色
    scatter1 = ax1.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                           c=proba[:, 0], cmap='Blues', s=60,
                           edgecolors='black', alpha=0.8)

    ax1.set_xlabel(selected_features[0])
    ax1.set_ylabel(selected_features[1])
    ax1.set_zlabel(selected_features[2])
    ax1.set_title(f'Setosa Probability (3D Scatter)')
    plt.colorbar(scatter1, ax=ax1, shrink=0.6, label='P(Setosa)')

    # 2. Versicolor类别的3D散点图
    ax2 = fig.add_subplot(222, projection='3d')
    scatter2 = ax2.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                           c=proba[:, 1], cmap='Oranges', s=60,
                           edgecolors='black', alpha=0.8)

    ax2.set_xlabel(selected_features[0])
    ax2.set_ylabel(selected_features[1])
    ax2.set_zlabel(selected_features[2])
    ax2.set_title(f'Versicolor Probability (3D Scatter)')
    plt.colorbar(scatter2, ax=ax2, shrink=0.6, label='P(Versicolor)')

    # 3. 2D投影的概率等高线图（固定第三个维度）
    ax3 = fig.add_subplot(223)

    # 固定第三个特征为平均值
    z_fixed = X_3d[:, 2].mean()
    x_min, x_max = X_3d[:, 0].min() - 0.5, X_3d[:, 0].max() + 0.5
    y_min, y_max = X_3d[:, 1].min() - 0.5, X_3d[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))

    grid_points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, z_fixed)]
    probabilities = classifier.predict_proba(grid_points)
    prob_setosa = probabilities[:, 0].reshape(xx.shape)

    # 绘制等高线
    contour = ax3.contourf(xx, yy, prob_setosa, levels=20, alpha=0.6, cmap='Blues')
    ax3.contour(xx, yy, prob_setosa, levels=10, colors='black', linewidths=0.5)

    # 绘制数据点
    scatter3 = ax3.scatter(X_3d[:, 0], X_3d[:, 1], c=y, cmap='coolwarm',
                           s=50, edgecolors='black')

    ax3.set_xlabel(selected_features[0])
    ax3.set_ylabel(selected_features[1])
    ax3.set_title(f'Probability Contour (Z fixed at {z_fixed:.2f})')
    plt.colorbar(contour, ax=ax3, label='P(Setosa)')

    # 4. 概率曲面图（2.5D可视化）
    ax4 = fig.add_subplot(224, projection='3d')

    # 创建概率曲面
    surf = ax4.plot_surface(xx, yy, prob_setosa, cmap='viridis',
                            alpha=0.7, linewidth=0, antialiased=True)

    # 在曲面上绘制数据点
    proba_points = classifier.predict_proba(X_3d)[:, 0]
    ax4.scatter(X_3d[:, 0], X_3d[:, 1], proba_points,
                c=y, cmap='coolwarm', s=50, edgecolors='black')

    ax4.set_xlabel(selected_features[0])
    ax4.set_ylabel(selected_features[1])
    ax4.set_zlabel('P(Setosa)')
    ax4.set_title('Probability Surface (2.5D)')
    plt.colorbar(surf, ax=ax4, shrink=0.6, label='P(Setosa)')

    plt.suptitle(f'{classifier_name} - 3D Probability Visualization',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'3d_probability_{classifier_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()