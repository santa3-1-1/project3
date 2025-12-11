# main.py
from data_processing import load_data
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from visualization import plot_3d_decision_boundary, plot_3d_probability_map_fixed
from evaluation import evaluate_classifiers, plot_feature_importance
import pandas as pd

if __name__ == "__main__":
    # 加载数据（只选择 Setosa 和 Versicolor）
    X, y, feature_names, target_names, df = load_data()

    print("数据基本信息:")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print(f"类别分布: Setosa: {sum(y == 0)}, Versicolor: {sum(y == 1)}")

    # 扩展分类器列表
    classifiers = [
        ('SVM Linear', SVC(kernel="linear", C=1.0, probability=True, random_state=42)),
        ('SVM RBF', SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]

    # 任务四：模型评估和比较
    print("\n=== 任务四：模型性能评估 ===")
    results = evaluate_classifiers(X, y, classifiers)

    # 输出结果表格
    results_df = pd.DataFrame(results).T
    print("\n模型性能比较:")
    print(results_df.round(4))

    # 绘制特征重要性
    plot_feature_importance(X, y, feature_names)

    # 选择最佳模型进行3D可视化
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_model_name = max(valid_results, key=lambda x: valid_results[x]['accuracy'])
        best_model = dict(classifiers)[best_model_name]

        print(f"\n最佳模型: {best_model_name}")

        # 使用最佳模型进行3D可视化
        selected_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
        feature_indices = [feature_names.index(f) for f in selected_features]
        X_selected = X[:, feature_indices]

        # 重新训练最佳模型
        best_model.fit(X_selected, y)

        # 任务二：3D决策边界
        print("可视化3D决策边界...")
        plot_3d_decision_boundary(X_selected, y, feature_names, best_model, f"Best Model: {best_model_name}")

        # 任务三：修复版的3D概率图
        print("可视化3D概率图...")
        plot_3d_probability_map_fixed(X_selected, y, best_model, feature_names, f"Best Model: {best_model_name}")
    else:
        print("没有有效的模型结果可用于可视化")

    print("\n所有任务完成！")