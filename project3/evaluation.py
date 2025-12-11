# evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def evaluate_classifiers(X, y, classifiers):
    """
    评估多个分类器的性能（任务四）
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for name, clf in classifiers:
        print(f"Evaluating {name}...")

        try:
            # 交叉验证
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)

            # 计算各项指标
            report = classification_report(y_test, y_pred, output_dict=True)

            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }

            # 绘制混淆矩阵
            plot_confusion_matrix(y_test, y_pred, name)

            # 绘制ROC曲线（如果是二分类）
            if len(np.unique(y)) == 2:
                plot_roc_curve(y_test, y_pred_proba, name)

        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = {'error': str(e)}

    # 绘制性能比较图
    plot_performance_comparison(results)

    return results


def plot_confusion_matrix(y_true, y_pred, classifier_name):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Setosa', 'Versicolor'],
                yticklabels=['Setosa', 'Versicolor'])
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{classifier_name.replace(" ", "_")}.png', dpi=300)
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, classifier_name):
    """
    绘制ROC曲线
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {classifier_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curve_{classifier_name.replace(" ", "_")}.png', dpi=300)
    plt.show()


def plot_performance_comparison(results):
    """
    绘制性能比较图
    """
    # 过滤掉有错误的结果
    valid_results = {k: v for k, v in results.items() if 'error' not in v}

    if not valid_results:
        print("No valid results to plot")
        return

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    classifiers = list(valid_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, metric in enumerate(metrics):
        values = [valid_results[clf][metric] for clf in classifiers]

        bars = axes[i].bar(classifiers, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)

        # 在柱子上添加数值
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom')

    plt.suptitle('Classifier Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('classifier_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(X, y, feature_names):
    """
    绘制特征重要性
    """
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(X.shape[1]), importances[indices], color='lightseagreen')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.title("Feature Importances (Random Forest)")
    plt.xlabel("Features")
    plt.ylabel("Importance")

    # 在柱子上添加数值
    for bar, importance in zip(bars, importances[indices]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{importance:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()