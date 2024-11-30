import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title):
    # Визначаємо границі області
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Прогнозуємо класи для кожної точки в сітці
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Візуалізуємо області
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

    # Відображаємо початкові точки
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()