# Пример по которому делался проект туть - https://habr.com/ru/company/ods/blog/328372/


# Serhii: Загружаем необходимые библиотеки 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rc, plot
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split

# Serhii added
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

# Serhii: Загружаем файл с даннными
print("Импорт файла!")
df = pd.read_csv('dataset.csv')

# Serhii: Смотрим первых 5 строк данных
print("Первых 5 строк после загрузки файла!")
print(df.head(5))

# Serhii: Предобработка данных(указываем что формат данных в столбике будет целочисленный int64)
df['One_click_order'] = df['One_click_order'].astype('int64')

# Serhii: Тест данных после обработки
print("Первых 5 строк данных после обработки!")
print(df.head(5))
print("Показываем колонку One_click_order!")
print(df['One_click_order'])


# Serhii: Обучение алгоритма и построение матрицы ошибок
X = df.drop('One_click_order', axis=1)
y = df['One_click_order']


# Serhii: Делим выборку на train и test, все метрики будем оценивать на тестовом датасете
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,  test_size=0.33, random_state=42)


# Serhii: Обучаем ставшую родной логистическую регрессию


# Serhii:Не запускался код с хабра https://habr.com/ru/company/ods/blog/328372/ (Логическая регресия) 
# Решение: переделал инициализацию логической регресии по официальной документации https://scikit-learn.org/stable/modules/preprocessing.html
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)  # apply scaling on training data
print(pipe.score(X_test, y_test))  # apply scaling on testing data, without leaking training data.



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


font = {'size' : 15}

plt.rc('font', **font)


# Serhii: тут была ошибка, нужно было импортнуть confusion_matrix
# from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_test, pipe.predict(X_test))
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cnf_matrix, classes=['Non-churned', 'Churned'],
                      title='Confusion matrix')

# Serhii: создаем картинку conf_matrix.png с метрикой и срхраняем ее в текущую директорию проекта
plt.savefig("conf_matrix.png")

# Serhii: показываем картинку после запуска проекта
plt.show()    