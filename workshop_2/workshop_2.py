import pandas
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_moons, make_circles, make_classification


u

# Read in data
iris_data = pandas.read_csv("workshop_2/data/iris.csv")

# Subset data
X = iris_data[["sepal.length", "sepal.width"]]
y = iris_data["variety"]
X = iris_data[(y == "Versicolor") | (y == "Virginica")]
X = X[["sepal.length", "sepal.width"]]
y = y[(y == "Versicolor") | (y == "Virginica")]

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# Plot raw data
plt.figure()
plt.scatter(X["sepal.length"][(y == "Versicolor")],
            X["sepal.width"][(y == "Versicolor")],
            label = "Versicolor",
            c="red",
            cmap=cm_bright, edgecolors='k')
plt.scatter(X["sepal.length"][(y == "Virginica")],
            X["sepal.width"][(y == "Virginica")],
            label = "Virginica",
            c="blue",
            cmap=cm_bright, edgecolors='k')
# plt.scatter(X["sepal.length"], X["sepal.width"], label = ["Versicolor", "Virginica"],
#                c=y.astype("category").cat.codes, cmap=cm_bright, edgecolors='k')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')
plt.xlim(4, 8.5)
plt.ylim(1.5, 4.5)
plt.yticks(np.arange(1.5, 4, step=0.5))
plt.xticks(np.arange(4, 8.5, step=0.5))
plt.legend(loc='upper left')
plt.savefig("workshop_2/figures/iris_raw_data.png")


# Aside on separability and sample size

# Creates just a figure and only one subplot
fig, axes = plt.subplots(2, 2)

X, y = make_classification(n_samples = 200, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 4 * rng.uniform(size=X.shape)

axes[0,0].scatter(X[:,0],
            X[:,1],
            s = 10,
            #label = "Versicolor",
            c=y,
            cmap=cm_bright, edgecolors='k')
#axes[0,0].set_xlabel('X1')
axes[0,0].set_ylabel('X2')
axes[0,0].set_title('')
axes[0,0].set_ylim(-6, 8)
axes[0,0].set_xlim(-3, 7)
#axes[0,0].set_yticks(np.arange(min(X[:,1]) - 1, max(X[:,1]) + 1, step=1))
axes[0,0].set_yticks(np.arange(-6, 8, step=2))
axes[0,0].set_xticks(())
#axes[0,0].set_xticks(np.arange(min(X[:,0]) - 1, max(X[:,0]) + 1, step=1))
#axes[0,0].set_xticks(np.arange(-3, 6, step=1))
#plt.legend(loc='upper left')
#plt.savefig("workshop_2/figures/linsep1.png")

h = 0.02
x_min, x_max = -3, 7
y_min, y_max = -6, 8
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

neigh = KNeighborsClassifier(n_neighbors = 5)
neigh.fit(X, y)
true_zero = np.logical_and((neigh.predict(X) == y), (y == 0))
false_zero = np.logical_and((neigh.predict(X) != y), (y == 0))
true_one = np.logical_and((neigh.predict(X) == y), (y == 1))
false_one = np.logical_and((neigh.predict(X) != y), (y == 1))

score = neigh.score(X, y)
score
Z = neigh.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
axes[0,1].set_title("K=" + str(5) + ", " + "Accuracy = " + str(score))
# Plot the training points
axes[0,1].contourf(xx, yy, Z, cmap=cm, alpha=.8)
axes[0,1].scatter(X[:,0][true_zero], X[:,1][true_zero], s = 10, c='red', marker='o', cmap=cm_bright, edgecolors='k')
axes[0,1].scatter(X[:,0][false_zero], X[:,1][false_zero], s = 10, c='red', marker='x', cmap=cm_bright, edgecolors='k')
axes[0,1].scatter(X[:,0][true_one], X[:,1][true_one], s = 10, c='blue', marker='o', cmap=cm_bright, edgecolors='k')
axes[0,1].scatter(X[:,0][false_one], X[:,1][false_one], s = 10, c='blue', marker='x', cmap=cm_bright, edgecolors='k')
axes[0,1].set_ylim(-6, 8)
axes[0,1].set_xlim(-3, 7)
axes[0,1].set_xticks(())
axes[0,1].set_yticks(())
#axes[0,0].plt.savefig("workshop_2/figures/linpred1.png")

X, y = make_classification(n_samples = 200, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 0.5 * rng.uniform(size=X.shape)
#plt.figure()
axes[1,0].scatter(X[:,0],
            X[:,1],
            s = 10,
            #label = "Versicolor",
            c=y,
            cmap=cm_bright, edgecolors='k')
axes[1,0].set_xlabel('X1')
axes[1,0].set_ylabel('X2')
axes[1,0].set_title('')
axes[1,0].set_ylim(-6, 8)
axes[1,0].set_xlim(-3, 7)
#axes[1,0].set_yticks(np.arange(min(X[:,1]) - 1, max(X[:,1]) + 1, step=1))
axes[1,0].set_yticks(np.arange(-6, 8, step=2))
#axes[1,0].set_xticks(np.arange(min(X[:,0]) - 1, max(X[:,0]) + 1, step=1))
axes[1,0].set_xticks(np.arange(-3, 7, step=1))
#plt.legend(loc='upper left')
#plt.savefig("workshop_2/figures/linsep2.png")

h = 0.02
x_min, x_max = -3, 7
y_min, y_max = -6, 8
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
neigh = KNeighborsClassifier(n_neighbors = 5)
neigh.fit(X, y)
true_zero = np.logical_and((neigh.predict(X) == y), (y == 0))
false_zero = np.logical_and((neigh.predict(X) != y), (y == 0))
true_one = np.logical_and((neigh.predict(X) == y), (y == 1))
false_one = np.logical_and((neigh.predict(X) != y), (y == 1))

score = neigh.score(X, y) 
score
Z = neigh.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
axes[1,1].set_title("K=" + str(5) + ", " + "Accuracy = " + str(score))
# Plot the training points
axes[1,1].contourf(xx, yy, Z, cmap=cm, alpha=.8)
axes[1,1].scatter(X[:,0][true_zero], X[:,1][true_zero], s = 10, c='red', marker='o', cmap=cm_bright, edgecolors='k')
axes[1,1].scatter(X[:,0][false_zero], X[:,1][false_zero], s = 10, c='red', marker='x', cmap=cm_bright, edgecolors='k')
axes[1,1].scatter(X[:,0][true_one], X[:,1][true_one], s = 10, c='blue', marker='o', cmap=cm_bright, edgecolors='k')
axes[1,1].scatter(X[:,0][false_one], X[:,1][false_one], s = 10, c='blue', marker='x', cmap=cm_bright, edgecolors='k')
axes[1,1].set_ylim(-6, 8)
axes[1,1].set_xlim(-3, 7)
axes[1,1].set_xlabel('X1')
axes[1,1].set_xticks(np.arange(-3, 7, step=1))
axes[1,1].set_yticks(())
fig.savefig("workshop_2/figures/linpred.png")


# Plotting parms
h = 0.02
x_min, x_max = min(X["sepal.length"]) - 0.5, max(X["sepal.length"]) + 0.5
y_min, y_max = min(X["sepal.width"]) - 0.5, max(X["sepal.width"]) + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])



k_seq = [1, 3, 5, 10]
for i in k_seq:
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X, y)
    score = neigh.score(X, y) 
    Z = neigh.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax = plt.subplot(1, 1, 1)
    ax.set_title("K=" + str(i) + ", " + "Accuracy = " + str(score))
# Plot the training points
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    ax.scatter(X["sepal.length"], X["sepal.width"],
               c=y.astype("category").cat.codes, cmap=cm_bright, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    plt.savefig("workshop_2/figures/irisk" + str(i) + ".png")


h = 0.02
x_min, x_max = 4, 8.5
y_min, y_max = 1.5, 4.5
x_min, x_max = 4, 8.5
y_min, y_max = 1.5, 4.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

iris_data = pandas.read_csv("workshop_2/data/iris.csv")

# Subset data
X = iris_data[["sepal.length", "sepal.width"]]
y = iris_data["variety"]
X = iris_data[(y == "Versicolor") | (y == "Virginica")]
X = X[["sepal.length", "sepal.width"]]
y = y[(y == "Versicolor") | (y == "Virginica")]
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

fig, axes = plt.subplots(2, 4, figsize=(16,8))
k_seq = [1, 3, 5, 10]
j = 0
i = 1
X_test.columns[0]
for i in k_seq:
    print(j)
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, y_train)
    score_train = neigh.score(X_train, y_train)
    score_test = neigh.score(X_test, y_test)
    Z = neigh.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    true_vers_test = np.logical_and((neigh.predict(X_test) == y_test), [x == "Versicolor" for x in y_test])
    false_vers_test = np.logical_and((neigh.predict(X_test) != y_test), [x == "Versicolor" for x in y_test])
    true_virg_test = np.logical_and((neigh.predict(X_test) == y_test), [x == "Virginica" for x in y_test])
    false_virg_test = np.logical_and((neigh.predict(X_test) != y_test), [x == "Virginica" for x in y_test])
    true_vers_train = np.logical_and((neigh.predict(X_train) == y_train), [x == "Versicolor" for x in y_train])
    false_vers_train = np.logical_and((neigh.predict(X_train) != y_train), [x == "Versicolor" for x in y_train])
    true_virg_train = np.logical_and((neigh.predict(X_train) == y_train), [x == "Virginica" for x in y_train])
    false_virg_train = np.logical_and((neigh.predict(X_train) != y_train), [x == "Virginica" for x in y_train])
    axes[0,j].set_title("Train, K=" + str(i) + ", " + "Acc=" + str(round(score_train,2)), fontsize=6)
    axes[0,j].contourf(xx, yy, Z, cmap=cm, alpha=.8)
    axes[0,j].scatter(X_train[X_test.columns[0]][true_vers_train], X_train[X_test.columns[1]][true_vers_train],
               marker = 'o', c="red", cmap=cm_bright, edgecolors='k', s=12)
    axes[0,j].scatter(X_train[X_test.columns[0]][false_vers_train], X_train[X_test.columns[1]][false_vers_train],
                      marker = 'x', c="red", cmap=cm_bright, edgecolors='k', s=12)
    axes[0,j].scatter(X_train[X_test.columns[0]][true_virg_train], X_train[X_test.columns[1]][true_virg_train],
               marker = 'o', c="blue", cmap=cm_bright, edgecolors='k', s=12)
    axes[0,j].scatter(X_train[X_test.columns[0]][false_virg_train], X_train[X_test.columns[1]][false_virg_train],
                      marker = 'x', c="blue", cmap=cm_bright, edgecolors='k', s=12)
    axes[0,j].set_xlim(4, 8.5)
    axes[0,j].set_ylim(1.5, 4.5)
    axes[0,j].set_xticks(())
    axes[0,j].set_yticks(())
    axes[1,j].set_title("Test, K=" + str(i) + ", " + "Acc=" + str(round(score_test,2)), fontsize=6)
    axes[1,j].contourf(xx, yy, Z, cmap=cm, alpha=.8)
    axes[1,j].scatter(X_test[X_test.columns[0]][true_vers_test], X_test[X_test.columns[1]][true_vers_test],
               marker = 'o', c="red", cmap=cm_bright, edgecolors='k',s = 12)
    axes[1,j].scatter(X_test[X_test.columns[0]][false_vers_test], X_test[X_test.columns[1]][false_vers_test],
                      marker = 'x', c="red", cmap=cm_bright, edgecolors='k',s = 12)
    axes[1,j].scatter(X_test[X_test.columns[0]][true_virg_test], X_test[X_test.columns[1]][true_virg_test],
               marker = 'o', c="blue", cmap=cm_bright, edgecolors='k',s = 12)
    axes[1,j].scatter(X_test[X_test.columns[0]][false_virg_test], X_test[X_test.columns[1]][false_virg_test],
               marker = 'x', c="blue", cmap=cm_bright, edgecolors='k',s = 12)
    axes[1,j].set_xlim(4, 8.5)
    axes[1,j].set_ylim(1.5, 4.5)
    axes[1,j].set_xticks(())
    axes[1,j].set_yticks(())
    j = j+1

fig.savefig("workshop_2/figures/iris_k_train_test.png")

    
k_list = list(range(1,21))

cv_parms = {"n_neighbors": k_list}    
neigh = KNeighborsClassifier()

cross_validate(neigh, X, y=y, fit_params = cv_parms)

search = GridSearchCV(neigh, cv_parms, cv=3)
search.fit(X, y)

mean_scores = search.cv_results_['mean_test_score']
best_k = k_list[search.best_params_['n_neighbors'] - 1]
best_k
best_k_score = max(mean_scores)
best_k_score

plt.figure()
plt.plot(k_list, mean_scores)
plt.xlabel('K')
plt.ylabel('Mean Accuracy')
plt.title('Cross Validated Accuracies')
plt.xlim(0, 20)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1, step=0.1))
plt.xticks(np.arange(1, 21, step=1))
plt.axvline(x=best_k, ls='--', linewidth=3.0, color='grey')
plt.axhline(y=best_k_score, ls='--', linewidth=3.0, color='grey')
plt.savefig("workshop_2/figures/cv_plot.png")


neigh = KNeighborsClassifier(n_neighbors=best_k)
neigh.fit(X, y)
score = neigh.score(X, y)
score

true_vers_test = np.logical_and((neigh.predict(X) == y), [x == "Versicolor" for x in y])
false_vers_test = np.logical_and((neigh.predict(X) != y), [x == "Versicolor" for x in y])
true_virg_test = np.logical_and((neigh.predict(X) == y), [x == "Virginica" for x in y])
false_virg_test = np.logical_and((neigh.predict(X) != y), [x == "Virginica" for x in y])


Z = neigh.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
y_pred = neigh.predict(X)
plt.figure()
ax = plt.subplot(1, 1, 1)
ax.set_title("K=" + str(best_k) + ", " + "Accuracy = " + str(score))
# Plot the training points
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
ax.scatter(X["sepal.length"], X["sepal.width"],
           c=y.astype("category").cat.codes, cmap=cm_bright, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
plt.savefig("workshop_2/figures/irisk-best-k.png")

confusion_matrix(y, y_pred, labels=["Virginica", "Versicolor"])


fpr, tpr, thresholds = roc_curve(y.astype("category").cat.codes,
                                 neigh.predict_proba(X)[:,0])
roc_auc = auc(tpr, fpr)

lw = 2
plt.figure()
plt.plot(tpr, fpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("workshop_2/figures/roc_curve.png")
