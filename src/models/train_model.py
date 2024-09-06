import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle('../../data/interim/03_data_features.pkl')

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(['participant','category','set'],axis=1)

X = df_train.drop('label',axis=1)
y = df_train['label']

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# Ilustrating how the splits are done. 

fig,ax = plt.subplots(figsize=(10,5))
df_train['label'].value_counts().plot(
    kind='bar', ax=ax, color='lightblue', label='total')
y_train.value_counts().plot(kind='bar',ax=ax,color='dodgerblue',label='train')
y_test.value_counts().plot(kind='bar',ax=ax,color='royalblue',label='test')
plt.legend()
plt.show()

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

basic_features = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z']
square_features = ['acc_r','gyr_r']
pca_features = ['pca_1','pca_2','pca_3']
time_features = [f for f in df_train.columns if "_temp_" in f]
frequency_features = [f for f in df_train.columns if ("_freq" in f) or ('_pse' in f)]
cluster_features = ['cluster']

print('Basic features', len(basic_features))
print('square features', len(square_features))
print('pca_features', len(pca_features))
print('Time Features', len(time_features))
print('frequency features', len(frequency_features))
print('cluster features', len(cluster_features)) 

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))

learner = ClassificationAlgorithms()
max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features,X_train,y_train
)

selected_features = [
    'acc_z_freq_0.0_Hz_ws_14',
    'acc_x_freq_0.0_Hz_ws_14',
    'gyr_r_freq_0.0_Hz_ws_14',
    'acc_y_temp_mean_ws_5',
    'acc_r_temp_std_ws_5',
    'gyr_x',
    'gyr_z_freq_1.071_Hz_ws_14',
    'gyr_y_freq_1.429_Hz_ws_14',
    'gyr_z_max_freq',
    'gyr_r']

ordered_scores = [
    0.885556704584626,
    0.9886246122026887,
    0.9955187866253016,
    0.9975870389520854,
    0.9989658738366081,
    0.9996552912788693,
    0.9996552912788693,
    0.9996552912788693,
    0.9996552912788693,
    0.9996552912788693]

plt.figure(figsize=(10,5))
plt.plot(np.arange(1,max_features + 1,1),ordered_scores)
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1,max_features+1,1))
plt.show() 

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features
]

feature_names = [
    'Feature Set 1',
    'Feature Set 2',
    'Feature Set 3',
    'Feature Set 4',
    'Selected Features'
]

iterations = 1 
score_df = pd.DataFrame()


for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

score_df.sort_values(by='accuracy',ascending=False)

plt.figure(figsize=(10,10))
sns.barplot(data=score_df,x='model',y='accuracy', hue='feature_set')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.7,1)
plt.legend(loc='lower right')
plt.show()

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

( 
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test,class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test,class_test_y,labels=classes)

import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test,class_test_y,labels=classes)
# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

participant_df = df.drop(['set','category'],axis=1)
X_train = participant_df[participant_df['participant']!='A'].drop('label',axis=1)
y_train = participant_df[participant_df['participant']!='A']['label']

X_test = participant_df[participant_df['participant']=='A'].drop('label',axis=1)
y_test = participant_df[participant_df['participant']=='A']['label']

X_train = X_train.drop('participant',axis=1)
X_test = X_test.drop('participant',axis=1)

# Visualizing splits 

fig,ax = plt.subplots(figsize=(10,5))
df_train['label'].value_counts().plot(
    kind='bar', ax=ax, color='lightblue', label='total')
y_train.value_counts().plot(kind='bar',ax=ax,color='dodgerblue',label='train')
y_test.value_counts().plot(kind='bar',ax=ax,color='royalblue',label='test')
plt.legend()
plt.show()


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

( 
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test,class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test,class_test_y,labels=classes)

import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test,class_test_y,labels=classes)

# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------

( 
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=False
)

accuracy = accuracy_score(y_test,class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test,class_test_y,labels=classes)

import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test,class_test_y,labels=classes)