import pandas as pd
import geopandas as gpd
import numpy as np
from vars import label, spatial_extent, date, data_l_threshold_lower, data_l_threshold_upper, data_m_threshold_upper, classification_method, dy
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt


def classifyRF(data, training_data, variables):
    """
    Classifies input data by supervised Random Forest.

    Params:
    - data
        input data to be classified (DataFrame)
    - training data
        training dataset used for classification (including all computed features) (DataFrame)

    Returns:
    Previous DataFrame "data" with an additional column "surging" predicted using Random Forest.
    """

    # remove nans from datasets
    data_all = data
    data = data.dropna(axis='index')
    training_data = training_data.dropna(axis='index')

    # Split the data into features (X) and target (y)
    X = data.drop(columns=['id', 'glacier_id', 'year', 'quality_flag'])
    #y = data.surging_rf.replace({True: 1, False: 0})

    # split training dataset into training and validation dat
    X_training_data = training_data.drop(columns=['id', 'surging'])
    y_training_data = training_data.surging

    acc = []
    prec = []
    rec = []
    f1 = []
    f1_train = []

    # test n_estimators
    if False:
        r = np.arange(2, 250, 1)
        for n in [x for x in r]:
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            # evaluate the model 20x
            for i in range(100):
                X_train, X_test, y_train, y_test = train_test_split(X_training_data, y_training_data, test_size=1/3, random_state=i)
                rf = RandomForestClassifier(n_estimators=n)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1score = f1_score(y_test, y_pred)
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1score)

            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
            avg_f1 = sum(f1_scores) / len(f1_scores)

            acc.append(avg_accuracy)
            prec.append(avg_precision)
            rec.append(avg_recall)
            f1.append(avg_f1)

        plt.plot([x for x in r], acc, label='accuracy')
        plt.plot([x for x in r], prec, label='precision')
        plt.plot([x for x in r], rec, label='recall')
        plt.plot([x for x in r], f1, label='f1')
        plt.title('RF n_estimators')
        plt.legend()
        plt.xlabel('parameter value')
        plt.ylabel('score')
        plt.savefig('figs/RF_n_estimators.png')
        plt.close()

    # max_depths
    if False:
        for n in [x for x in np.arange(1, 25, 1)]:
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []

            # evaluate the model 20x
            for i in range(100):
                X_train, X_test, y_train, y_test = train_test_split(X_training_data, y_training_data, test_size=1/3, random_state=i)
                rf = RandomForestClassifier(max_depth=n)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1score = f1_score(y_test, y_pred)

                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1score)

            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
            avg_f1 = sum(f1_scores) / len(f1_scores)

            acc.append(avg_accuracy)
            prec.append(avg_precision)
            rec.append(avg_recall)
            f1.append(avg_f1)

        plt.plot([x for x in np.arange(1, 25, 1)], acc, label='accuracy')
        plt.plot([x for x in np.arange(1, 25, 1)], prec, label='precision')
        plt.plot([x for x in np.arange(1, 25, 1)], rec, label='recall')
        plt.plot([x for x in np.arange(1, 25, 1)], f1, label='f1')
        plt.title('RF max_depth')
        plt.xlabel('parameter value')
        plt.ylabel('score')
        plt.legend()
        plt.savefig('figs/RF_max_depth.png')
        plt.close()

        #print(confusion_matrix(y_test, y_pred))
        #print(classification_report(y_test, y_pred))

    # min_samples_split
    if False:
        for n in [x for x in np.arange(2, 75, 1)]:
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            # evaluate the model 20x
            for i in range(100):
                X_train, X_test, y_train, y_test = train_test_split(X_training_data, y_training_data, test_size=1/3, random_state=i)
                rf = RandomForestClassifier(min_samples_split=n)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1score = f1_score(y_test, y_pred)

                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1score)

            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
            avg_f1 = sum(f1_scores) / len(f1_scores)

            acc.append(avg_accuracy)
            prec.append(avg_precision)
            rec.append(avg_recall)
            f1.append(avg_f1)

        plt.plot([x for x in np.arange(2, 75, 1)], acc, label='accuracy')
        plt.plot([x for x in np.arange(2, 75, 1)], prec, label='precision')
        plt.plot([x for x in np.arange(2, 75, 1)], rec, label='recall')
        plt.plot([x for x in np.arange(2, 75, 1)], f1, label='f1')
        plt.title('RF min_samples_split')
        plt.legend()
        plt.xlabel('parameter value')
        plt.ylabel('score')
        plt.savefig('figs/RF_min_samples_split.png')
        plt.close()

    # min_leaf_nodes
    if False:
        for n in [x for x in np.arange(1, 25, 1)]:

            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []

            # evaluate the model 20x
            for i in range(100):
                X_train, X_test, y_train, y_test = train_test_split(X_training_data, y_training_data, test_size=1/3, random_state=i)
                rf = RandomForestClassifier(min_samples_leaf=n)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1score = f1_score(y_test, y_pred)

                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1score)

            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
            avg_f1 = sum(f1_scores) / len(f1_scores)

            acc.append(avg_accuracy)
            prec.append(avg_precision)
            rec.append(avg_recall)
            f1.append(avg_f1)

        plt.plot([x for x in np.arange(1, 25, 1)], acc, label='accuracy')
        plt.plot([x for x in np.arange(1, 25, 1)], prec, label='precision')
        plt.plot([x for x in np.arange(1, 25, 1)], rec, label='recall')
        plt.plot([x for x in np.arange(1, 25, 1)], f1, label='f1')

        plt.title('RF min_samples_leaf')
        plt.legend()
        plt.xlabel('parameter value')
        plt.ylabel('score')

        plt.savefig('figs/RF_min_samples_leaf.png')
        plt.close()

    # max_features
    if False:
        r = np.arange(2, 98, 1)
        for n in [x for x in r]:

            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            f1_scores_train = []

            # evaluate the model 100x
            for i in range(100):
                X_train, X_test, y_train, y_test = train_test_split(X_training_data, y_training_data, test_size=1/3, random_state=i)
                rf = RandomForestClassifier(max_features=n)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                y_pred_t = rf.predict(X_train)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1score = f1_score(y_test, y_pred)
                f1score_train = f1_score(y_train, y_pred_t)

                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1score)
                f1_scores_train.append(f1score_train)

            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_f1_t = sum(f1_scores_train) / len(f1_scores_train)

            acc.append(avg_accuracy)
            prec.append(avg_precision)
            rec.append(avg_recall)
            f1.append(avg_f1)
            f1_train.append(avg_f1_t)

        plt.plot([x for x in r], acc, label='accuracy')
        plt.plot([x for x in r], prec, label='precision')
        plt.plot([x for x in r], rec, label='recall')
        plt.plot([x for x in r], f1, label='f1')
        #plt.plot([x for x in r], f1_train, label='f1 (train)')

        plt.title('RF max_features')
        plt.legend()
        plt.xlabel('parameter value')
        plt.ylabel('score')

        plt.savefig('figs/RF_max_features.png')
        plt.close()

    # plot accuracies
    #plt.scatter([x for x in range(len(accuracies))], accuracies)
    #plt.title('accuracies for Random Forest')
    #plt.savefig('figs/accuracies.png')
    #plt.close()

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X_training_data, y_training_data, test_size=1 / 3,
                                                            random_state=i)
        rf = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_depth=10)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1score)

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print(f'accuracy: {avg_accuracy}')
    print(f'precision: {avg_precision}')
    print(f'recall: {avg_recall}')
    print(f'f1: {avg_f1}')

    # Get and reshape confusion matrix d/ata
    matrix = confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    # Build the plot
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                cmap=plt.cm.Blues, linewidths=0.2)
    # Add labels to the plot
    class_names = ['not_surging', 'surging']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for Random Forest Model \n accuracy: ({accuracy})')
    plt.savefig(f'figs/confusionmatrix_RF_{label}_{date}.png')
    plt.close()

    # create classification based on whole training dataset
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=20, max_depth=80)
    rf.fit(X_training_data, y_training_data)

    # plot tree
    from sklearn.tree import export_graphviz
    #feature_names = variables[4:-2]
    #target_names = ['not surging', 'surging']
    #fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    #tree.plot_tree(rf.estimators_[0], feature_names=feature_names, class_names=target_names,
    #               filled=True)
    #fig.savefig('figs/rf_individualtree.png')

    # create dataframe for glaciers that were in the classification
    X['surging'] = rf.predict(X)
    X['probability'] = [x[0] if (x[0] > x[1]) else x[1] for x in rf.predict_proba(X.drop(columns=['surging']))]
    result = pd.concat([data, X], axis=1, join='inner')

    # create dataframe for empty glaciers (none, were not in classification)
    # Merge the dataframes
    merged_df = pd.merge(data_all, data, on=['glacier_id', 'year'], how='left', indicator=True)

    # Filter the rows where the indicator is 'left_only' (i.e., only in all_glaciers)
    empty_glaciers = merged_df[merged_df['_merge'] == 'left_only']

    # Select only the necessary columns
    empty_glaciers = empty_glaciers[['year', 'glacier_id']]
    empty_glaciers['id'] = empty_glaciers['glacier_id'] + '_' + empty_glaciers['year'].astype(str)
    empty_glaciers['surging'] = -999

    # only return surge/not surge
    subset = result[['id', 'surging', 'probability', 'glacier_id', 'year']]

    # merge with empty glaciers
    subset = pd.concat([subset, empty_glaciers])

    return subset


def RF():
    """
    Create training dataset, so far for the southern part.

    :param data: dataframe with extracted features, glacier_id, geometry

    :return:
    """
    # read input data
    if dy:
        data = gpd.read_file(f'data/temp/{label}_features_dy.gpkg', engine='pyogrio', use_arrow=True)
    else:
        data = gpd.read_file(f'data/temp/{label}_features.gpkg', engine='pyogrio', use_arrow=True)

    data['id'] = data['glacier_id'] + '_' + data['year'].astype(str)

    # read training data
    training_data = pd.read_csv('data/data/trainingdata4.csv', engine='pyarrow')

    rf_variables = ['glacier_id',
                    'year',
                    'quality_flag',
                    'id',
                    #'dh_max_l',
                    #'dh_mean_l',
                    #'dh_std',
                    #'dh_std_l',
                    #'hdh_std',
                    #'hdh_std_l',
                    'lin_coef_binned',
                    'lin_coef_l_binned',
                    #'variance_dh',
                    #'variance_dh_l',
                    #'correlation',
                    #'correlation_l',
                    #'ransac',
                    'ransac_l',
                    #'over15',
                    'bin1',
                    'bin2',
                    'bin3',
                    'bin4',
                    'bin5',
                    'bin6',
                    'bin7',
                    'bin8',
                    'bin9',
                    'bin10',
                    #'bin11',
                    #'bin12',
                    #'bin13',
                    #'bin14',
                    #'bin15',
                    #'bin16',
                    #'bin17',
                    #'bin18',
                    'residuals',
                    #'bin_max',
                    'surging']

    # select the data subset
    training_data = pd.merge(data, training_data, on=['glacier_id', 'year'])
    print(len(training_data))

    # filter out the bad quality flags
    training_data = training_data[training_data['quality_flag'] > 1]
    print(len(training_data))
    #training_data.to_file('data/temp/training_data_features_dy.gpkg')

    # add unique ID (combo of glacier ID and year)
    training_data['id'] = training_data['glacier_id'] + '_' + training_data['year'].astype(str)

    # prepare training data and input data for Random Forest
    training_data = training_data[rf_variables[3:]]
    input_data = data[rf_variables[:-1]]

    # classify data using Random Forest
    result = classifyRF(input_data, training_data, rf_variables)

    # merge results of RF with original data
    result = pd.merge(data, result, on=['glacier_id', 'year', 'id'])
    result = result[['glacier_id', 'glac_name', 'year', 'surging', 'probability', 'geometry', 'quality_flag']]
    result = result.rename(columns={'glac_name': 'glacier_name'})

    # convert df to gdf
    result = gpd.GeoDataFrame(result)
    result.to_file(f'data/results/RFresults_{label}_{date}.gpkg')
    # save data
    r_4326 = result.to_crs(4326)
    r_4326.to_file(f'data/results/RFresults_{label}_{date}.geojson')

    # split data by year and save as individual files
    for year in [2019, 2020, 2021, 2022, 2023]:
        subset = result[result['year'] == year]
        subset.to_file(f'data/results/RFresults_{label}_{year}_{date}.gpkg')
        s_4326 = subset.to_crs(4326)
        s_4326.to_file(f'data/results/RFresults_{year}.geojson')
        print(year)
        print(len(subset[subset['surging'] == 1]))

    return result


def classifyThresholdMethod(data, th_dh_max_l, th_dh_max_l_2, th_lin_coef_l, th_lin_coef):

    # classify based on thresholds
    surging = []
    l = len(data)
    for i in range(l):
        v = 0
        if data['lin_coef_l'][i] < th_lin_coef_l:
            v = v + 1
        if data['lin_coef'][i] < th_lin_coef:
            v = v + 1
        if data['dh_max_l'][i] > th_dh_max_l:
            v = v + 1
        if data['dh_max_l'][i] > th_dh_max_l_2:
            v = v + 1

        surging.append(v)
        v = 0


    return surging


def TH():
    """
    Classify glaciers to surging/not surging based on specified thresholds.
    :return: 
    """

    # set thresholds
    if dy:
        th_dh_max_l = 15
        th_dh_max_l_2 = 35
        th_lin_coef = 0
        th_lin_coef_l = 0

    else:
        th_dh_max_l = 15
        th_dh_max_l_2 = 35
        th_lin_coef = 0
        th_lin_coef_l = 0

    # read input data
    if dy:
        data = gpd.read_file(f'data/temp/{label}_features_dy.gpkg', engine='pyogrio', use_arrow=True)
    else:
        data = gpd.read_file(f'data/temp/{label}_features.gpkg', engine='pyogrio', use_arrow=True)

    # read training data
    training_data = pd.read_csv(f'data/data/trainingdata3.csv', engine='pyarrow')
    training_data = pd.merge(data, training_data, on=['glacier_id', 'year'])

    # classify
    surging = classifyThresholdMethod(data, th_dh_max_l, th_dh_max_l_2, th_lin_coef_l, th_lin_coef)

    # append results of threshold classification
    data['surging'] = surging

    # validation
    td_surging = classifyThresholdMethod(training_data, th_dh_max_l, th_dh_max_l_2, th_lin_coef_l, th_lin_coef)
    training_data['surging_th'] = td_surging

    td_result = training_data[['glacier_id', 'year', 'surging', 'surging_th']]
    surging_correct = len(td_result.where((td_result['surging'] == 1) & (td_result['surging_th'] >0)).dropna())
    notsurging_correct = len(td_result.where((td_result['surging'] == 0) & (td_result['surging_th'] == 0)).dropna())
    surging_notsurging = len(td_result.where((td_result['surging'] == 1) & (td_result['surging_th'] == 0)).dropna())
    notsurging_surging = len(td_result.where((td_result['surging'] == 0) & (td_result['surging_th'] > 0)).dropna())


    print(f'surging training data: {len(training_data[training_data["surging"] == 1])}')
    print(f'not surging training data: {len(training_data[training_data["surging"] == 0])}')

    print(surging_correct)
    print(notsurging_correct)
    print(surging_notsurging)
    print(notsurging_surging)

    # merge results of RF with original data
    result = data[['glacier_id', 'glac_name', 'year', 'surging', 'geometry', 'quality_flag']]

    # convert df to gdf
    result = gpd.GeoDataFrame(result)

    # save data
    result.to_file(f'data/results/THresults_{label}_{date}.gpkg')

    # split data by year and save as individual files
    for year in [2018, 2019, 2020, 2021, 2022, 2023]:
        subset = result[result['year'] == year]
        subset.to_file(f'data/results/THresults_{label}_{year}_{date}.gpkg')

        # save to github website project as geojson in wgs84
        s4326 = subset.to_crs(4326)
        s4326.to_file(f'eliskasieglova.github.io/data/result{year}.geojson')

    return

def classifyHGB(data, training_data):
    """
    Classifies input data by supervised Random Forest.

    Params:
    - data
        input data to be classified (DataFrame)
    - training data
        training dataset used for classification (including all computed features) (DataFrame)

    Returns:
    Previous DataFrame "data" with an additional column "surging" predicted using Random Forest.
    """

    from sklearn import ensemble

    # remove nans from datasets
    data_all = data
    data = data.dropna(axis='index')
    training_data = training_data.dropna(axis='index')

    # Split the data into features (X) and target (y)
    X = data.drop(columns=['id', 'glacier_id', 'year', 'quality_flag'])
    #y = data.surging_rf.replace({True: 1, False: 0})

    # split training dataset into training and validation dat
    X_train = training_data.drop(columns=['id', 'surging'])
    y_train = training_data.surging

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=5)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.datasets import load_iris

    # fitting and evaluating the model
    clf = HistGradientBoostingClassifier().fit(X_train, y_train)
    clf.score(X_train, y_train)

    # evaluate the model by comparison with actual data
    y_pred = clf.predict(X)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print(accuracy)

    import seaborn as sns
    import matplotlib.pyplot as plt
    # Get and reshape confusion matrix d/ata
    matrix = confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    # Build the plot
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                cmap=plt.cm.Blues, linewidths=0.2)
    # Add labels to the plot
    class_names = ['surging', 'not surging']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Histogram Gradient Boosting')
    plt.savefig(f'figs/confusionmatrix_HGB_{label}_{date}.png')
    plt.close()

    # create dataframe for glaciers that were in the classification
    X['surging'] = clf.predict(X)
    X['probability'] = [x[0] if (x[0] > x[1]) else x[1] for x in clf.predict_proba(X.drop(columns=['surging']))]
    result = pd.concat([data, X], axis=1, join='inner')

    # create dataframe for empty glaciers (none, were not in classification)
    # Merge the dataframes
    merged_df = pd.merge(data_all, data, on=['glacier_id', 'year'], how='left', indicator=True)

    # Filter the rows where the indicator is 'left_only' (i.e., only in all_glaciers)
    empty_glaciers = merged_df[merged_df['_merge'] == 'left_only']

    # Select only the necessary columns
    empty_glaciers = empty_glaciers[['year', 'glacier_id']]
    empty_glaciers['id'] = empty_glaciers['glacier_id'] + '_' + empty_glaciers['year'].astype(str)
    empty_glaciers['surging'] = -999

    # only return surge/not surge
    subset = result[['id', 'surging', 'probability', 'glacier_id', 'year']]

    # merge with empty glaciers
    subset = pd.concat([subset, empty_glaciers])

    return subset


def HGB():

    """
    Create training dataset, so far for the southern part.

    :param data: dataframe with extracted features, glacier_id, geometry

    :return:
    """
    # read input data
    if dy:
        data = gpd.read_file(f'data/temp/{label}_features_dy.gpkg', engine='pyogrio', use_arrow=True)
    else:
        data = gpd.read_file(f'data/temp/{label}_features.gpkg', engine='pyogrio', use_arrow=True)

    data['id'] = data['glacier_id'] + '_' + data['year'].astype(str)

    # read training data
    training_data = pd.read_csv('data/data/trainingdata2.csv', engine='pyarrow')

    rf_variables = ['glacier_id',
                    'year',
                    'quality_flag',
                    'id',
                    'dh_max_l',
                    'dh_max_m',
                    'dh_mean_l',
                    'dh_mean_m',
                    'dh_std',
                    'dh_std_l',
                    'dh_std_m',
                    'hdh_std',
                    'hdh_std_l',
                    'hdh_std_m',
                    'lin_coef',
                    'lin_coef_l',
                    'lin_coef_binned',
                    'lin_coef_l_binned',
                    'lin_coef_m',
                    'variance_dh',
                    'variance_dh_l',
                    'variance_dh_m',
                    'correlation',
                    'correlation_m',
                    'correlation_l',
                    'ransac',
                    'ransac_l',
                    'ransac_m',
                    'over15',
                    'over15_m',
                    'bin1',
                    'bin2',
                    'bin3',
                    'bin4',
                    'bin5',
                    'bin6',
                    'bin7',
                    'bin8',
                    'bin9',
                    'bin10',
                    'surging']

    # select the data subset
    training_data = pd.merge(data, training_data, on=['glacier_id', 'year'])
    print(len(training_data))

    # filter out the bad quality flags
    training_data = training_data[training_data['quality_flag'] > 1]
    print(len(training_data))
    #training_data.to_file('data/temp/training_data_features_dy.gpkg')

    # add unique ID (combo of glacier ID and year)
    training_data['id'] = training_data['glacier_id'] + '_' + training_data['year'].astype(str)

    # prepare training data and input data for Random Forest
    training_data = training_data[rf_variables[3:]]
    input_data = data[rf_variables[:-1]]

    # classify data using Random Forest
    result = classifyHGB(input_data, training_data)

    # merge results of RF with original data
    result = pd.merge(data, result, on=['glacier_id', 'year', 'id'])
    result = result[['glacier_id', 'glac_name', 'year', 'surging', 'probability', 'geometry', 'quality_flag']]
    result = result.rename(columns={'glac_name': 'glacier_name'})

    # convert df to gdf
    result = gpd.GeoDataFrame(result)
    result.to_file(f'data/results/HGBresults_{label}_{date}.gpkg')
    # save data
    r_4326 = result.to_crs(4326)
    r_4326.to_file(f'data/results/HGBresults_{label}_{date}.geojson')

    # split data by year and save as individual files
    for year in [2018, 2019, 2020, 2021, 2022, 2023]:
        subset = result[result['year'] == year]
        subset.to_file(f'data/results/HGBresults_{label}_{year}_{date}.gpkg')
        s_4326 = subset.to_crs(4326)
        s_4326.to_file(f'data/results/HGBresults_{year}.geojson')
        print(year)
        print(len(subset[subset['surging'] == 1]))

    return result


def classify(classification_method):

    if classification_method == 'RF':
        return RF()

    if classification_method == 'TH':
        return TH()

    if classification_method == 'HGB':
        return HGB()

