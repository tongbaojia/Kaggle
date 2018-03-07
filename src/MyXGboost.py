import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np

# Test performance on test set
def evaluate(alg, data, dumb=None):
    '''data is a dictionary.
    	alg is the model.
    	return the accuracy'''
    X_te_temp = data["X_test"]
    y_te_temp = data["y_test"]
     
    if dumb == None: 
        te_predictions = alg.predict(X_te_temp)
        te_predprob    = alg.predict_proba(X_te_temp)[:,1]
    else: ## this is just using all 0 or 1s
        X_te_temp["pred"] = dumb
        X_dumb = X_te_temp["pred"] 
        te_predictions = X_dumb
        te_predprob = X_dumb
    #print(y_te_temp.values, te_predictions)
    test_acc = metrics.accuracy_score(y_te_temp.values, te_predictions)
#     #Print model report:
#     #print(CSI + "32;40m" + str(metrics.accuracy_score(y_vr_temp.values, vr_predictions)) + CSI + "0m")
#     print("Test Accuracy :  %.3g" % test_acc)
#     try:
#         print("Test AUC Score (Test): %.3f" % metrics.roc_auc_score(y_te_temp, te_predprob))
#     except ValueError:
#         print("Opps, 1")
    #Validation report
    return test_acc

def training(data, name="xgb_temp", optimize=False, rankplot=False):
    '''data is a dictionary.
    if optimize, will do the parameter search'''
    param_grid = {
            'learning_rate': np.arange(1e-2, 1e-1, 1e-2), #np.arange(0.001, 0.3, 0.005), ##0.07
            'max_depth': np.arange(3, 7, 1), ##prefer smaller values, 4
            'gamma' : np.arange(0.001, 0.01, 0.001), ##insensitive
            'n_estimators' : np.arange(30, 45, 3), ##best around 75
            'colsample_bytree': [.6, .7, .8, .9, 1], ##best 1
            'reg_alpha' : np.arange(0.3, 1.0, 0.1), ##insensitive
            'reg_lambda' : np.arange(0.1, 1.0, 0.1), ##insensitive
            'subsample' : [.4, .5, .6, .7, ], ##best 1
            'min_child_weight' : [.4, .5, .6],
            #'base_score' : [0.8], ##initial score
    }
    
    if optimize:
        xgb_temp = XGBClassifier(objective= 'binary:logistic', eval_metric='auc')
        xgb_temp = modelfit(xgb_temp, data, param_grid=param_grid)
        #joblib.dump(param_grid.best_estimator_, name + ".parlib.dat")
    else:
        xgb_temp = XGBClassifier(
            #max_depth = 4, #Maximum tree depth for base learners.
            #learning_rate = 1e-2, #Boosting learning rate (XGBoost's "eta")
            n_estimators= 25, ## Number of boosted trees, CV doesn't change after ~ 90
            ##silent = False, ##print messages or not
            objective= 'binary:logistic',
            eval_metric='auc',
            #nthread=-1,
            #gamma=0.01, #Minimum loss reduction required to make a further partition on a leaf node of the tree.
            #min_child_weight=1, # Minimum sum of instance weight(hessian) needed in a child.
            #scale_pos_weight=1,
            reg_alpha=0.6,
            #reg_lambda=0.6,
            #subsample=0.7, #Subsample ratio of the training instance.
            #colsample_bytree = 0.7, #Subsample ratio of columns when constructing each tree.
            seed=24)
        xgb_temp = modelfit(xgb_temp, data, search=False)

        #neigh = KNeighborsClassifier(n_neighbors=3)
        #neigh = RandomForestClassifier(n_estimators=200)
        #neigh = SVC(probability=True)
        #xgb_temp = modelfit(neigh, data, useTrainCV=False)
    
    ##save model
    joblib.dump(xgb_temp, name + ".joblib.dat")
    ##load model
    #loaded_model = joblib.load("pima.joblib.dat")
    
    ##finish and plot
    if rankplot:
	    plt.clf()
	    fig = plt.figure(figsize=(10,6))
	    feat_imp = pd.Series(xgb_temp._Booster.get_fscore()).sort_values(ascending=False)
	    feat_imp.plot(kind='bar', title='Feature Importances')
	    plt.ylabel('Feature Importance Score')
	    plt.show()
	    plt.savefig('Plot/feature_ranking_' + name + '.pdf')
    
    return xgb_temp

## A new more systematic approach
def modelfit(alg, data, search=True, param_grid={}):
#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
#https://www.dataiku.com/learn/guide/code/python/advanced-xgboost-tuning.html
    
    X_tr_temp = data["X_train"]
    y_tr_temp = data["y_train"]
    X_vr_temp = data["X_val"]
    y_vr_temp = data["y_val"]
    
    if search:

        scoring = {'AUC': 'roc_auc', 'Accuracy': metrics.make_scorer(metrics.accuracy_score)}
        
        ##Randomized Search CV
        gsearch = RandomizedSearchCV(estimator=alg, param_distributions=param_grid, n_iter=50, cv=3, n_jobs=2)
        
        #Grid search CV; much slower
        # gsearch = GridSearchCV(estimator = alg, param_grid = param_grid, 
        # 	n_jobs=3, iid=False, cv=5, scoring=scoring, return_train_score=True, refit="AUC")
        
        ##Search
        gsearch.fit(X_tr_temp, y_tr_temp)
        
#         ##check trend
#         results = gsearch.cv_results_
#         plt.clf()
#         plt.figure(figsize=(13, 13))
#         plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)
#         plt.xlabel("Parameter")
#         plt.ylabel("Score")
#         plt.grid()

#         ax = plt.axes()
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0.75, 1)

#         # Get the regular numpy array from the MaskedArray
#         X_axis = np.array(results['param_' + "min_child_weight"].data, dtype=float)

#         for scorer, color in zip(sorted(scoring), ['g', 'k']):
#             for sample, style in (('train', '--'), ('test', '-')):
#                 sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
#                 sample_score_std = results['std_%s_%s' % (sample, scorer)]
#                 ax.fill_between(X_axis, sample_score_mean - sample_score_std,
#                                 sample_score_mean + sample_score_std,
#                                 alpha=0.1 if sample == 'test' else 0, color=color)
#                 ax.plot(X_axis, sample_score_mean, style, color=color,
#                         alpha=1 if sample == 'test' else 0.7,
#                         label="%s (%s)" % (scorer, sample))

#             best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
#             best_score = results['mean_test_%s' % scorer][best_index]

#             # Plot a dotted vertical line at the best score for that scorer marked by x
#             ax.plot([X_axis[best_index], ] * 2, [0, best_score],
#                     linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

#             # Annotate the best score for that scorer
#             ax.annotate("%0.2f" % best_score,
#                         (X_axis[best_index], best_score + 0.005))

#         plt.legend(loc="best")
#         plt.grid('off')
#         plt.show()

        ## from the documentation
        best_parameters = gsearch.best_params_
        score = gsearch.best_score_
        print("par", best_parameters)
        print("score %.4f" % score)
        alg = gsearch.best_estimator_
    
    ## fit alg
    #alg.fit(X_tr_temp, y_tr_temp)
    #XGboost specific
    alg.fit(X_tr_temp, y_tr_temp, 
    	eval_set=[(X_tr_temp, y_tr_temp), (X_vr_temp, y_vr_temp)], eval_metric="error", verbose=False)
    
    #Predict training set:
    tr_predictions = alg.predict(X_tr_temp)
    tr_predprob    = alg.predict_proba(X_tr_temp)[:,1]
    vr_predictions = alg.predict(X_vr_temp)
    vr_predprob    = alg.predict_proba(X_vr_temp)[:,1]
        
#     #Print model report:
#     print("Train Accuracy : %.3g" % metrics.accuracy_score(y_tr_temp.values, tr_predictions))
#     print("Train AUC Score (Train): %.3f" % metrics.roc_auc_score(y_tr_temp, tr_predprob))
#     #Validation report
#     print("Val Accuracy : %.3g" % metrics.accuracy_score(y_vr_temp.values, vr_predictions))
#     try:
#         print("Val AUC Score (Val): %.3f" % metrics.roc_auc_score(y_vr_temp, vr_predprob))
#     except ValueError:
#         print("Opps, 1")
    
    return alg