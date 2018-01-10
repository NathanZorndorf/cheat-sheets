#-------------------------------------------
# WITH SCIKIT LEARN
#-------------------------------------------

# API REFERENCE: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

#--- TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X = df[['predictor_cols']] # should be the predictors
y = df['response_col'] # should be the response

	# Example
	from sklearn.model_selection import train_test_split
	X = iris.as_matrix(columns=iris.columns.tolist()[:4])
	y = iris.as_matrix(columns=iris.columns.tolist()[:5])
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=, random_state=) 


	#Example:
	from sklearn.model_selection import train_test_split
	X = df.loc[:-1].values()
	y = np.ravel(df.loc[-1].values())
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=, random_state=) 





#-----------------------------
#--- MODEL EVALUATION / METRICS
#-----------------------------
#--- Linear Model
from sklearn import metrics
print(lm.coef_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#--- Logistic Model
from sklearn import metrics

#---- Classification Report
from sklearn.metrics import classification_report
classification_report()

	# Example 
	from sklearn.metrics import classification_report
	predictions = lr.predict(X_test)
	print classification_report(y_test, predictions, digits=4)

#---- Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)


	# Example : plotting confusion matrix
	Y_hat = lr.predict(X_holdout)
	cm = confusion_matrix(y_true=Y_holdout, y_pred=Y_hat)
	TN = cm[0][0]
	FN = cm[1][0]
	TP = cm[1][1]
	FP = cm[0][1]
	print 'Precision (TP, FP): ', TP, FP, 'Prec = ', (TP)/(TP + FP)
	print 'recall (TP, FN): ', TP, FN, 'Recall = ', (TP)/(TP + FN)

	# Example 
	See tips / workflows / 

#----- Accuracy
from sklearn.metrics import accuracy_score

#----- Recall
from sklearn.metrics import recall_score

#---- Precision
from sklearn.metrics import precision_score




#----------------- PRE-PROCESSING -----------------------#

#---- Label Encoder 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
	
	# Example
	le = LabelEncoder()
	le.fit(iris['Name'])
	iris['Name_class'] = le.transform(iris['Name'])


#---- Standardize / Normalize features by removing the mean and scaling to unit variance
StandardScaler(copy=True, with_mean=True, with_std=True)
	.fit(X[, y])	# Compute the mean and std to be used for later scaling.
	.fit_transform(X[, y])	# Fit to data, then transform it.
	.get_params([deep])	# Get parameters for this estimator.

	# Example
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	predictor_matrix = sc.fit_transform(X=df.loc[:,factor_cols], y=df[target_col])

	# Example:
	# before scaling, exclude the categorical features
	df_cont.astype(float)
	# Create standard scale object
	sc = StandardScaler()
	# could combine the next two statements into 1 with .fit_transform()
	sc.fit(X=df_cont)
	X_norm = sc.transform(df_cont)
	# put normalized values back into dataframe 
	df_cont.iloc[:,:] = X_norm


#--------------- DECOMPOSITION 

#---- Principal Component Analysis 
class sklearn.decomposition.PCA
PCA(n_components=None)

	# Eample:
	from sklearn.decomposition import PCA
	
	pca = PCA(n_components=2)

	pca_matrix = pca.fit_transform(design_matrix)

	print pca.explained_variance_ratio_

	# plot review_pca
	colors = train['sentiment'].values[:]/1.01
	plt.scatter(x=review_pca[:, 0], y=review_pca[:, 1], c=colors, cmap='spring')
	plt.title('PCA with 2 components')
	plt.xlabel('1st component')
	plt.ylabel('2nd component')



#---- Singular Value Decomposition SVD 
from sklearn.decomposition import TruncatedSVD
TruncatedSVD(n_components=, algorithm=)

	# Example :
	tsvd = TruncatedSVD(n_components=2, algorithm='arpack')
	svd_mat = tsvd.fit_transform(design_matrix) # This transformer performs linear dimensionality reduction by means of truncated SVD 
	mat_norm = Normalizer(copy=False).fit_transform(svd_mat)
	

#-------------------- VALIDATION -------------------------#



#---- K-Fold Cross validation 	
from sklearn.model_selection import cross_val_score
cross_val_score(estimator=, X=, y=, cv=3, scoring=)
	# Scoring options = http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values

	# Example 
	scores = cross_val_score(estimator=LogisticRegression(), X=df_new.loc[:,nc], y=df_new.loc[:,'SHOTS_MADE'], cv=10, scoring='neg_mean_squared_error')
	print np.mean(scores)


#---- Stratified K-Folds cross-validator : Provides train/test indices to split data in train/test sets.
# The stratified version of cross-validation ensures that there are equal proportions the predicted class in each train-test fold.
StratifiedKFold(n_splits=3, shuffle=False, random_state=None)
	.split(X, y, groups=None)  # Generate indices to split data into training and test set. Use as a generator (like iterator, but yields values once per call, not all at once)

	# Example 
	from sklearn.model_selection import StratifiedKFold
	scores = []
	skf = StratifiedKFold(n_splits=5)                   # create StratifiedKFold object 
    for train_index, test_index in skf.split(X, y):     # iterate through through the folds. train_index and test_index are arrays of indices 
        knn.fit(X[train_index], y[train_index])         # train, test, and score your classifier
        scores.append(knn.score(X[test_index], y[test_index]))         # save the accuracy score from each iteration to a list called scores
    print np.mean(scores) # print mean score for all folds 


#---- Grid Search Cross Validation : fits many models using a range of hyper parameters, then chooses the best coefficients for each hyper parameter set using (stratified?) Cross Validation to find the mminimum errir 
GridSearchCV(estimator, param_grid)
	.fit(X[, y, groups])		# Run fit with all sets of parameters.
	.predict(*args, **kwargs)	# Call predict on the estimator with the best found parameters.
	.score(X[, y])				# Returns the score on the given data, if the estimator has been refit.
	
	# Example
	from sklearn.model_selection import GridSearchCV
	X_train, X_test, y_train, y_test = train_test_split(*arrays, test_size=0.3)
	# Set the parameters of the estimator model to scan over 
	params = {
    	'n_neighbors':range(1,20),
    	'weights':['uniform','distance']
	}
	# create grid search CV 
	gs = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params)
	# fit model 
	gs.fit(X_train, y_train)
	# gs.cv_results_
	print gs.best_score_, gs.best_params_
	# score ????
	gs.best_estimator_.score(X_test, y_test)




#------------------- UNSUPERVISED LEARNING ------------------#

#---- K means clustering
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans
km = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300)

#---- silhouette score
from sklearn.metrics import silhouette_score
mean_coef = silhouette_score(X, labels, metric='euclidean', sample_size=None) # Returns : Mean Silhouette Coefficient for all samples.


#---- Hierarchical clustering 
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist

	# Example
	X = df.as_matrix(columns=df.columns.tolist()[:-1])
	Z = linkage(X, 'ward')

	c, coph_dists = cophenet(Z, pdist(X))
	print c

	def plot_dendogram(df):
	    # Data prep
	#     X = df.as_matrix(columns=df.columns.tolist()[:])
	    X = df.as_matrix(columns=None)
	    Z = linkage(X, 'ward')
	#     Z = linkage(X, 'complete')
	    
	    # plotting
	    plt.figure(figsize = (12,6))
	    plt.title('Dendrogram')
	    plt.xlabel('Index Numbers')
	    plt.ylabel('Distance')
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=15,  
        show_leaf_counts=False,  
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True
    )
	    
	    plt.show()
	    
	    return Z
    
    





#-------------------- SUPERVISED LEARNING -------------------#
        
#------------------- MODELS -------------------#



#----------- REGRESSION ------------#

#---- Linear Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
		
	# Example 
	lm = LinearRegression()
	lm.fit(X_train,y_train)
	predictions = lm.predict(X_test)
	# Plot predictions vs actual values
	plt.scatter(y_test,predictions)
	# residual histogram to test prediction
	sns.distplot((y_test-predictions),bins=50); 

#---- Ridge Regression
from sklearn.linear_model import Ridge
Ridge(alpha=)
	.fit()
	.score()

	# Example 
	from sklearn.linear_model import Ridge
	rr = Ridge(alpha=1.0)
	rr.fit(X=X_train, y=y_train)
	rr.score(X=X_test, y=y_test)

#---- Elastic Net Regression
from sklearn.linear_model import ElasticNet
en = ElasticNet(alpha=1.0, l1_ratio=0.5)

#---- Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()

#---- Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None)


#---- Ada Boost Regression
from sklearn.ensemble import AdaBoostRegressor
AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)



#---------- CLASSIFICATION ----------#

#--- Logistic Model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

#---- K Nearest Neighbors 
KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
	.fit(X, y)		# Fit the model using X as training data and y as target values
	.predict(X)		# Predict the class labels for the provided data
	.score(X, y)	# Returns the mean accuracy on the given test data and labels.

	# Example
	from sklearn.neighbors import KNeighborsClassifier
	knn_n5 = KNeighborsClassifier(n_neighbors=5, weights='uniform')


#---- Support Vector Classifier (SVC)
# Kernel Functions: http://scikit-learn.org/stable/modules/svm.html#svm-kernels
from sklearn.svm import SVC
svc = SVC(kernel='rbf')

	# Example






#------------------ ERROR METRICS --------------#

#---- classification reports
from sklearn.metrics import classification_report
sklearn.metrics.classification_report(y_true, y_pred, digits=4, target_names=None)

#---- ROC / AUC
from sklearn.metrics import roc_curve, auc

	# Example 
	from sklearn.metrics import roc_curve, auc

	y_prob = gs.best_estimator_.predict_proba(X_test)

	FPR = dict()
	TPR = dict()
	ROC_AUC = dict()

	# For class 1, find the area under the curve
	FPR[1], TPR[1], _ = roc_curve(y_test, y_prob[:, 1])
	ROC_AUC[1] = auc(FPR[1], TPR[1])

	plt.figure(figsize=[11,9])
	plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
	plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate', fontsize=18)
	plt.ylabel('True Positive Rate', fontsize=18)
	plt.title('Receiver operating characteristic for cancer detection', fontsize=18)
	plt.legend(loc="lower right")
	plt.show()

#---- Precision/Recall curves 
from sklearn.metrics import precision_recall_curve

	# Exmpale
	from sklearn.metrics import precision_recall_curve


	# Compute Precision-Recall and plot curve
	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(2):
	    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
	                                                        y_score[:, i])
	    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

	# Compute micro-average ROC curve and ROC area
	precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
	    y_score.ravel())
	average_precision["micro"] = average_precision_score(y_test, y_score,
	                                                     average="micro")

	# Plot Precision-Recall curve
	plt.clf()
	plt.plot(recall[0], precision[0], lw=lw, color='navy',
	         label='Precision-Recall curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
	plt.legend(loc="lower left")
	plt.show()

#-----------------------------
#--- TUTORIALS / WORKFLOWS 
#-----------------------------

#---- Plot a confusion matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline

class_names = ['over $200k','under $200k']

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

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


plt.show()







