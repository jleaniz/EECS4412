import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

############################ CONFUGIRATION ###########################################
# Set to True to perform an exhaustive parameter grid search
# WARNING: It will take a long time to train if set to True
param_grid_search = False 

# Number of folds for cross validation
# NOTE: Only used if param_grid_search is True
k_fold = 3 

# Absolute path to the datasets - MUST be configured
train_dataset_path='train3.csv'
test_dataset_path='test3.csv'
stop_words_path='stop_words.txt'
#######################################################################################

def upsample_minority_class(df):
    '''
    This function performs up-sampling of the class with fewer samples
    to balance the training dataset more evenly.
    '''
    df_positive = df[df['class'] == 'positive']
    df_negative = df[df['class'] == 'negative']
    df_neutral = df[df['class'] == 'neutral']

    df_neutral_upsampled = resample(df_neutral, 
         replace=True,    # sample with replacement
         n_samples=11897, # to match negative class
         random_state=42 # to have reproducible results over multiple runs 
    )
    
    # Combine with upsampled minority class
    df_upsampled = pd.concat([df_positive, df_negative, df_neutral_upsampled])
    return df_upsampled

def downsample_majority_class(df):
    '''
    This function performs down-sampling of the class with fewer samples
    to balance the training dataset more evenly.
    '''
    df_positive = df[df['class'] == 'positive']
    df_negative = df[df['class'] == 'negative']
    df_neutral = df[df['class'] == 'neutral']

    df_positive_downsampled = resample(df_positive, 
         replace=False,    # sample without replacement
         n_samples=20000, # downsample to 20,000 samples
         random_state=42 # to have reproducible results over multiple runs 
    )
    
    # Combine with downsampled majority class
    df_downsampled = pd.concat([df_neutral, df_negative, df_positive_downsampled])
    return df_downsampled

def load_and_split():
    '''
    This function loads the train3.csv dataset from the assignment and performs
    pre-processing tasks such as stop word removal and train/test split
    
    return: x_train: 'text' values used for training
            x_test: 'text' values used for testing accuracy of the model
            y_train: 'class' values used for training
            y_test: 'class' values used for testing accuracy of the model
    '''
    df = pd.read_csv(train_dataset_path)
    df = upsample_minority_class(df)
    df = downsample_majority_class(df)
    df = df.set_index('ID')
    
    x = df['text']
    y = df['class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
   
    return (x_train, x_test, y_train, y_test)


def fit_classifier(x_train, y_train, classifier, param_search, n):
    '''
    This function trains a classifier model using a scikit-learn Pipeline.

    Performing a parameter grid search with k-fold cross-validation is optional
    and can be enabled by setting 'param_grid_search' to True.

    x_train: 'text' values used for training the model
    y_train: 'class' values used for training the model
    classifier: 'svm', 'logreg', 'rf'
    param_search: boolean value, True or False
    n: cross validation n-fold (integer)
    
    return: trained SVM classifier
    '''
    # Load stop words from stop_words.txt
    try:
        f = open(stop_words_path, 'r')
        stop_words_list = f.read().split('\n')
        f.close()
    except OSError as err:
        print("OS error: {0}".format(err))

    if classifier == 'svm':    
        parameters = {
            'sgd__max_iter':[500,2000,2000],
            'sgd__n_iter_no_change': [5, 10, 20],
            'sgd__class_weight': ('balanced', None)
        }
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(strip_accents='unicode', stop_words=stop_words_list, min_df=3, ngram_range=(1,2))),
            ('svm', SGDClassifier(n_jobs=-1))
        ])
        
    elif classifier == 'logreg':
        parameters = {
            'lr__max_iter':[1000,2000,10000],
            'lr__class_weight': ('balanced',None),
            'lr__solver': ('saga', 'newton-cg', 'lbfgs'),
            'lr__penalty': ('l2', 'elasticnet')                   
        }
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(strip_accents='unicode',stop_words=stop_words_list,  min_df=3, ngram_range=(1,2))),
            ('logreg', LogisticRegression(solver='liblinear',multi_class='auto'))
            #('lr', KNeighborsClassifier(n_jobs=-1))
        ])
        
    elif classifier == 'rf':
        parameters = {
            'rf__n_estimators':[100,500,1000],
            'rf__criterion': ('gini', 'entropy')
        }
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(strip_accents='unicode', stop_words=stop_words_list, min_df=3, ngram_range=(1,2))),
            ('rf', RandomForestClassifier(n_estimators=500, criterion='gini',n_jobs=-1))
        ])
        
    else:
        print("Invalid classifier option")
        return
    
    if param_search:
        # parameter grid search + cross validation
        model = GridSearchCV(pipe, parameters, n_jobs=-1, cv=n)
        model.fit(x_train, y_train)
    else:
        # cross validation only
        model = pipe.fit(x_train, y_train)
        
    return model

def validate_classifier(model, model_name, x_test, y_test):
    '''
    Predict values using the validation test set from our 70/30 split
    and output accuracy of the models
    '''
    # Define list of classes in the dataset
    data_classes = ['positive', 'negative', 'neutral']

    y_pred = model.predict(x_test)
    print(model_name,' accuracy is: ', accuracy_score(y_test, y_pred),'\n')
    print(classification_report(y_test, y_pred,target_names=data_classes))
    report = classification_report(y_test, y_pred,target_names=data_classes, output_dict=True)
    return report

def main():
    # Show some information about the dataset
    df = pd.read_csv(train_dataset_path)
    print("==== TRAINING DATASET CLASSES ===")
    print("The number of reviews for each class is: \n",df['class'].value_counts())
    print("")
    '''
    Split the 'train3' dataset into train and test datasets which we will use to train
    the SVM classifier and test the accuracy of the model
    '''
    x_train, x_test, y_train, y_test = load_and_split()

    # Train the models
    print("=== Training a Linear SVM Classifier ===")
    sgd = fit_classifier(x_train, y_train, 'svm', param_grid_search, k_fold)
    print("=== Training a Logistic Regression Classifier ===")
    logreg = fit_classifier(x_train, y_train, 'logreg', param_grid_search, k_fold)
    print("=== Training a Random Forest Classifier ===")
    rf = fit_classifier(x_train, y_train, 'rf', param_grid_search, k_fold)

    # Test the accuracy of the models using the validation test set from the split
    svm_report = validate_classifier(sgd, 'Linear SVM Classifier', x_test, y_test)
    logreg_report = validate_classifier(logreg, 'Logistic Regression Classifier', x_test, y_test)
    rf_report = validate_classifier(rf, 'Random Forest Classifier', x_test, y_test)

    # Print the name and accuracy of the best model
    f_measure_scores = {
        'linear_svm': svm_report['weighted avg']['f1-score'],
        'logistic_reg': logreg_report['weighted avg']['f1-score'],
        'random_Forest': rf_report['weighted avg']['f1-score']
    }
    best_f_measure = f_measure_scores[max(f_measure_scores, key=f_measure_scores.get)]
    best_model_name = list(f_measure_scores.keys())[list(f_measure_scores.values()).index(best_f_measure)]
    print("Best model (using f1-score/f-measure): name=",best_model_name,'avg f1-score=', best_f_measure,"\n")

    # Predict values for the unlabeled dataset
    df_rf_pred = pd.read_csv("test3.csv")
    df_rf_pred = df_rf_pred.set_index('ID')
    rf_preds = rf.predict(df_rf_pred['text'])
    df_rf_pred['CLASS'] = rf_preds
    df_rf_pred.index.names = ['REVIEW-ID']
    df_rf_pred = df_rf_pred.drop(columns=['text'])

    # Save the predicted outputs to a csv file
    print("Saving predicted outputs to prediction.csv at the current working directory\n")
    df_rf_pred.to_csv("prediction.csv")
    print("=== PREDICTED CLASS VALUE COUNT ===")
    print("Total reviews in the test dataset: ", len(df_rf_pred))
    print("Number of reviews per class: \n", df_rf_pred['CLASS'].value_counts())
    print("\nAll tasks completed.")

main()