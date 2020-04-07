import pandas as pd
from sklearn.utils import resample
#from trainer import upsample_minority_class, downsample_majority_class

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

def remove_stop_words(text):
    '''
    This function is used to decode the base64 stop words
    No need to modify this
    
    text: some text
    return: list of stop words
    '''
    try:
        text = text.split()
        resultwords  = [word.lower() for word in text if word.lower() not in stop_words]
        result = ' '.join(resultwords)
        return result
    except:
        print("Error removing stop words!")
        exit(1)

def main():
    df = pd.read_csv("train3.csv")
    df = upsample_minority_class(df)
    df = downsample_majority_class(df)
    df['text'] = df['text'].apply(remove_stop_words)
    df = df.set_index('ID')
    df.to_csv('preprocessed_train3.csv')
    print("Done removing stopwords. See preprocessed_train3.csv")


f = open('stop_words.txt')
stop_words = f.read().split()
f.close()
main()