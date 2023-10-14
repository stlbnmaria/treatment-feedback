import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the given dataframe by extracting drug, disease, and antibody information 
    and tokenizing the comments.
    
    Parameters:
    - dataframe (pd.DataFrame): The input dataframe. Must have 'medication' and 'comment' columns.
    
    Returns:
    - pd.DataFrame: The processed dataframe with added 'drug', 'disease', 'antibody', 
                    and 'processed_comment' columns.
    """
    
    # Extract drug, disease, and antibody information
    dataframe['drug'] = dataframe['medication'].str.extract('^(.*?)(?:\s*\(.*\)|\s*for)')[0].str.strip()
    dataframe['disease'] = dataframe['medication'].str.extract('for (.*?)(?:,|$)')[0].str.strip()
    dataframe['antibody'] = dataframe['medication'].str.extract('\(([^)]+)\)')
    
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase, tokenize, remove stopwords, and apply lemmatization
    dataframe['processed_comment'] = dataframe['comment'].apply(lambda text: [
        lemmatizer.lemmatize(token) 
        for token in word_tokenize(text.lower()) 
        if token not in stop_words and token.isalpha()
    ])
    
    return dataframe
