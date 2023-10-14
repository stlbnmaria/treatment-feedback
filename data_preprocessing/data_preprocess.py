import pandas as pd
import nltk
import yaml
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_data(filter: str = "yes") -> pd.DataFrame:
    """
    Preprocesses the CSV file specified in config.yaml by extracting treatment, disease, and antibody information 
    and tokenizing the comments.
    
    Parameters:
    - filter (str): If set to "yes", filter the dataframe to only include rows where 
                    the disease is 'Crohn's Disease' or 'Ulcerative Colitis'.
                    If "no", no filtering is applied.
    
    Returns:
    - pd.DataFrame: The processed dataframe with added 'treatment', 'disease', 'antibody', 
                    and 'processed_comment' columns.
    """
    
    # Read the path from the config.yaml file
    with open("config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    file_path = cfg['file_path']
    
    # Read the CSV file into a dataframe
    dataframe = pd.read_csv(file_path)
    
    # Extract treatment, disease, and antibody information
    dataframe['treatment'] = dataframe['medication'].str.extract('^(.*?)(?:\s*\(.*\)|\s*for)')[0].str.strip()
    dataframe['disease'] = dataframe['medication'].str.extract('[fF]or (.*?)(?:,|$)')[0].str.strip()
    dataframe['antibody'] = dataframe['medication'].str.extract('\(([^)]+)\)')
    
    # Filter the dataframe based on the disease if filter is set to "yes"
    if filter == "yes":
        dataframe = dataframe[dataframe['disease'].isin(["Crohn's Disease", "Ulcerative Colitis"])]
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase, tokenize, remove stopwords, and apply lemmatization
    dataframe['processed_comment'] = dataframe['comment'].apply(lambda text: [
        lemmatizer.lemmatize(token) 
        for token in word_tokenize(text.lower()) 
        if token not in stop_words and token.isalpha()
    ])
    
    return dataframe
