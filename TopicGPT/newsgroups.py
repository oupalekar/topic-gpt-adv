import numpy as np
from sklearn.datasets import fetch_20newsgroups
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from tqdm import tqdm
import os
import json

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class NewsGroupsPreprocessor:
    def __init__(self, remove_headers=True, remove_footers=True, remove_quotes=True):
        """
        Initialize the preprocessor with customizable options
        
        Args:
            remove_headers (bool): Whether to remove email headers
            remove_footers (bool): Whether to remove footers
            remove_quotes (bool): Whether to remove quoted text
        """
        self.remove_headers = remove_headers
        self.remove_footers = remove_footers
        self.remove_quotes = remove_quotes
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and normalize text content"""
        # Convert to lowercase
        text = text.lower()    
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def lemmatize_text(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(token) for token in tokens 
                        if token not in self.stop_words and len(token) > 2])
    
    def preprocess_dataset(self, subset, output_file='20news'):
        """
        Fetch and preprocess the 20 newsgroups dataset
        
        Args:
            output_file (str): Path to save the preprocessed documents
        """
        # Fetch the dataset
        cats = ['alt.atheism','comp.os.ms-windows.misc','comp.graphics','comp.sys.ibm.pc.hardware',
                    'comp.sys.mac.hardware', 'comp.windows.x', 'rec.autos','misc.forsale','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics',
                    'sci.med','sci.space','talk.politics.guns','soc.religion.christian','talk.politics.misc','talk.politics.mideast','talk.religion.misc']

        print("Fetching 20 Newsgroups dataset...")
        processed_docs = []
        out_file = "./20newsgroups.jsonl"
        for cat in cats:
            newsgroups = fetch_20newsgroups(subset=subset, 
                                        remove=('headers', 'footers', 'quotes')
                                        if all([self.remove_headers, 
                                                self.remove_footers, 
                                                self.remove_quotes]) else [], categories= [cat])
            
            # Process each document
            print("Preprocessing documents...")
            for j, doc in tqdm(enumerate(newsgroups.data)):
                doc = re.sub(r'\s+', ' ', doc).strip()
                json_obj = {'id': f'{cat}-{j}', "text": doc, "label": cat}
                # print(doc)
                # return
                # Clean and lemmatize the text
                # cleaned_text = self.clean_text(doc)
                # processed_text = self.lemmatize_text(cleaned_text)
                
                # Only add non-empty documents
                # if processed_text.strip():
                processed_docs.append(json_obj)
            # out_file = f'./data/20news-individual/{cat}_{subset}_lines.txt'
            # Save to file
            print(f"Saving {len(processed_docs)} documents to {out_file}")
            with open(out_file, 'w', encoding='utf-8') as f:
                for doc in processed_docs:
                    f.write(json.dumps(doc) + '\n')
        
        return processed_docs

def main():
    """Main function to run the preprocessing"""
    # Initialize preprocessor
    preprocessor = NewsGroupsPreprocessor(
        remove_headers=True,
        remove_footers=True,
        remove_quotes=True
    )
    
    # Process the dataset
    preprocessor.preprocess_dataset('train')
    # test_processed_docs = preprocessor.preprocess_dataset('test')

    
    # Print some statistics
    print("\nPreprocessing completed!")
    # print(f"Total documents processed: {len(train_processed_docs)}")
    # print(f"Average document length: {np.mean([len(doc.split()) for doc in train_processed_docs]):.2f} words")
    # print(f"Median document length: {np.median([len(doc.split()) for doc in train_processed_docs]):.2f} words")
    # print()
    # print(f"Total documents processed: {len(test_processed_docs)}")
    # print(f"Average document length: {np.mean([len(doc.split()) for doc in test_processed_docs]):.2f} words")
    # print(f"Median document length: {np.median([len(doc.split()) for doc in test_processed_docs]):.2f} words")

if __name__ == "__main__":
    main()