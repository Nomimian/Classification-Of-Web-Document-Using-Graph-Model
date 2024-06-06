import os
import string

def preprocess_text(text):
    # Removing punctuation
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)
    
    # Convert to lowercase
    text = text.lower()

    return text

def preprocess_files():
    folder_path = os.getcwd()  # Get the current working directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                preprocessed_text = preprocess_text(text)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(preprocessed_text)

if __name__ == "__main__":
    preprocess_files()
    print("Preprocessing completed successfully.")
