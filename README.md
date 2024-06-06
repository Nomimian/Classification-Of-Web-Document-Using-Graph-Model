# Document Classification Using Graph Model and KNN

This project focuses on document classification using a graph-based model and the K-Nearest Neighbors (KNN) algorithm. The primary objective is to classify documents into different categories by leveraging data scraped from various websites.

## Project Workflow

### 1. Data Collection
- **Websites:** Data is scraped from 45 different websites.
  - **Categories:**
    - Food
    - Lifestyle & Hobbies
    - Marketing & Sales
  - **Data Split:**
    - **Training Set:** Data from 36 websites.
    - **Test Set:** Data from 9 websites.
  - **Topics:** Collect 15 topics from each of these three categories.

### 2. Data Preprocessing
- **Cleaning:** Remove HTML tags, scripts, and other non-text elements.
- **Tokenization:** Break down the text into individual words or tokens.
- **Stop Words Removal:** Eliminate common words that do not contribute to classification.
- **Stemming/Lemmatization:** Reduce words to their root forms for consistency.

### 3. Graph-Based Representation
- **Graph Model:** Convert the preprocessed text data into a graph.
  - **Nodes:** Represent unique words or tokens.
  - **Edges:** Represent relationships or co-occurrences between words within documents.

### 4. Feature Extraction
- Extract features from the graph representations for the KNN model.
  - **Metrics:** Node degrees, centrality measures, and other graph-based metrics.

### 5. KNN Classification
- **Algorithm:** Implement the KNN algorithm to classify documents.
- **Data Split:** Split the data into training and test sets.
- **Training:** Use the training set to train the KNN model.
- **Evaluation:** Use the test set to evaluate model performance.

### 6. Confusion Matrix
- **Evaluation:** Assess the performance of the KNN model using a confusion matrix.
  - **Metrics:** Accuracy, precision, recall, and F1-score.

## Tools and Libraries
- **Web Scraping:** BeautifulSoup, Scrapy
- **Text Preprocessing:** NLTK, spaCy
- **Graph Construction and Analysis:** NetworkX
- **Machine Learning:** scikit-learn
- **Data Handling:** Pandas, NumPy
- **Evaluation:** sklearn.metrics (confusion matrix, classification report)

## Detailed Steps

### Scraping Data
- Use web scraping tools to extract text data from the websites.
- Store the scraped data in a structured format.

### Preprocessing Data
- **Cleaning:** Remove unnecessary elements from the text.
- **Tokenization:** Break the text into individual words.
- **Stop Words Removal:** Filter out common words that do not contribute to the classification.
- **Stemming/Lemmatization:** Apply stemming or lemmatization to reduce words to their root forms.

### Creating Graphs
- Represent each document as a graph where words are nodes and edges represent word co-occurrences.
- Use graph libraries for constructing and analyzing the graphs.

### Extracting Features
- Compute graph metrics such as node degree, closeness centrality, and betweenness centrality.
- Use these metrics as features for the KNN model.

### Applying KNN
- Implement the KNN algorithm using an appropriate library.
- Select a suitable distance metric (e.g., Euclidean, cosine).
- Train the model with the training set and classify documents in the test set.

### Evaluating the Model
- Generate a confusion matrix to evaluate classification performance.
- Calculate accuracy, precision, recall, and F1-score from the confusion matrix.

## Project Goals
This project aims to efficiently classify documents from various domains using a combination of graph-based features and the KNN algorithm, providing insights into the categorization and relationships within the data.
