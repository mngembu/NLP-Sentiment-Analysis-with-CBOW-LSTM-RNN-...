# NLP-Sentiment-Analysis-with-CBOW-LSTM-RNN-...


## Overview
This project focuses on predicting whether stock prices will increase or decrease based on sentiment analysis of news headlines. By leveraging Natural Language Processing (NLP) and various machine learning algorithms, the project aims to identify the correlation between news sentiment and stock market movements.

## Dataset
The dataset used in this study is sourced from Kaggle: Stock News Sentiment Analysis Massive Dataset. Given its large size with 108,301 unique values, a sample of 5000 observations is utilized for the analysis.

## Project Structure
The project explores different algorithms to address the use case. Each approach is documented in separate notebooks:

1. **LSTM (Long Short Term Memory) RNN (Recurrent Neural Network) with NLP**:
- Implements deep learning techniques to analyze sentiment using sequential data.
2. **Random Forest with NLP's Bag of Words (BOW)**:
- Utilizes a traditional machine learning algorithm in combination with BOW for sentiment classification.
3. **Word2Vec (CBOW and Skip Grams) with Machine Learning and Deep Learning**:
- Applies advanced word embedding techniques to capture contextual word representations and enhance sentiment prediction.

## Dependencies
This project is implemented in Python and relies on the following libraries:
- Pandas
- Numpy
- NLTK
- TensorFlow
- Gensim

## Installation
To run this project, you need to have Python installed on your system. You can install the required libraries using pip:

      pip install pandas numpy nltk tensorflow gensim

## Usage
Each notebook in this repository demonstrates a unique approach to sentiment analysis. You can run the notebooks in any Jupyter environment. Below are the steps to get started:

1. **Clone the Repository**:

            git clone https://github.com/mngembu/NLP-Sentiment-Analysis-with-CBOW-LSTM-RNN-....git
            cd NLP-Sentiment-Analysis-with-CBOW-LSTM-RNN-...

2. **Open Jupyter Notebook**:

            jupyter notebook

3. **Run the Notebooks**:

- Open each notebook and run the cells to see the implementation of different algorithms.


## Notebooks
- **LSTM-RNN-NLP.ipynb**: Implements LSTM RNN for sentiment analysis.
- **RandomForest-BOW-NLP.ipynb**: Uses Random Forest classifier with Bag of Words.
- **Word2Vec-ML-DL.ipynb**: Explores CBOW and Skip Grams with both machine learning and deep learning techniques.

## Results
The results from different models are compared to evaluate their performance in predicting stock price movements based on news sentiment. Detailed results and visualizations are available within the respective notebooks.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
- Kaggle for providing the dataset.
- The authors and contributors of the libraries used in this project.

Feel free to explore, use, and modify the project to suit your needs. Happy coding!







