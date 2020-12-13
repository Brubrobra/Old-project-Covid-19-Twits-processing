# COVID19Predictor

COVID19Predictor is a Apache Spark based Machine Learning model designed to predict Novel Coronavirus cases in Canada. It does so by using historical tweets from Twitter.

### Dataset
* Historical top 1000 terms, bigrams and trigrams on twitter
* Provincial Daily Cases (Canada)

### Preprocessing
The data is processed into a more Spark-readable format in Preprocess.scala and SentimentPreprocess.scala.

### Sentiment Analysis
Obtain a sentiment analysis of twitter sentiment of COVID19. Use this sentiment as a feature to ML algorithm.

### Algorithm
Use top terms as features to a Lasso Regression. In addition, use the COVID19 sentiment as a feature.

### Results
* Predictions were not accurate
* More data is needed for proper sentiment analysis (preferably full tweets)
* Potentially a better model could have been used (e.g. LSTM)
