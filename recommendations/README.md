# Udacity's Tech Article Recommendations Project
## Introduction
<img src="./media/articles-ibm.png" alt="Disaster Response" width="60%">

In this project a recommender system is developed to make recommendations to users about technical articles they could like. The datasets given are about interactions that users have with articles and about the articles themselves on the IBM Watson Studio platform. 

## Project Structure
The project is divided into the following tasks:

I. Exploratory Data Analysis

II. Rank Based Recommendations (most popular articles)

III. User-User Based Collaborative Filtering (similar users by comparing user-item interactios)

IV. Content Based Recommendations (extraction of features from articles, e.g. topics with Latent Dirichlet Allocation, TF-IDF, visualization with <a href="https://github.com/bmabey/pyLDAvis">pyLDAvis</a>, t-SNE)

V. Matrix Factorization (applying Singular Value Decomposition for recommendations)

The final recommender system would be a mix of the developed Rank based, Collaborative Filtering based (both User-User and User-Item), Content based recommender system. By doing this, problems like the <a href="https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)">cold start problem</a> can be avoided, and recommendations are provided  for all users and including all items.

For more information: [see notebook](./Recommendations_with_IBM.ipynb)

## Possible improvements (out of the scope of this project)

As possible improvements, our dataset with user information could be enriched (age, gender, job role, etc.). A knowledge based system could be added to the rank based recommendations enabling users to filter by article type (e.g. programming languages, authors, frameworks, use cases, etc.). Recommendations would include the most popular items matching the filters. In the Content based recommender system other methods of feature extraction using Neural Networks (Word2Vec, <a href="https://radimrehurek.com/gensim/models/doc2vec.html">Doc2vec</a>, <a href="https://github.com/cemoody/lda2vec">LDA2Vec</a>), pre-trained language models (BERT, GPT-2) and so on could be tested.

For testing our recommender system offline, other metrics could be computed to measure how the models are recommending relevant articles (that interest more users), novel items (that users haven't seen), serendipitous articles (items that are different from what users have already seen, related to different topics, different article types, etc.), diverse articles (how much of our collection of articles are recommended, if all types of aticles and topics are recommended, and so on.).

Finally to productionize our recommender system we could develop a scalable API for our model (e.g. in AWS with SageMaker), externalize pre-computed features like pairwise distances of articles, user-item interactions, etc. on a feature store (e.g. graph, NoSQL, SQL databases, S3, HDFS), monitor our models (e.g. collecting new user-article interactions, perform A/B tests, etc.), re-train, re-compute and update features in the feature store.