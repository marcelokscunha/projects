import json
import plotly
import pandas as pd
import re
from collections import defaultdict

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Table
from joblib import load
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    # remove leading and trailing spaces before lemmatizing
    tokens = [lemmatizer.lemmatize(word.strip()) for word in tokens if word not in stop_words]  

    return tokens

# load data
db_path = '../data/DisasterResponse.db'
engine = create_engine(f'sqlite:///{db_path}')

df = pd.read_sql('disaster_data', engine)
df.drop('child_alone', axis=1, inplace=True)
X = df.iloc[:,1].values
Y = df.iloc[:,4:].values
col_names = df.columns

# load model - best model was the Grid Search tuned Gradient Boosting Classifier
best_model = 'cv_gbc'
model = load(f'../models/{best_model}.joblib')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visual 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract data needed for visual 2
    top_k = 15
    most_freq_categories = df.iloc[:,4:].sum(axis=0).sort_values(ascending=False)[:top_k].index.values
    df_X = df.iloc[:,4:]
    value_counts = defaultdict(list)
    for col in most_freq_categories:
        value_counts[0].append(df_X[col].value_counts()[0])
        value_counts[1].append(df_X[col].value_counts()[1])
        if len(df_X[col].value_counts())==3:
            value_counts[2].append(df_X[col].value_counts()[2])
            
    # extract data needed for visual 3
    df_metadata = pd.read_csv('../data/metadata.csv', sep=';')
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        
        {
            'data': [
                Bar(
                    x=most_freq_categories,
                    y=value_counts[0],
                    name='0',
                    marker_color='indianred'
                ),
                Bar(
                    x=most_freq_categories,
                    y=value_counts[1],
                    name='1',
                    marker_color='lightsalmon'
                ),
                Bar(
                    x=[most_freq_categories[0]],
                    y=value_counts[2],
                    name='2',
                    marker_color='purple'
                )
            ],

            'layout': {
                'title': f'Distributions of values for the top-{top_k} Labels that appear the most (class imbalances)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label"
                },
                'barmode':'group',
                'xaxis_tickangle':-45
            }
        },
        
        {
            'data': [
                Table(
                      header=dict(
                                values=list(df_metadata.columns),
                                align='center',
                                fill_color='darkslateblue',
                                font=dict(color='White', size=14)
                      ),
                      cells=dict(
                                values=[df_metadata[col] for col in df_metadata.columns],
                                align='center',
                                line_color='white',
                                fill=dict(color='white')
                      )
                )]
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i in range(len(graphs))]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(col_names[4:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()