import pandas as pd
import time

from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz as fuzzywuzzy_fuzz, process as fuzzywuzzy_process
from flask import Flask, render_template, request, jsonify
from jinja2 import Environment, FileSystemLoader
from pyjarowinkler import distance
from collections import Counter
from math import sqrt

# Define the Flask p n ij2tmlt environment
env = Environment(loader=FileSystemLoader('templates'))
env.globals.update(zip=zip)
app = Flask(__name__)
app.jinja_env = env

# Path
PATH_APOTEK = 'apotek_medicalclinicid_cleaned.csv'
PATH_RS = 'rumahsakit_medicalclinicid_cleaned.json'
PATH_PENYAKIT = 'penyakit_medicalclinicid_cleaned.csv'

# Read file CSV
data = pd.read_json(PATH_RS)

# String initial and space deletion 
def cleanString(text, pickstart, delspace=True):
    string_with_words = text
    pickup_list = string_with_words.split()
    
    if len(pickup_list) > pickstart: 
      pickup_list = pickup_list[pickstart:]

    pickup_word = "".join(pickup_list)

    if delspace:
      return pickup_word.replace(" ", "")
    else:
      return pickup_word

def fuzzy_search(query, names, limit=10):
    results = fuzzywuzzy_process.extract(query, names, limit=limit)
    return results 

def jarowink_search(dataframe, pattern, threshold=0.8):
    occurrences = []
    for idx, substring in dataframe.items():
        try:  
            similarity = distance.get_jaro_distance(substring, pattern, winkler=True, scaling=0.1)
            if similarity is not None and similarity >= threshold:
                    occurrences.append([idx, similarity])                
        except Exception as e:
          continue

    def takeSecond(elem):
        return elem[1]
    
    occurrences.sort(key=takeSecond, reverse=True)  
    return occurrences

def cosine_similarity(s, t):
    s_counter, t_counter = Counter(s), Counter(t)
    intersection = sum(s_counter[char] * t_counter[char] for char in s_counter)
    s_norm = sqrt(sum(s_counter[char] ** 2 for char in s_counter))
    t_norm = sqrt(sum(t_counter[char] ** 2 for char in t_counter))
    return intersection / (s_norm * t_norm)

def lcs_similarity(s, t):
    n, m = len(s), len(t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m] / max(n, m)

def dice_coefficient_similarity(s, t):
    s_bigrams = set(zip(s[:-1], s[1:]))
    t_bigrams = set(zip(t[:-1], t[1:]))
    intersection = len(s_bigrams & t_bigrams)
    return 2 * intersection / (len(s_bigrams) + len(t_bigrams))

def ngram_similarity(query, document, n=3):
    query_ngrams = set(ngrams(query.lower(), n))
    document_ngrams = set(ngrams(document.lower(), n))
    intersection = query_ngrams.intersection(document_ngrams)
    union = query_ngrams.union(document_ngrams)
    return len(intersection) / len(union)

def search_data(data, query, threshold=0.5):
    algorithms = ['fuzzy', 'ngram', 'jarowinkler']
    results = {}
    processing_times = {}
    for algorithm in algorithms:
        start_time = time.time()
        if algorithm == 'fuzzy':
            fuzzy_results = fuzzy_search(query, data)
            results[algorithm] = [[result[2], result[1]] for result in fuzzy_results]
        elif algorithm == 'ngram':
            results[algorithm] = [[idx, round(globals()[f"{algorithm}_similarity"](query, value),2)] for idx, value in data.items() if globals()[f"{algorithm}_similarity"](query, value) >= threshold]
        elif algorithm == 'jarowinkler':
            occurrences = jarowink_search(data, query)
            results[algorithm] = [[name, similarity] for name, similarity in occurrences]
            #results[algorithm] = [idx for idx, value in data.items() if cosine_similarity_custom(query, value) >= threshold]   
        end_time = time.time()
        processing_times[algorithm] = end_time - start_time
    return results, processing_times

@app.route('/')
def home_dashboad():
    results_df = data.to_dict(orient='records')
    return render_template('index.html', results=results_df)


@app.route('/search_rumahsakit', methods=['GET', 'POST'])
def search_rumahsakit():
    query = request.form.get('query')
    if query:
        #data = pd.read_csv(PATH_RS)
        results, processing_times = search_data(data['Nama Rumah Sakit'], query)
        results_df = {algorithm: data.loc[(idx for idx, score in indices)].to_dict(orient='records') for algorithm, indices in results.items()}
        score_df = {algorithm: [[idx, score] for idx, score in indices] for algorithm, indices in results.items()}
    else:
        results_df = pd.DataFrame()
        score_df = {}
        processing_times = {}
    return jsonify({"results": results_df, "score": score_df, "processing_times": processing_times})

if __name__ == '__main__':
    app.run(debug=True)