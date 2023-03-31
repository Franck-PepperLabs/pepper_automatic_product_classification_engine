# **Classification automatique de produits**

[tx_prep.py]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/tx_prep.py
[tx_ml.py]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/tx_ml.py
[tx_ex.ipynb]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/tx_ex.ipynb
[tx_prep.ipynb]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/tx_prep.ipynb
[tx_ml.ipynb]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/tx_ml.ipynb
[tx_ml_emb.ipynb]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/tx_ml_emb.ipynb

[im_prep.py]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/im_prep.py
[im_ml.py]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/im_ml.py
[im_ex.ipynb]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/im_ex.ipynb
[im_prep.ipynb]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/im_prep.ipynb
[im_ml.ipynb]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/im_ml.ipynb
[im_ml_emb.ipynb]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/im_ml_emb.ipynb
[pipeline.py]: https://github.com/Franck-PepperLabs/pepper_automatic_product_classification_engine/blob/main/notebooks/pipeline.ipynb

## Classification des textes

* [**`tx_prep.py`**][tx_prep.py] : module principal pour le prétraitrement des textes.
* [**`tx_ml.py`**][tx_ml.py] : module principal pour la classification des textes.
* [**`tx_ex.ipynb`**][tx_ex.ipynb] : exemple (retravaillé) du NLP appliqué au cas [**Twitter US Airline Sentiment** (Kaggle)](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment).
* [**`tx_prep.ipynb`**][tx_prep.ipynb] : prétraitement des corpus flipkart.
* [**`tx_ml.ipynb`**][tx_ml.ipynb] : approche NLP classique basée sur tf-idf, bag of words, réduction de dimensionnalité (ACP, LDA) et classification k-Means.
* [**`tx_ml_emb.ipynb`**][tx_ml_emb.ipynb] : approche NLP contemporaine basée sur le *word embedding* Word2Vec, BERT et USE.

## Classification des images

* [**`im_prep.py`**][im_prep.py] : module principal pour le prétraitrement des images.
* [**`im_ml.py`**][im_ml.py] : module principal pour la classification des images.
* [**`im_ex.ipynb`**][im_ex.ipynb] : exemple (retravaillé) du CV appliqué au cas [**...** (Kaggle)](...).
* [**`im_prep.ipynb`**][im_prep.ipynb] : prétraitement des images flipkart.
* [**`im_ml.ipynb`**][im_ml.ipynb] : approche CV classique basée sur extraction de caractéristiques, bag of features, réduction de dimensionnalité et classification k-Means.
* [**`im_ml_emb.ipynb`**][im_ml_emb.ipynb] : approche CV contemporaine basée le *transfer learning* de modèles de *deep learning** préentraînés de type CNN.




Projet de NLP et CV appliqué à la classification automatique de produits, basée sur des textes de description et des images des produits.


# Documentation des modules

## [**`tx_prep.py`**][tx_prep.py]

Le module `tx_prep` contient un ensemble de fonctions pour le prétraitement de données textuelles. Ces fonctions incluent notamment le comptage d'occurrences d'étiquettes, le calcul de diverses métriques textuelles et la transformation de texte en un format adapté à l'analyse.

* `save_lexicon(data: pd.DataFrame, name: str) -> None` :
    * Sauvegarde un Dataframe en fichier CSV.
    * Les paramètres :
        * `data` : le Dataframe à sauvegarder.
        * `name` : le nom du fichier CSV.

* `word_tokenize_series(series: pd.Series) -> pd.Series`
    * Décompose une série de chaînes en utilisant la fonction `word_tokenize` du package NLTK.
    * Les paramètres :
        * `series` : la série des chaînes à décomposer.

* `count_tag_occurrences(series: pd.Series, tag: str) -> pd.Series` :
    * Compte le nombre d'occurrences de `tag` dans chaque élément d'une série de chaînes de caractères.

* `word_counts(sentences: pd.Series) -> pd.Series` :
    * Compte le nombre de mots dans chaque phrase d'une série de chaînes de caractères.
    * Les paramètres :
        * `sentences` : la série de chaînes qui représentent les phrases.

* `show_word_count_dist(word_count: np.ndarray, clip: Union[Tuple[int, int], int] = None) -> None` :
    * Affiche un histogramme de la distribution des fréquences des mots dans un corpus de texte donné sous forme de vecteur de comptage.
    * Les paramètres :
        * `word_count` : un vecteur de comptage des mots.

        * `clip` : un tuple `(a, b)` ou un entier `a`, qui indique la plage d'affichage sur l'axe des x. Par défaut, `clip` est `(1, word_count.max())`.

* `get_lexicon_dict(sentences: pd.Series) -> dict` :
    * Retourne un dictionnaire qui associe sa fréquence à chaque mot unique du corpus de phrases donné.
    * Les paramètres :
        * `sentences` : une série de phrases.

* `get_lexicon_dataframe(sentences: pd.Series, tokenize_func: Optional[Callable[[str], List[str]]] = word_tokenize) -> pd.DataFrame` :
    * Retourne un DataFrame qui contient le nombre de fois où chaque mot apparaît dans le corpus de phrases donné, ainsi que la longueur de chaque mot.
    * Les paramètres :
        * `sentences` : une série de phrases.
        * `tokenize_func` : une fonction utilisée pour décomposer chaque chaîne de la série. La valeur par défaut indique d'utiliser la fonction `word_tokenize` du package NLTK.

* `show_lexicon_dist(lexicon_data, feature='len', clip: Optional[Union[int, Tuple[int, int]]] = None)` :
    * Affiche un histogramme de la distribution des longueurs des mots dans un corpus donné sous forme de DataFrame.
    * Les paramètres :
        * `lexicon_data` : le DataFrame généré par `get_lexicon_dataframe`.
        * `feature` : la caractéristique à afficher sur l'axe des x. Par défaut, `feature` est `'len'`.
        * `clip` : un tuple `(a, b)` ou un entier `a`, qui indique la plage d'affichage sur l'axe des x. Par défaut, `clip` est `(1, word_count.max())`.

* `get_casefolded_lexicon_dataframe(sentences: pd.Series) -> pd.DataFrame` :
    * Retourne un DataFrame contenant les mots en en minuscules d'une série de phrases.
    * Les paramètres :
        * `sentences` : la série de phrases.

* `get_nltk_en_stopwords() -> List[str]` :
    * Récupère une liste des stopwords en anglais depuis la bibliothèque NLTK.

* `casefold_words(words: pd.Series) -> pd.Series` :
    * Convertit tous les mots d'une série de chaînes de caractères en minuscules.

* `is_stopword_en(words: pd.Series) -> pd.Series` :
    * Retourne une série de booléens indiquant si chaque mot d'une série de chaînes de caractères est un stopword en anglais.

* `show_n_grams(lexicon_data: pd.DataFrame, n: int) -> None` :
    * Affiche une liste triée des n-grammes dans un corpus donné sous forme de DataFrame, triée par ordre décroissant de fréquence.
    * Les paramètres :
        `lexicon_data` : le DataFrame généré par `get_lexicon_dataframe`.
        `n` : le nombre de mots dans chaque n-gramme à afficher.

* `is_integer(words: pd.Series) -> pd.Series` :
    * Retourne une série de booléens indiquant si chaque mot d'une série de chaînes de caractères est un nombre entier.

* `is_upper_alpha(words)`
    * Retourne une série de booléens indiquant si chaque mot d'une série de chaînes de caractères est entièrement composé de lettres majuscules.

* `compute_text_metrics(text[, language])`
    * Calcule plusieurs métriques de texte telles que sa longueur, son nombre de phrases, son nombre de mots, sa densité de ponctuation et sa lisibilité.


## [**`tx_ml.py`**][tx_ml.py]


* **`load`**`(`*`filename`*`)` : chargement du fichier CSV `filename` comme *dataframe* Pandas.
* **`save_as`**`(`*`data, filename`*`)` : sauvegarde le dataframe Pandas `data` dans le fichier CSV `filename`.
* **`tweets_3k_sample`**`(`*`raw_tweets`*`)` : sélection (aléatoire) dans `raw_tweets` d'un échantillon de 1500 tweets positifs et de 1500 tweets négatifs.
* **`tokenize_text`**`(`*`sent`*`)` : décomposition d'une phrase en mots à l'aide de la méthode `word_tokenize` de la librairie NLTK.
* **`filter_stop_words`**`(`*`word_list`*`)` : suppression des *stop words* dans la liste `word_list`.

* `lowercase_and_filter(word_list)`
    * Passage en minuscules des mots de `word_list` et filtrage de ceux commençant par `'@'`, `'http'` or `'#'`.

* `lemmatize_words(word_list)`
    * Lemmatise une liste de mots.

* `prepare_text_for_bow(text)`
    * Prépare `text` pour l'analyse *bag of words* avec lemmatisation (`Countvectorizer`, `Tf-idf`, `Word2Vec`).

* `prepare_text_for_bow_with_lemmatization(text)`
    * Prépare `text` pour l'analyse *bag of words* avec lemmatisation (`Countvectorizer`, `Tf-idf`, `Word2Vec`). 

* `prepare_text_for_deep_learning(text)`
    * Prépare `text` pour l'analyse *deep learning* (`USE`, `BERT`, etc.).

* `preprocess(raw_tweets)`
    * Pré-traite `raw_tweets` en applicant les fonctions spécialisées `prepare_text_for_bow`, `prepare_text_for_bow_with_lemmatization` et `prepare_text_for_deep_learning` pour produire les 3 colonnes 'sent_bow', 'sent_bow_lem', 'sent_deep_learn'

* `encode_cats(y)`
    * Encode les catégories `y` en valeurs numériques.

* `show_sparsity(matrix[, contrast='nearest'])`
    * Affiche un visuel de la *sparsité* d'une matrice.

* `tsne_kmeans_ari(features, cat_codes, cat_labels)`
    * Calcule le TSNE, détermine les clusters et calcule l'ARI entre les catégories réelles et les clusters.

* `match_class(y_clu, y_cla)`
    * Retourne la meilleure correspondance entre les classes réelles `y_cla` et les classes prédites `y_clu`.

* `show_tsne(cat_codes, cat_labels, X_tsne, clu_labels, ARI)`
    * Trace les diagrammes de dispersion de la représentation t-SNE des tweets en utilisant étiquettes des catégories réelles et celles des cluster.

* `gesim_simple_preprocess(bow_lem)`
    * Retourne la liste de listes de mots, résultat du pré-traitement de la liste de phrases `bow_lem` à l'aide de `gensim.simple_preprocess`

* `fit_word2vec(sents, [w2v_size, w2v_window, w2v_min_count, w2v_epochs)`
    * Retourne un modèle Word2Vec entraîné sur la liste de phrases `sents`.

* `fit_keras_tokenizer(sents[, maxlen])`
    * Segmente et complète la liste de phrases `sents` à la longueur `maxlen` (24 par défaut).

* `get_embedding_matrix(w2v_model, keras_tokenizer)`
    * Retourne la matrice d'incorporation de mots créée à l'aide d'un modèle gensim word2vec et d'un tokenizer Keras.

* `get_embedding_model(x_sents, w2v_model, keras_tokenizer, embedding_matrix[, maxlen])`
    * Retourne un modèle Keras construit à l'aide d'une matrice d'incorporation Word2Vec pré-entraîné.

* `encode_sentences_with_bert(sents, bert_tokenizer, max_length)`
    * Encode la liste de phrases `sents` en entrées compatibles BERT.

* `extract_bert_sentence_embeddings(model, model_type, sents, max_length, batch_size[, mode])`
    * Retourne les incorporations de la liste de phrases `sents` à l'aide d'un modèle BERT pré-entraîné.

* `extract_use_sentence_embeddings(sents[, batch_size])`
    * Retourne les incorporations USE (Universal Sentence Encoder) de la liste de phrases `sents`

## [**`pipeline.py`**][pipeline.py]

* `get_raw_data(name)`: This function returns raw data for a given name.
* `get_class_labels()`: This function returns a list of class labels.
* `get_class_label_name_map()`: This function returns a dictionary that maps class labels to their names.
* `preprocess_corpus(corpus, discard_words_function)`: This function preprocesses the given corpus by removing the stop words and other words provided by discard_words_function.
* `get_sents_class_labels(sents_index, class_labels)`: This function maps the sentences in sents_index to their respective class labels in class_labels.
* `tsne_kmeans_ari(features, cla_labels)`: This function performs t-SNE dimensionality reduction and K-means clustering on the given features and returns the resulting labels and the Adjusted Rand Index (ARI) score.
* `show_tsne(cla_labels, cla_names, X_tsne, clu_labels, ari, title)`: This function displays a 2D t-SNE plot of the given features, where each point is labeled with its true class label and its predicted cluster label.

Additionally, you have provided the following functions for different feature extraction methods:

* `tx_ml_bow_count(sents, cla_labels, name)`: This function uses CountVectorizer to extract features from the given sentences and performs clustering using t-SNE and K-means.
* `tx_ml_bow_tfidf(sents, cla_labels, name)`: This function uses TfidfVectorizer to extract features from the given sentences and performs clustering using t-SNE and K-means.
* `tx_ml_word2vec(sents, cla_labels, name)`: This function uses Word2Vec to embed the given sentences into a dense vector space and performs clustering using t-SNE and K-means.
* `extract_bert_sentence_embeddings(sentences)`: This function uses BERT to extract sentence embeddings from the given sentences.
* `extract_use_sentence_embeddings(sentences)`: This function uses Universal Sentence Encoder (USE) to extract sentence embeddings from the given sentences.

All of these functions take in a list of sentences sents and their corresponding class labels cla_labels, as well as a name parameter that specifies the type of data being used (e.g., 'product_name', 'description', etc.). Some of these functions also take in additional hyperparameters, which will need to be tuned based on the specific dataset and problem at hand.

