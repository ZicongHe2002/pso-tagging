from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import nltk

nltk.download('averaged_perceptron_tagger')


def parse_txt(filepath):
    sentences = []
    with open(filepath, 'r') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line:
                word, pos, chunk = line.split()
                sentence.append((word, pos, chunk))
            else:
                sentences.append(sentence)
                sentence = []
    return sentences
def parse_unlabeled_txt(filepath):
    sentences = []
    with open(filepath, 'r') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line:
                sentence.append(line)
            else:
                if sentence:  # If there was a sentence before the empty line, add it
                    sentences.append(sentence)
                    sentence = []
                sentences.append([''])  # Represent the empty line
    return sentences

train_sentences = parse_txt('train.txt')
test_sentences_words = parse_unlabeled_txt('unlabeled_test_test.txt')

# Feature Extraction
def features(sentence, index):
    is_tuple_format = isinstance(sentence[index], tuple)
    word = sentence[index][0] if is_tuple_format else sentence[index]

    # Handle empty words immediately
    if not word:
        return {}

    # Handle previous and next words based on the format
    prev_word = '' if index == 0 else (sentence[index - 1][0] if is_tuple_format else sentence[index - 1])
    next_word = '' if index == len(sentence) - 1 else (
        sentence[index + 1][0] if is_tuple_format else sentence[index + 1])

    return {
        'word': word,
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
        'is_all_lower': word.lower() == word,
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'prev_word': prev_word,
        'next_word': next_word,
        'has_hyphen': '-' in word,
        'is_numeric': word.isdigit(),
        'capitals_inside': word[1:].lower() != word[1:]
    }


def prepare_dataset(sentences):
    X, y = [], []
    for sentence in sentences:
        for index in range(len(sentence)):
            X.append(features(sentence, index))
            y.append(sentence[index][1])
    return X, y


X_train, y_train = prepare_dataset(train_sentences)

# Model Training
vectorizer = DictVectorizer()

'''
nb_classifier = make_pipeline(vectorizer, MultinomialNB())
nb_classifier.fit(X_train, y_train)

lr_classifier = make_pipeline(vectorizer, LogisticRegression(max_iter=10000))
lr_classifier.fit(X_train, y_train)
'''

svm_classifier = make_pipeline(vectorizer, SVC(kernel='linear'))
svm_classifier.fit(X_train, y_train)
def save_predictions(model, sentences):
    X_unlabeled = [features(s, idx) for s in sentences for idx in range(len(s))]

    predictions = model.predict(X_unlabeled)
    words = [w for s in sentences for w in s]

    with open('LLaMA.test.txt', 'a') as f:
        # f.write(f"Predictions using {name}:\n")
        for word, predicted_pos in zip(words, predictions):
            if word == '':  # This represents an empty line in the original text
                f.write("\n")
            else:
                f.write(f"{word} {predicted_pos}\n")

# Clear or create output.txt before saving any data
with open('LLaMA.test.txt', 'w') as f:
    f.write("")

# save_predictions(nb_classifier, test_sentences_words)
# save_predictions(lr_classifier, test_sentences_words)
save_predictions(svm_classifier, test_sentences_words)