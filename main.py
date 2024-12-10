import os, re
import requests
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Инициализация ресурсов NLTK
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
load_dotenv()
stop_words = stopwords.words("russian")


def get_vk_user_info(user_id, access_token):
    url = 'https://api.vk.com/method/users.get'
    params = {
        'user_ids': user_id,
        'access_token': access_token,
        'v': '5.131'
    }
    response = requests.get(url, params=params)
    return response.json()

def get_groups_info(user_id, token):
    '''
    Получить список групп с описанием (desc)
    '''
    user_info = get_vk_user_info(user_id, token)
    url = 'https://api.vk.com/method/users.getSubscriptions'
    params = {
        'user_id': user_info['response'][0]['id'],
        'access_token': token,
        'extended': 1,
        'fields': 'description',
        'count': 100,
        'v': '5.131'
    }
    response = requests.get(url, params=params)
    return create_groups_dict(response.json())

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def create_groups_dict(data):
    groups = {}
    for group in data['response']['items']:
        name = group.get('name', 'Без названия')  # Если нет названия, будет возвращено 'Без названия'
        description = group.get('description', 'Нет описания')
        groups[name] = description
    return groups

class Model:
    def __init__(self, groups):
        self.groups = groups
        self.descriptions = list(self.groups.values())

    def preprocess_text(self):
        processed_texts = []
        for desc in self.descriptions:
            cleaned_text = self.clean_text(desc)
            tokens = self.tokenization(cleaned_text)
            lemmatized_tokens = self.lemmatize(tokens)
            processed_texts.append(lemmatized_tokens)
        return processed_texts

    def clean_text(self, text):
        # Переводим текст в нижний регистр
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)  # Удаление HTML тегов
        text = re.sub(r"[^a-zA-Zа-яА-Я\s]", "", text)  # Удаление всех символов, кроме букв
        text = re.sub(r"\s+", " ", text).strip()  # Удаление лишних пробелов
        return text

    def tokenization(self, text):
        return word_tokenize(text)

    def lemmatize(self, words):
        return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]

    def thematic_modeling(self, texts, num_topics=5):

        # Создание словаря и корпуса
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Обучение LDA модели
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

        # Печать тем
        print("Топ-5 тем:")
        for idx, topic in lda_model.print_topics(num_words=5):
            print(f"Тема {idx}: {topic}")

        return lda_model, corpus, dictionary

    def extract_keywords(self, texts):
        # Используем TF-IDF
        joined_texts = [" ".join(text) for text in texts]

        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words=stop_words)
        tfidf_matrix = tfidf_vectorizer.fit_transform(joined_texts)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1

        # Сортируем ключевые слова
        keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)

        print("Топ ключевых слов:")
        for word, score in keywords[:10]:
            print(f"{word}: {score}")
        # Визуализация TF-IDF
        self.plot_keywords(keywords)

    def plot_keywords(self, keywords, top_n=10):
        top_keywords = keywords[:top_n]
        words, scores = zip(*top_keywords)

        plt.figure(figsize=(10, 6))
        plt.bar(words, scores, color='blue')
        plt.xticks(rotation=45)
        plt.title("Топ ключевых слов")
        plt.xlabel("Слова")
        plt.ylabel("Вес (TF-IDF)")
        plt.show()





# Получение токена и user_id из .env
user_id = os.getenv('USER_ID')
token = os.getenv('TOKEN')

# Получаем данные групп
groups = get_groups_info(user_id, token)

model = Model(groups)
processed_texts = model.preprocess_text()

lda_model, corpus, dictionary = model.thematic_modeling(processed_texts)

# Извлечение ключевых слов
model.extract_keywords(processed_texts)


# Тестирование
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f"Coherence Score: {coherence_lda}")



