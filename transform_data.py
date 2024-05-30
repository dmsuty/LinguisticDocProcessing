import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import pymorphy2
from tqdm import tqdm

nltk.download('punkt')

# Загрузка данных
# Предположим, что ваш DataFrame назван fdf и содержит колонку 'text'

# Инициализация Pymorphy2 для морфологического анализа
morph = pymorphy2.MorphAnalyzer()

# Функция для очистки текста
def preprocess_text(text):
    # Токенизация текста
    tokens = word_tokenize(text)
    # Морфологический анализ и лемматизация
    lemmatized_tokens = []
    for token in tqdm(tokens, desc="Processing tokens", leave=False):
        parsed_token = morph.parse(token)[0]
        lemma = parsed_token.normal_form
        grammatical_info = {'Lemma': lemma, 'Token': token, 'POS': parsed_token.tag.POS, 'Case': parsed_token.tag.case, 'Number': parsed_token.tag.number,
                            'Gender': parsed_token.tag.gender, 'Tense': parsed_token.tag.tense}
        lemmatized_tokens.append(grammatical_info)
    return lemmatized_tokens

fdf = pd.read_csv('raw_datasets/small_filtered_news.csv')

# Применение функции к колонке 'text' в DataFrame
tqdm.pandas(desc="Applying preprocessing")
fdf['clean_text'] = fdf['text'].progress_apply(preprocess_text)

# Вывод первых нескольких строк для проверки результата
print(fdf.head())

fdf.to_csv('processed_datasets/tokenized_small_news.csv')
