from django.http import JsonResponse
from rest_framework.decorators import api_view
import MeCab
from wordcloud import WordCloud
import os
from collections import Counter
from django.conf import settings
import sqlite3
import pandas as pd
import unicodedata
import re
import pprint
from rich.console import Console
import base64
from sklearn.feature_extraction.text import TfidfVectorizer


# ================== テキスト正規化関連関数 ======================
def normalize(text):
    """
    テキストを正規化する関数
    - Unicode正規化
    - 数値の正規化
    - 小文字化
    - 改行、タブ、余分なスペースの削除
    """
    text = normalize_unicode(text)  # Unicodeの正規化
    text = normalize_number(text)   # 数字の正規化
    text = text.lower()             # 小文字化
    text = re.sub(r'\s+', '', text)  # 複数の空白を削除（空白をなくす）
    text = text.strip()             # 両端の空白文字を削除
    return text


def normalize_unicode(text, form='NFKC'):
    """Unicodeを正規化する関数"""
    return unicodedata.normalize(form, text)


def normalize_number(text):
    """テキスト内の数字を統一する関数 (例: １ → 1)"""
    return text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))

# ===============================================================


# ================= MeCab形態素解析(単語抽出) ========================
def parse_text_with_mecab(text, include_pos=None):
    """MeCabを使ってテキストを解析し、単語を抽出する"""
    # テキスト正規化
    normalized_text = normalize(text)
    
    parser = JpParser()
    tokens = parser.tokenize(normalized_text)
    
    # 空白を含む単語を除外
    words = [
        token['surface']
        for token in tokens
        if token['surface'] not in STOPWORDS and
           token['surface'].strip() and
           (include_pos is None or token['pos'] in include_pos)
    ]
    
    print(f"解析単語: {words}")  # デバッグ用
    return words

# ==============================================================


# ================ ワードクラウド生成 ========================
def generate_wordcloud(words, output_path):
    """単語リストからワードクラウドを生成して保存"""
    wordcloud = WordCloud(
        font_path=FONT_PATH,
        background_color="white",
        width=1000,
        height=600,
    ).generate(" ".join(words))
    wordcloud.to_file(output_path)

def encode_image_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# ==========================================================


# 定数の定義
STOPWORDS = ["する", "ある", "なる", "こと", "それ", "これ", "し", "の", "もの"]
EXCLUDE_POS = ['記号', '助詞', '助動詞', '接続詞']
FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"

# SQLite設定
DB_PATH = os.path.join(settings.BASE_DIR, 'db.sqlite3')  # SQLiteデータベースのパス
POLARITY_TABLE_NAME = "polarity"

def get_polarity_dic():
    """SQLiteから感情極性辞書を取得"""
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT headword, score FROM {POLARITY_TABLE_NAME}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# JpParserクラス
class JpParser:
    POS_DIC = {
        'BOS/EOS': 'EOS',
        '形容詞': 'ADJ',
        '副詞': 'ADV',
        '名詞': 'NOUN',
        '動詞': 'VERB',
        '助動詞': 'AUX',
        '助詞': 'PART',
        '連体詞': 'ADJ',
        '感動詞': 'INTJ',
        '接続詞': 'CONJ',
        '*': 'X',
    }

    def __init__(self):
        self._tagger = MeCab.Tagger("-Ochasen")
        self.pol_dic = get_polarity_dic()

    # =================== 文節ごと =======================
    def tokenize_bunsetsu(self, text):
        """助詞・助動詞を含めて文節単位で解析"""
        node = self._tagger.parseToNode(text)
        bunsetsu_list = []
        current_bunsetsu = []

        while node:
            feature = node.feature.split(',')
            # 文節の終了を明確にせず、すべてを収集
            if node.surface.strip():
                current_bunsetsu.append(node.surface)
            # EOSで文節終了
            if feature[0] in ['記号', '助詞', '助動詞']:
                if current_bunsetsu:
                    bunsetsu_list.append(''.join(current_bunsetsu))
                    current_bunsetsu = []
            node = node.next

        # 最後の文節を追加
        if current_bunsetsu:
            bunsetsu_list.append(''.join(current_bunsetsu))

        return bunsetsu_list


    def get_bunsetsu_score(self, bunsetsu):
        """SQLiteデータベースから文節のスコアを取得"""
        # pol_dicはpandasのDataFrameなので、文節に対するスコアを取得
        res = self.pol_dic[self.pol_dic['headword'] == bunsetsu]
        if not res.empty:
            score = res['score'].values[0]
        else:
            score = 0.0  # 存在しない場合はスコア0
        return score

    def apply_negation_and_emphasis(self, bunsetsu, score):
        """否定・強調語の影響を考慮したスコア調整"""
        if "ない" in bunsetsu or "じゃない" in bunsetsu:
            score *= -1
        if "とても" in bunsetsu or "すごく" in bunsetsu or "かなり" in bunsetsu:
            score *= 1.5
        return score


    def senti_analysis_by_bunsetu(self, text):
        """文節ごとに感情スコアを計算"""
        # 文節分割
        bunsetsu_list = self.tokenize_bunsetsu(text)
        print(f"Bunsetu List: {bunsetsu_list}")

        bunsetsu_scores = []

        for bunsetsu in bunsetsu_list:
            # bunsetsuのスコアをデータベースから取得
            score = self.get_bunsetsu_score(bunsetsu)
            score = self.apply_negation_and_emphasis(bunsetsu, score)
            bunsetsu_scores.append(score)

        # 全体スコアの計算
        overall_score = round(sum(bunsetsu_scores) / len(bunsetsu_scores), 2) if bunsetsu_scores else 0
        sentiment_label = "肯定的" if overall_score > 0 else "否定的" if overall_score < 0 else "中立"

        print(f"Overall Score: {overall_score}, Sentiment Label: {sentiment_label}")

        return {
            "bunsetsu_scores": dict(zip(bunsetsu_list, bunsetsu_scores)),
            "overall_score": overall_score,
            "label": sentiment_label
        }

    # =================== 単語ごと =======================
    def tokenize(self, sent):
        """テキストを形態素解析して単語ごとにトークン化"""
        node = self._tagger.parseToNode(sent)
        tokens = []
        while node:
            feature = node.feature.split(',')
            tokens.append({
                'surface': node.surface,
                'pos': feature[0],
                'base_form': feature[6] if len(feature) > 6 else '',
            })
            node = node.next
        return tokens


#======================== APIエンドポイント=================================
@api_view(['POST'])
def mecab_analyze(request):
    text = request.data.get('text', '').replace('\n', '').replace('\r', '')
    if not text:
        return JsonResponse({'error': 'No text provided'}, status=400)

    # ================== コメントデータ取得 ==================
    comments_by_age = request.data.get('comments_by_age', {})
    
    # コメントデータが期待通りの形式であることを確認
    print(f"Received Text: \n{text}")  # デバッグ用
    
    # 'comments_by_age' が適切に含まれているか確認し、データを取得
    if isinstance(comments_by_age, dict):
        # comments_by_age が辞書であることを確認
        for age_group, comments in comments_by_age.items():
            print(f"\n=== 年代: {age_group} ===")  # 各年齢グループ
            for comment in comments:
                print(f"コメント: {comment}")  # 各コメント
    else:
        # エラー処理
        return JsonResponse({'error': 'comments_by_age is not in the expected format or is empty'}, status=400)
    
    # ================== 感情分析 ==================
    parser = JpParser()
    sentiment_resultby_bunsetu = parser.senti_analysis_by_bunsetu(text)
    pprint.pprint(f"文節感情分析結果: {sentiment_resultby_bunsetu}")  # デバッグ用

    # ================== メインテキストの解析 ==================
    include_pos = ['名詞', '動詞', '形容詞', '副詞']
    words = parse_text_with_mecab(text, include_pos=include_pos)
    output_image_path = os.path.join(settings.BASE_DIR, 'static', 'wordcloud.png')
    generate_wordcloud(words, output_image_path)

    # ================== 上位10語の取得 ==================
    word_counts = Counter(words)
    top_10 = word_counts.most_common(10)
    print(f"トップ10: {top_10}")  # デバッグ用

    # TF-IDF分析
    corpus = [" ".join(words)]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    tfidf_scores = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
    sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    # ================== 年代別コメントの処理 ==================
    age_distribution = {}

    if isinstance(comments_by_age, dict):
        for age, comments in comments_by_age.items():
            if isinstance(comments, list):
                # コメントリストを単一のテキストとして結合して解析
                combined_text = " ".join(comments)
                age_words = parse_text_with_mecab(combined_text, include_pos=include_pos)
                age_word_counts = Counter(age_words)

                # 年代ごとのトップ10を取得
                top_10_for_age = age_word_counts.most_common(10)
                age_distribution[int(age)] = top_10_for_age

                # デバッグ用に結果を出力
                print(f"Age: {age}, Top 10: {top_10_for_age}")
            else:
                print(f"Invalid data for age {age}: Expected list, got {type(comments)}")  # 修正されたデバッグ用メッセージ
    else:
        print(f"Invalid comments_by_age data: Expected dict, got {type(comments_by_age)}")  # 修正されたデバッグ用メッセージ

    # ================== 5. 時系列での変化の可視化 ==================
    # （仮の例としてコメントの投稿順を時系列と見なす）
    time_series_data = []
    for age, comments in comments_by_age.items():
        for idx, comment in enumerate(comments):
            tokens = parse_text_with_mecab(comment, include_pos=include_pos)
            counter = Counter(tokens)
            time_series_data.append({
                "age": int(age),
                "index": idx,
                "top_words": counter.most_common(5)
            })

    # ================== 8. 関連するコメントのピックアップ ==================
    related_comments = {}
    for word, _ in top_10:
        related_comments[word] = []
        for age, comments in comments_by_age.items():
            for comment in comments:
                if word in comment:
                    related_comments[word].append(comment)
                if len(related_comments[word]) >= 3:
                    break
            if len(related_comments[word]) >= 3:
                break

    # ================== 9. コメントの要約文（簡易的に最も長いコメントを代表にする） ==================
    summaries_by_age = {}
    for age, comments in comments_by_age.items():
        if comments:
            summaries_by_age[int(age)] = max(comments, key=len)

    result = {
        "top_10": top_10,
        "wordcloud_image": 'static/wordcloud.png',
        "wordcloud_base64": encode_image_base64(output_image_path),
        "tfidf_top_10": sorted_tfidf,
        "age_distribution": age_distribution,
        "bunsetsu_scores": sentiment_resultby_bunsetu["bunsetsu_scores"],
        "sentiment_score": sentiment_resultby_bunsetu["overall_score"],
        "sentiment_label": sentiment_resultby_bunsetu["label"],
        "time_series_data": time_series_data,
        "related_comments": related_comments,
        "summaries_by_age": summaries_by_age,
    }
    return JsonResponse(result)
# ========================================================================