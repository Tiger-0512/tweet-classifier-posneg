import pandas as pd

import neologdn
import re
import emoji
import MeCab


# csvファイル読み込み
df = pd.read_csv('./tweet_extructor/final_data.csv')
# print(df['Text'][9049:9050])

# 日本語テキスト正規化
df['Text'] = df.apply(lambda x: neologdn.normalize(x['Text']), axis=1)

# URLテキスト削除
df['Text'] = df.apply(
    lambda x: re.sub(
        r'(http|https)://([-\w]+\.)+[-\w]+(/[-\w./?%&=]*)?',
        "",
        x['Text']
        ),
    axis=1)
# print(df['Text'][9049:9050])

#記号削除
df['Text'] = df.apply(
    lambda x: re.sub(
        r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+',
        "",
        x['Text']
        ),
    axis=1)

# 絵文字削除
def remove_emoji(src_str):
    return ''.join(c for c in src_str if c not in emoji.UNICODE_EMOJI)
df['Text'] = df.apply(lambda x: remove_emoji(x['Text']), axis=1)

# MeCab
tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/ipadic')
df['Mecab'] = df.apply(lambda x: tagger.parse(x['Text']), axis=1)
