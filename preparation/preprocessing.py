import neologdn
import re
import emoji
import MeCab


# 前処理全般を行う関数
def pre_processing(sentence):
    # 日本語テキスト正規化
    sentence = neologdn.normalize(sentence)

    # URLテキスト削除
    sentence = re.sub(r'(http|https)://([-\w]+\.)+[-\w]+(/[-\w./?%&=]*)?',
                "",
                sentence)
    #記号削除
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+',
                "",
                sentence)
    # 絵文字削除
    sentence = remove_emoji(sentence)

    # MeCab
    tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/ipadic')
    sentence = tagger.parse(sentence)
    return sentence


def remove_emoji(src_str):
    return ''.join(c for c in src_str if c not in emoji.UNICODE_EMOJI)
