import unicodedata
from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator
QUOTE_MAP = str.maketrans({
    "ʻ": "'",
    "‘": "'",
    "’": "'",
    "‛": "'",
    "`": "'",
    "´": "'",
    "ʼ": "'",
    "＇": "'",
    "❛": "'",
    "❜": "'",})

import regex
def tokenize(text):
    result = []
    for part in regex.split(r"(?w)\b", text):
        token = part.strip()
        if token:
            result.append(token)
    return result
import regex
import uroman as ur


# 只初始化一次；避免每次调用都重新加载 uroman 数据
UROMAN = ur.Uroman()

# Unicode 字符属性
LETTER_RE = regex.compile(r"\A\p{Letter}\Z")
MARK_RE = regex.compile(r"\A\p{Mark}\Z")
LATIN_RE = regex.compile(r"\A\p{Script_Extensions=Latin}\Z")
GRAPHEME_RE = regex.compile(r"\X")


def is_latin_letter_cluster(cluster: str) -> bool:
    """
    判断一个完整字素是否属于拉丁字母。

    支持：
      é             单码位拉丁字母
      e + ◌́         拉丁字母 + 组合附加符
      ā, ø, œ, ǅ    扩展拉丁字母
    """
    has_latin_letter = False

    for char in cluster:
        if MARK_RE.fullmatch(char):
            # 组合重音符等跟随拉丁字母一起保留
            continue

        if LETTER_RE.fullmatch(char) and LATIN_RE.fullmatch(char):
            has_latin_letter = True
            continue

        # 含有非拉丁内容，就不作为受保护的拉丁字素
        return False

    return has_latin_letter


def uroman_except_latin(text: str, lcode: str | None = None) -> str:
    """
    仅对非拉丁字母部分使用 uroman；
    所有拉丁字母，包括扩展拉丁字母，保持原样。

    lcode 可传 ISO 639-3 语言代码，例如：
      rus、ukr、ell、fas
    """
    result: list[str] = []
    pending: list[str] = []

    def flush_non_latin() -> None:
        if not pending:
            return

        segment = "".join(pending)

        if lcode is None:
            romanized = UROMAN.romanize_string(segment)
        else:
            romanized = UROMAN.romanize_string(segment, lcode=lcode)

        result.append(romanized)
        pending.clear()

    # \X 按完整字素切分，避免把 e + 组合重音符拆开
    for cluster in GRAPHEME_RE.findall(text):
        if is_latin_letter_cluster(cluster):
            flush_non_latin()
            result.append(cluster)
        else:
            pending.append(cluster)

    flush_non_latin()
    return "".join(result)






t1 = Transliterator("uzb", Script.LATIN, Script.COMMON_TURKIC)
t2 = Transliterator("uzb", Script.CYRILLIC, Script.COMMON_TURKIC)
t3 = Transliterator("uig", Script.PERSO_ARABIC, Script.COMMON_TURKIC)
t4 = Transliterator("azb", Script.PERSO_ARABIC, Script.COMMON_TURKIC)

file_path = "rawwikitxt/uz_20260101.txt"
print(file_path)
linecount = 0

pure_uroman_list = []
common_turkic_list = []

with open(file_path, "r", encoding="utf-8") as rawwikitxt_f:
    for each_rawwikitxt_line in rawwikitxt_f:
        linecount += 1
        if linecount % 100000 == 0:
            print(linecount)
        each_rawwikitxt_line = unicodedata.normalize("NFC", each_rawwikitxt_line)
        each_rawwikitxt_line = each_rawwikitxt_line.rstrip("\r\n")
        each_rawwikitxt_line = each_rawwikitxt_line.translate(QUOTE_MAP)

        each_pureuroman_line = UROMAN.romanize_string(each_rawwikitxt_line)
        each_pureuroman_line = unicodedata.normalize("NFC", each_pureuroman_line)
        pure_uroman_list.append(each_pureuroman_line)

        t1each_commonturk_line = t1.transliterate(each_rawwikitxt_line)
        t2each_commonturk_line = t2.transliterate(t1each_commonturk_line)
        t3each_commonturk_line = t3.transliterate(t2each_commonturk_line)
        t4each_commonturk_line = t4.transliterate(t3each_commonturk_line)
        t4each_commonturk_line = unicodedata.normalize("NFC", t4each_commonturk_line)
        uroman_t4each_commonturk_line = uroman_except_latin(t4each_commonturk_line)
        uroman_t4each_commonturk_line = unicodedata.normalize("NFC", uroman_t4each_commonturk_line)
        common_turkic_list.append(uroman_t4each_commonturk_line)

with open("uroman_uz.txt", 'w', encoding='utf-8') as uromanf:
    for each_pureroman in pure_uroman_list:
        uromanf.write(' '.join(tokenize(each_pureroman))+'\n')

with open("comturk_uz.txt", 'w', encoding='utf-8') as comturkf:
    for uroman_each_commonturk in common_turkic_list:
        comturkf.write(' '.join(tokenize(uroman_each_commonturk))+'\n')

