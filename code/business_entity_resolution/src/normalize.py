"""Text normalisation + tokenisation for business names / addresses.

Language-agnostic: works for US, India and France (and any other Latin-script country).
Non-Latin scripts (Devanagari, Telugu, ...) are kept as opaque tokens in names and dropped
from addresses.  No external lookups: only small static dictionaries defined here.
"""
import re
import unicodedata

import json, os

_NMAP_PATH = os.environ.get("BER_NATIVE_MAP", os.path.join(os.environ.get("BER_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "work")), "native_map.json"))
try:
    NATIVE_MAP = json.load(open(_NMAP_PATH, encoding="utf-8"))
except OSError:
    NATIVE_MAP = {}

_COMB = re.compile(r"[̀-ͯ]")
_NONWORD = re.compile(r"[^0-9a-zऀ-෿一-鿿 ]+")
_SPACES = re.compile(r"\s+")
_ADDRSEP = re.compile(r"[^0-9a-z\u0900-\u0dff\u4e00-\u9fff/\- ]+")

LEGAL = {
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation",
    "co", "company", "llp", "lp", "plc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "gmbh",
    "pty", "the", "and", "of", "de", "la", "le", "les", "du", "des", "et", "en", "un",
    "public", "ltda", "opc", "lp", "pllc", "pc", "ag", "nv", "bv", "com", "net", "org", "www",
    "http", "https", "formerly", "dba", "fka", "smt", "shri", "sri", "m", "s",
}
LEGAL_CANON = {
    "incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd",
    "private": "pvt",
}
# French legal forms measured frequent in test (SNC 14.8k, Cie 36k, Ets 27k, EI 14k records) and Ets = Etablissements.
# Opt-in (BER_FR_NORM=1): the leaderboard showed the model rebuilt with them did not beat the submitted configuration.
if os.environ.get("BER_FR_NORM", "0") == "1":
    LEGAL |= {"snc", "cie", "ets", "ei"}
    LEGAL_CANON.update({"etablissements": "ets", "etablissement": "ets"})

ABBR = {
    # english street words -> canonical short
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "avenu": "ave", "boulevard": "blvd", "bd": "blvd", "boulevar": "blvd", "drive": "dr",
    "lane": "ln", "court": "ct", "place": "pl", "square": "sq", "highway": "hwy",
    "parkway": "pkwy", "circle": "cir", "terrace": "ter", "trail": "trl", "way": "wy",
    "north": "n", "south": "s", "east": "e", "west": "w", "suite": "ste", "apartment": "apt",
    "floor": "fl", "building": "bldg", "cross": "cr", "main": "main", "nagar": "nagar",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "1st": "1", "2nd": "2", "3rd": "3", "4th": "4", "5th": "5", "6th": "6", "7th": "7",
    "8th": "8", "9th": "9", "10th": "10",
    # french
    "rue": "rue", "r": "rue", "chemin": "ch", "che": "ch", "impasse": "imp", "allee": "all",
    "place": "pl", "pl": "pl", "route": "rte", "rte": "rte", "passage": "pass", "quai": "qu",
    "cours": "crs", "faubourg": "fg", "fbg": "fg", "residence": "res", "zone": "z",
    "avenue": "ave", "bld": "blvd", "cite": "cite",
    # indian
    "colony": "col", "cly": "col", "marg": "marg", "sector": "sec", "phase": "ph",
    "opposite": "opp", "near": "near", "behind": "behind", "number": "no", "num": "no",
}
ADDR_DROP = {
    "no", "h", "hno", "door", "dno", "plot", "pmb", "null", "nr", "near", "opp", "behind",
    "flat", "house", "the", "and", "of", "de", "la", "le", "du", "des", "&",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "newhampshire": "nh",
    "newjersey": "nj", "newmexico": "nm", "newyork": "ny", "northcarolina": "nc",
    "northdakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhodeisland": "ri", "southcarolina": "sc", "southdakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "westvirginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
IN_STATES = {
    "andhrapradesh": "ap", "arunachalpradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachalpradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhyapradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamilnadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttarpradesh": "up", "uttarakhand": "uk", "westbengal": "wb", "delhi": "dl",
    "jammuandkashmir": "jk", "jammukashmir": "jk", "chandigarh": "ch", "puducherry": "py",
    "ladakh": "la",
}
STATE_MAP = {**US_STATES, **IN_STATES}
# multi-word states are joined without spaces before lookup
_MULTI = sorted([s for s in STATE_MAP if " " not in s and len(s) > 9], key=len, reverse=True)

_JOIN_STATES = [
    ("new hampshire", "newhampshire"), ("new jersey", "newjersey"), ("new mexico", "newmexico"),
    ("new york", "newyork"), ("north carolina", "northcarolina"),
    ("north dakota", "northdakota"), ("rhode island", "rhodeisland"),
    ("south carolina", "southcarolina"), ("south dakota", "southdakota"),
    ("west virginia", "westvirginia"), ("andhra pradesh", "andhrapradesh"),
    ("arunachal pradesh", "arunachalpradesh"), ("himachal pradesh", "himachalpradesh"),
    ("madhya pradesh", "madhyapradesh"), ("tamil nadu", "tamilnadu"),
    ("uttar pradesh", "uttarpradesh"), ("west bengal", "westbengal"),
    ("jammu and kashmir", "jammuandkashmir"),
]


def fold(s: str) -> str:
    """lowercase + strip Latin diacritics, keep other scripts intact."""
    s = unicodedata.normalize("NFKD", s.lower())
    return _COMB.sub("", s)


def norm_name(name: str):
    """Returns (tokens_without_legal, compact_string, legal_flag_tokens)."""
    s = fold(name).replace("&", " and ")
    s = s.replace(".com", " ").replace(".net", " ").replace(".org", " ")
    s = _NONWORD.sub(" ", s)
    toks = [NATIVE_MAP.get(t, t) for t in s.split()]
    core, legal = [], []
    for t in toks:
        t = LEGAL_CANON.get(t, t)
        if t in LEGAL:
            legal.append(t)
        else:
            core.append(t)
    if not core:  # name consisted only of "legal" words
        core, legal = legal, []
    return core, legal


_NUMSPLIT = re.compile(r"[-/]")


def norm_addr(addr: str):
    """Returns list of canonical address tokens (latin only) incl. number sub-tokens."""
    s = fold(addr)
    s = s.replace("<null>", " ").replace("&", " ")
    for a, b in _JOIN_STATES:
        if a in s:
            s = s.replace(a, b)
    # protect numbers with separators: '1-0/13/9', '36-'
    raw = []
    for t in _ADDRSEP.sub(" ", s).split():
        if any(c.isdigit() for c in t):
            raw.append(t)
        else:
            raw.extend(_NUMSPLIT.split(t))
    out = []
    for t in raw:
        if not t:
            continue
        if any(c.isdigit() for c in t):
            digs = _NUMSPLIT.split(t)
            digs = [d.strip("-/") for d in digs if d.strip("-/")]
            for d in digs:
                d = d.lstrip("0") or "0"
                out.append(d)
            if len(digs) > 1:
                j = "".join(d.lstrip("0") for d in digs)
                if j:
                    out.append(j)
            continue
        t = t.strip("-/")
        if not t:
            continue
        # non-latin script tokens are dropped from addresses
        if not t.isascii():
            continue
        t = STATE_MAP.get(t, t)
        t = ABBR.get(t, t)
        if t in ADDR_DROP:
            continue
        out.append(t)
    return out


def norm_record(name: str, addr: str):
    core, legal = norm_name(name)
    atoks = norm_addr(addr)
    return core, legal, atoks


def tokens_for_index(core, atoks):
    """String tokens used for hashed sparse index (namespaced)."""
    out = []
    for t in core:
        out.append("n:" + t)
        if len(t) >= 6:
            out.append("np:" + t[:5])
        if len(t) >= 5 and t.isascii():
            out.extend("g:" + t[k:k + 4] for k in range(len(t) - 3))
    if core:
        out.append("ne:" + "".join(core))
    # domain-like compact token (first 2-3 words concatenated) helps 'renaudmayesbuckley.com'
    if len(core) == 1:
        out.append("nc:" + core[0])
        out.append("nc3:" + core[0])
    if len(core) >= 2:
        out.append("nc:" + "".join(core[:2]))
    if len(core) >= 3:
        out.append("nc3:" + "".join(core[:3]))
    for t in atoks:
        out.append("a:" + t)
        if len(t) >= 6 and not t.isdigit():
            out.append("ap:" + t[:5])
    # composite name-token x address-token keys: "same name AND shares a (possibly common) address token" is rare & strong
    nts = [t for t in core if len(t) >= 3][:4]
    if nts and atoks:
        seen = []
        for t in atoks:
            if t not in seen and (len(t) >= 2 or (t.isdigit() and len(t) >= 2)):
                seen.append(t)
        if len(seen) > 12:
            seen = seen[:6] + seen[-6:]
        for nt in nts:
            for at in seen:
                out.append("nx:" + nt + "|" + at)
    # adjacent address tokens (number+street) are strong keys
    for i in range(len(atoks) - 1):
        if atoks[i] and atoks[i + 1] and (atoks[i][0].isdigit() or atoks[i + 1][0].isdigit()):
            out.append("ab:" + atoks[i] + "_" + atoks[i + 1])
    return out
