"""Task 8: house-number parsing fixes for number geometry (wordstats.number_geometry / cluster features).

Old: every all-digit address token is a house number ("8 mai 1945" in "allee du 8 mai 1945" -> 8 and 1945, so two
different houses on that street look 'equal'), and glued suffixes ("12bis", "12b") are not digits -> number 'missing'.
mode "fr" : drop day(+er) + month (+ year) tokens of date-named streets (French month names); 12bis/12ter/12quater -> 12
mode "gen": "fr" + a single glued letter (12b -> 12)
Pattern-based (language words), not country-based.  apply(mode) patches wordstats._nums in place.
"""
import re
import wordstats

MONTHS = {"janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre", "novembre", "decembre"}
_DAY = re.compile(r"\d{1,2}(er)?")
_YEAR = re.compile(r"1[5-9]\d\d|20\d\d")
_SUF = {"fr": re.compile(r"(\d+)(bis|ter|quater)?"), "gen": re.compile(r"(\d+)(bis|ter|quater|[a-z])?")}


def make_nums(mode):
    suf = _SUF[mode]

    def _nums(s):
        toks = s.split(); skip = set()
        for i, t in enumerate(toks):
            if t in MONTHS:
                if i > 0 and _DAY.fullmatch(toks[i - 1]):
                    skip.add(i - 1)
                if i + 1 < len(toks) and _YEAR.fullmatch(toks[i + 1]):
                    skip.add(i + 1)
        out = []
        for i, t in enumerate(toks):
            if i in skip:
                continue
            m = suf.fullmatch(t)
            if m and len(m.group(1)) <= 9:
                out.append(int(m.group(1)))
        return out
    return _nums


def apply(mode):
    if mode:
        wordstats._nums = make_nums(mode)
