# Overnight research log (read-only analyses; no pipeline changes)

Started 2026-09-27 ~05:00 IST. Each finding lists the data it came from.

## R1. Sibling-business vocabulary per country (test sample, 20k S1 per country, current model s5)
Records at a *different* house number than the S1's supported number; words they add to the S1 name.

| country | most frequent added words | share unknown to train vocabulary | model acceptance |
|---|---|---|---|
| US | partners, holdings, group, metro, summit, highland, north, southside, coastal, downtown, west, westgate, greater, central, midtown, east, harbor, eastgate, riverside, uptown, northside, valley, lakeside, south | 20% (mostly person-name typos) | branch words 0%; noise words center 13%, services 12%, partners 3% |
| India | enterprises, industries, exports, ventures, infratech, group, holdings, overseas, solutions, technologies, trading, engineering, constructions, foods | 16% | branch words 0% |
| **France** | france, **groupe, developpement, participations**, distribution, international, holding, club, **snc**, ecole, comite, amicale, fils, sportive, maison, cie, union, amis, college, federation | **66%** | **groupe 21%, france 18%, developpement 11%, participations 11%, snc 31%** |

**Conclusion:** US/India sibling words are fully known and already rejected. France's sibling words are French equivalents
(groupe≈group, holding≈holdings, developpement, participations, distribution, international) that the model has never
seen, so it accepts a share of them. Augmenting train with *English* branch words will not fix France by itself.
Note `snc` is a French legal form (société en nom collectif) and should probably be treated as a legal suffix like `sarl`.
