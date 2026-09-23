"""Hand labels for lines.json, from looking at every frame on the contact sheets
(line_sheets.py). 1 = a caption of what someone says, 0 = text that is in the scene or a
non-dialogue overlay (watermark, phone screen, wall scroll, book page, window reflection,
product box, character intro card, title card, reference sheet), x = cannot be labelled
from the frame (calligraphy poem overlays of the narration, pseudo-English subtitle
fragments, a fragment that may be part of either).

auto classes (lines.py) were accepted after a spot check: 'dialogue' (sim >= 0.35 to a
scripted line) 25/25 were captions; 'latin_upper' (Latin-only, upper 60% of frame) were all
the vertical strip on the window frame, the intro cards or a title card on the sheets; the
bottom-centre 'check' lines were NOT accepted wholesale -- only the 50 that were looked at
(48 captions, a phone label, one unclear) are used.
"""
X = "x"
OTHER = {  # the 249 'check' lines outside the bottom-centre band, every one looked at
    **{i: 0 for i in [2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                      30, 31, 32, 36, 51, 58, 59, 60]},
    16: X,
    61: 0, 69: X, 79: 1, 83: 0, 90: 1, 92: 1, 98: 0, 107: 1, 108: 1, 112: 0, 117: 1,
    **{i: 0 for i in [118, 119, 120, 121]}, **{i: 1 for i in [122, 123, 124]},
    **{i: 0 for i in [143, 144, 145, 146, 147, 148, 149]},          # character intro card
    **{i: 0 for i in [156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167]},  # phone
    168: 0,                                                          # print on the hoodie
    **{i: 0 for i in [177, 178, 179]}, 183: 1, 191: 0, 195: 0, 196: 0,
    **{i: 0 for i in [199, 200, 201, 203, 204]},
    **{i: 0 for i in [213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224]},
    **{i: 0 for i in [229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 242, 243, 244, 245, 246]},
    250: 0, 251: 0, 263: 0, 317: 0, 318: 0, 322: X, 323: 0,
    **{i: 0 for i in [327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342]},
    **{i: 0 for i in [354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364]},  # opening title block
    372: 0, **{i: X for i in [376, 377, 378, 379, 380, 381]},       # big top-centre text, name card or line
    385: 0, 386: 0, **{i: 0 for i in [387, 388, 389, 390]}, 396: X, 397: X, 401: 1,
    **{i: 0 for i in [406, 407, 408, 409, 429, 430, 431, 432, 433, 434, 435, 436, 437]},
    457: 0, 460: 0, 461: 0,                                          # the shop sign, not the caption
    **{i: 0 for i in [464, 465, 466, 467, 468, 469, 470, 471, 481]},
    **{i: 1 for i in [502, 503, 504, 505, 506, 517, 518, 519]},
    **{i: 0 for i in [527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]},
    **{i: 0 for i in [554, 555, 556, 557, 558, 559, 560, 565, 568]},
    600: X, 608: 0, 611: X, **{i: 0 for i in range(624, 632)},
    **{i: 0 for i in [633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643]}, 644: X, 645: X,
    651: 1, 652: 1, 654: 0, 655: 0, **{i: 0 for i in [656, 657, 658, 659, 660, 661]},
}
BOTTOM_SAMPLE = {i: 1 for i in [46, 62, 68, 81, 84, 85, 110, 115, 172, 182, 208, 212, 253, 265, 268, 271,
                                279, 291, 293, 307, 319, 345, 352, 367, 395, 400, 405, 419, 422, 427,
                                442, 450, 452, 456, 479, 490, 494, 498, 514, 563, 576, 588, 602, 604,
                                619, 648, 653, 673]}
BOTTOM_SAMPLE.update({218: 0, 663: X})
