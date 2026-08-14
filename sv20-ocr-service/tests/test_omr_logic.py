"""
Testovi za processors/omr_logic.py - fokus na statističku (ne izmišljenu)
confidence za OMR polja (WS2).

Pre ove izmene, confidence je delimično dolazila iz fiksnih konstanti
(0.6/0.7/0.9/1.0, MIN_SCORE_THRESHOLD=0.10, FILL_THRESHOLD=0.15). Sad se
računa preko `_relative_margin_confidence` - razmernog (ratio) marginа
između najjače opcije i reference (druga najjača za single-select, najslabija
za multi-select), IZRAČUNATOG UVEK UNUTAR JEDNOG POLJA.

VAŽNO - zašto NIJE korišćena medijana/MAD preko VIŠE polja na strani (prva
verzija ovog fixa): validacija protiv stvarnog skeniranog ŠV-20 obrasca
(primer.pdf, /api/debug/omr-rois) je pokazala da različita polja imaju
DRASTIČNO različitu "prirodnu" količinu odštampanog mastila (npr. polje sa
dugim tekstualnim opcijama nasuprot polju sa dva mala kružića) - mešanje tih
skala u jednu statistiku je davalo lažne negative na stvarnim podacima iako
su sintetički testovi prolazili. `test_real_scan_regression_*` testovi ispod
koriste stvarne fill/edge/combined brojeve izmerene na primer.pdf da spreče
ponavljanje te greške.

Pokretanje: py -3 -m pytest sv20-ocr-service/tests/test_omr_logic.py -v
"""

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.omr_logic import OMRProcessor


def _make_option_field(marked_index, num_options=4, width=400, height=200):
    """
    Napravi sintetičku sliku sa `num_options` odštampanih brojeva raspoređenih
    horizontalno, gde je opcija na `marked_index` "zaokružena" (nacrtana
    elipsa oko nje) - simulira ručno zaokruživanje na ŠV-20 obrascu.
    """
    img = np.ones((height, width), dtype=np.uint8) * 255
    option_w = width // num_options
    options = []

    for i in range(num_options):
        cx = i * option_w + option_w // 2
        cy = height // 2
        cv2.putText(img, str(i + 1), (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        if i == marked_index:
            cv2.ellipse(img, (cx, cy), (option_w // 2 - 5, height // 2 - 10),
                        0, 0, 360, (0, 0, 0), 3)

        options.append({
            'label': str(i + 1),
            'value': f"Opcija {i + 1}",
            'x': i * option_w,
            'y': 0,
            'width': option_w,
            'height': height,
        })

    return img, options


def test_clearly_marked_option_gets_high_confidence():
    processor = OMRProcessor()
    img, options = _make_option_field(marked_index=2, num_options=4)

    result = processor.detect_marked_option(img, options)

    assert result['value'] == 'Opcija 3'
    assert result['confidence'] > 0.5


def test_nothing_marked_returns_none_not_a_fake_number():
    """Kad ništa nije zaokruženo, mora vratiti value=None (i nisku confidence),
    ne izmišljenih 0.5/0.7 kao ranije."""
    processor = OMRProcessor()
    img, options = _make_option_field(marked_index=-1, num_options=4)  # ništa zaokruženo

    result = processor.detect_marked_option(img, options)

    assert result['value'] is None
    assert result['confidence'] <= 0.5


def test_confidence_is_a_continuous_function_of_the_signal():
    """
    Regresioni test za glavni bug: stari kod je vraćao "okrugle" fiksne
    vrednosti (0.6, 0.7, 0.9, 1.0) NEZAVISNO od stvarnog signala. Nova
    confidence (_relative_margin_confidence) mora biti STROGO RASTUĆA
    funkcija razlike između score-a i reference - svaka jačina signala daje
    svoju, drugačiju vrednost, ne fiksnu konstantu iz malog skupa brojeva.
    """
    processor = OMRProcessor()
    reference = 0.05

    confidences = [
        processor._relative_margin_confidence(score, reference)
        for score in [0.06, 0.08, 0.10, 0.20, 0.40, 0.80]
    ]

    for a, b in zip(confidences, confidences[1:]):
        assert b > a, f"confidence mora strogo rasti sa signalom: {confidences}"

    assert len(set(round(c, 4) for c in confidences)) == len(confidences), \
        "svaka različita jačina signala mora dati različitu confidence, ne istu konstantu"


def test_multi_select_independent_per_option():
    processor = OMRProcessor()
    img = np.ones((100, 400), dtype=np.uint8) * 255
    options = []
    for i in range(4):
        cx = i * 100 + 50
        cv2.putText(img, str(i + 1), (cx - 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        options.append({'label': str(i + 1), 'value': f"Opcija {i + 1}", 'x': i * 100, 'y': 0, 'width': 100, 'height': 100})

    # Zaokruži opcije 1 i 3 (indeksi 0 i 2)
    cv2.ellipse(img, (50, 50), (40, 40), 0, 0, 360, (0, 0, 0), 3)
    cv2.ellipse(img, (250, 50), (40, 40), 0, 0, 360, (0, 0, 0), 3)

    result = processor.detect_multi_select(img, options)

    assert set(result['values']) == {'Opcija 1', 'Opcija 3'}


def _decide_single_from_raw(processor, combined_scores, labels=None):
    """Helper: napravi option_scores rečnike iz sirovih combined_score brojeva
    (koristi se za regresione testove sa stvarnim, ranije izmerenim vrednostima)."""
    labels = labels or [str(i + 1) for i in range(len(combined_scores))]
    option_scores = [
        {'label': lbl, 'value': lbl, 'fill_ratio': cs, 'edge_score': cs, 'combined_score': cs}
        for lbl, cs in zip(labels, combined_scores)
    ]
    return processor._decide_single(option_scores)


def test_real_scan_regression_two_option_field_with_clear_winner():
    """
    Stvarni combined_score brojevi izmereni preko /api/debug/omr-rois na
    primer.pdf (polje 'nacin_finansiranja', 2 opcije: 'Samofinansiranje' je
    bila jasno zaokružena). Prva verzija statističke confidence (medijana/MAD
    preko cele strane) je ovde davala confidence ~0.06 i vraćala None -
    lažni negativ na stvarnom, jasno zaokruženom polju. Ratio-pristup mora
    ispravno prepoznati pobednika.
    """
    processor = OMRProcessor()
    result = _decide_single_from_raw(processor, [0.0577, 0.1996], labels=['1', '2'])

    assert result['value'] == '2'
    assert result['confidence'] > 0.5


def test_real_scan_regression_ambiguous_field_stays_unmarked():
    """
    Stvarni brojevi za polje 'pol' (2 opcije) sa primer.pdf, gde su obe
    opcije imale skoro identičan combined_score (0.2469 vs 0.2503) - ili
    zaokruživanje nije jasno vidljivo, ili ROI koordinate nisu tačno
    poravnate sa krugom. U oba slučaja, sistem NE SME da izmisli pobednika -
    is ovo mora ostati "nedovoljno pouzdano", ne nasumično 70% kao pre.
    """
    processor = OMRProcessor()
    result = _decide_single_from_raw(processor, [0.2469, 0.2503], labels=['M', 'Z'])

    assert result['value'] is None
    assert result['confidence'] < 0.2


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} testova prošlo")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
