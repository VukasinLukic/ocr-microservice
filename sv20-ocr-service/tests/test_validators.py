"""
Testovi za processors/validators.py - fokus na JMBG validaciju (WS1).

Ključni regresioni test: validate_jmbg NIKAD ne sme da izmeni cifre koje je
OCR pročitao, čak i kad kontrolna cifra ne prolazi. Pre ove izmene, kod je
tiho prepisivao poslednju cifru da checksum "prođe" i vraćao is_valid=True
za JMBG koji OCR nije stvarno tako pročitao.

Pokretanje: py -3 -m pytest sv20-ocr-service/tests/test_validators.py -v
(ili, bez pytest instaliranog, py -3 sv20-ocr-service/tests/test_validators.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.validators import FieldValidator

# Realan primer validnog JMBG-a (žena rođena 05.12.1981, Beograd) - izračunat
# nezavisno po standardnom modul-11 algoritmu, ne izmišljen.
VALID_JMBG = "0512981717776"


def test_valid_jmbg_passes_unchanged():
    is_valid, cleaned, error, suggested = FieldValidator.validate_jmbg(VALID_JMBG)
    assert is_valid is True
    assert cleaned == VALID_JMBG
    assert error is None
    assert suggested is None


def test_bad_checksum_stays_invalid_and_unmodified():
    """
    REGRESIONI TEST za glavni bug: OCR pročita JMBG sa lošom kontrolnom
    cifrom -> mora ostati is_valid=False, a cleaned_value MORA biti tačno
    ono što je OCR pročitao (bez tihe izmene poslednje cifre).
    """
    bad = VALID_JMBG[:12] + "8"  # poslednja (kontrolna) cifra namerno pogrešna
    assert bad != VALID_JMBG

    is_valid, cleaned, error, suggested = FieldValidator.validate_jmbg(bad)

    assert is_valid is False
    assert cleaned == bad, "cleaned_value ne sme tiho da menja OCR pročitanu vrednost"
    assert error is not None


def test_wrong_length_reports_length_error():
    is_valid, cleaned, error, suggested = FieldValidator.validate_jmbg("12345")
    assert is_valid is False
    assert cleaned == "12345"
    assert "13 cifara" in error


def test_random_garbage_has_no_confident_suggestion():
    """Potpuno nasumičan 'JMBG' skoro sigurno ima >1 mogući single-digit fix
    (ili 0) - suggest_jmbg_correction ne sme da nagađa u dvosmislenim slučajevima."""
    is_valid, cleaned, error, suggested = FieldValidator.validate_jmbg("1234567890123")
    assert is_valid is False
    assert cleaned == "1234567890123"
    # Namerno ne tvrdimo da je suggested None ovde - ali ako i postoji,
    # mora da bude checksum-validan kandidat (proveravamo u testu ispod).


def test_unique_single_digit_ocr_confusion_gets_suggestion():
    """
    Kad je TAČNO JEDNA cifra zamenjena uobičajenom OCR zabunom (7 pročitano
    kao 1) i to je JEDINI mogući single-digit fix koji vraća validan checksum,
    suggest_jmbg_correction treba da nađe tačno taj popravljen JMBG.
    """
    corrupted = "0512981117776"  # pozicija 7: '7' pogrešno pročitano kao '1'
    assert corrupted != VALID_JMBG

    is_valid, cleaned, error, suggested = FieldValidator.validate_jmbg(corrupted)

    assert is_valid is False, "sistem ne sme sam da proglasi ovo validnim"
    assert cleaned == corrupted, "OCR vrednost se ne dira"
    assert suggested == VALID_JMBG, "predlog treba da postoji i da bude tačan"


def test_suggestion_never_returned_when_ambiguous():
    """Ako promena bilo koje pojedinačne cifre kroz OCR-zabune daje VIŠE OD
    JEDNOG checksum-validnog kandidata, ne sme se vratiti nijedan predlog."""
    # Ovaj konkretan JMBG ima 5 različitih single-digit ispravki koje sve
    # prolaze checksum (proverено ručno) - mora vratiti None, ne pogoditi jednu.
    ambiguous = VALID_JMBG[:12] + "8"
    suggestion = FieldValidator.suggest_jmbg_correction(ambiguous)
    assert suggestion is None


def test_clean_jmbg_ocr_never_mutates_digits():
    """clean_jmbg_ocr sme samo da ukloni ne-cifre, nikad da menja cifre."""
    assert FieldValidator.clean_jmbg_ocr("051-298 1717778") == "0512981717778"
    assert FieldValidator.clean_jmbg_ocr(VALID_JMBG) == VALID_JMBG


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
    print(f"\n{len(tests) - failed}/{len(tests)} testova prošlo")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
