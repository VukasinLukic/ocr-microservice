"""
SV-20 OCR Mikroservis - Validators
Validacija ekstraktovanih OCR vrednosti prema pravilima dokumentacije.
Podrzava: JMBG, datum, broj indeksa, i druge formate.
"""

import re
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class FieldValidator:
    """
    Validacija ekstraktovanih OCR vrednosti prema pravilima dokumentacije.
    """

    JMBG_LENGTH = 13
    INDEX_PATTERN = r"^\d{4}/\d{4}$"
    SCHOOL_YEAR_PATTERN = r"^\d{4}/\d{4}$"
    DATE_PATTERNS = [
        (r"^\d{2}\.\d{2}\.\d{4}$", "%d.%m.%Y"),
        (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
        (r"^\d{2}\.\d{2}\.\d{2}$", "%d.%m.%y"),
    ]

    @staticmethod
    def clean_year_ocr(value: str) -> str:
        """
        SMART CLEANING za godine u formatu 20XX ili 19XX.

        OCR greške:
        - "2025" -> "2925"
        - "2006" -> "12101006" (datum prepoznat kao godina)

        Strategija:
        1. Traži 4 uzastopne cifre koje počinju sa "20" ili "19"
        2. Ako ne pronađe, uzmi poslednje 4 cifre
        3. Ako počinje sa "29", zameni sa "20"

        Args:
            value: OCR tekst (može biti "2925", "12101006", "25", itd.)

        Returns:
            Očišćena godina u formatu 20XX ili 19XX
        """
        # Izvuci sve cifre
        digits = re.sub(r'\D', '', value)

        if not digits:
            return value

        # PRIORITET: Traži 4 uzastopne cifre koje počinju sa "20" ili "19"
        if len(digits) >= 4:
            for i in range(len(digits) - 3):
                candidate = digits[i:i+4]
                if candidate.startswith(('20', '19')):
                    # Pronađena validna godina!
                    return candidate

        # Specijalni slučaj: 8 cifara = možda datum u formatu DDMMYYYY
        # OCR može pročitati "12.10.2006" kao "12101006"
        if len(digits) == 8:
            # Uzmi poslednje 4 cifre (YYYY)
            year = digits[4:]
            # Ali ako ne počinje sa 19/20, probaj da dodaš "20" ispred poslednje 2
            if not year.startswith(('19', '20')):
                # Možda je OCR promašio prvu cifru godine
                # "12101006" -> godina bi trebalo biti "2006", ali OCR čita "1006"
                # Heuristika: ako je godina < 100, dodaj "20" ispred
                year_int = int(year)
                if year_int < 100:
                    # "06" -> "2006"
                    return "20" + year[-2:]
                elif 100 <= year_int < 1940:
                    # "1006" -> verovatno OCR greška, probaj "2006"
                    return "20" + year[-2:]
            return year

        # Ako ima više od 4 cifre (i nije pronađena "20XX" ili "19XX"), uzmi POSLEDNJE 4
        if len(digits) > 4:
            digits = digits[-4:]

        # Ako ima manje od 4, a više od 2, uzmi POSLEDNJE 2 i dodaj "20"
        if 2 <= len(digits) < 4:
            digits = "20" + digits[-2:]

        # Ako je tačno 4 cifre
        if len(digits) == 4:
            # Proveri da li počinje sa "29" (OCR greška za "20")
            if digits.startswith("29"):
                digits = "20" + digits[2:]  # 2925 -> 2025

        return digits

    @staticmethod
    def clean_date_ocr(value: str) -> str:
        """
        SMART CLEANING za datume.

        Primer: "У БЕОГРАДУ, 1.7. 2025. год" -> "1.7.2025"

        Strategija:
        - Pretraži regex pattern za DD.MM.YYYY ili D.M.YYYY
        - Izvuci i normalizuj
        """
        # Tražimo pattern: 1-2 cifre . 1-2 cifre . 4 cifre
        pattern = r'(\d{1,2})\.(\d{1,2})\.?\s*(\d{4})'
        match = re.search(pattern, value)

        if match:
            day = match.group(1).zfill(2)    # Dodaj 0 ispred ako treba (1 -> 01)
            month = match.group(2).zfill(2)
            year = match.group(3)
            return f"{day}.{month}.{year}"

        # Ako nema match, vrati original
        return value

    # Standardne OCR zabune cifara (vizuelno slična, česta pogrešna čitanja).
    # Koristi se SAMO za predlog ispravke JMBG-a, nikad za tihu izmenu vrednosti.
    _DIGIT_CONFUSIONS: Dict[str, List[str]] = {
        '0': ['8', '6', '9'],
        '1': ['7'],
        '2': ['7'],
        '3': ['8'],
        '4': ['9'],
        '5': ['6', '8'],
        '6': ['5', '8', '0'],
        '7': ['1', '2'],
        '8': ['0', '3', '5', '6'],
        '9': ['4', '0'],
    }

    @staticmethod
    def clean_jmbg_ocr(value: str) -> str:
        """
        Čisti OCR tekst JMBG-a - uklanja sve što nije cifra.

        VAŽNO: ovo NIKAD ne menja same cifre (nema "auto-correct" checksuma).
        Kontrolna cifra postoji da OTKRIJE grešku u OCR čitanju, ne da bude
        dugme za tihu ispravku - to bi sakrilo pravu OCR grešku iza lažnog
        "is_valid=True".

        Returns:
            Očišćeni JMBG (samo cifre, dužina može biti != 13 ako OCR nije
            sve pročitao)
        """
        return re.sub(r'\D', '', value)

    @classmethod
    def suggest_jmbg_correction(cls, digits: str) -> Optional[str]:
        """
        Pokušava da pronađe JEDNOZNAČAN predlog ispravke JMBG-a sa neispravnom
        kontrolnom cifrom, menjajući TAČNO JEDNU cifru kroz uobičajene OCR
        zabune (npr. 0<->8, 1<->7).

        Ovo se NIKAD ne primenjuje automatski na `validated_value` - samo se
        vraća kao predlog za ručnu proveru operatera (Java/GUI strana odlučuje
        da li i kako da ga prikaže).

        Ako promena bilo koje od 13 pozicija kroz njene uobičajene zabune daje
        VIŠE OD JEDNOG validnog kandidata, predlog je dvosmislen i vraćamo
        None (bolje nijedan predlog nego pogrešan).

        Returns:
            Predloženi ispravan JMBG (13 cifara) ili None ako nema jednoznačnog predloga.
        """
        if len(digits) != 13 or not digits.isdigit():
            return None

        candidates = set()
        for i, ch in enumerate(digits):
            for alt in cls._DIGIT_CONFUSIONS.get(ch, []):
                candidate = digits[:i] + alt + digits[i + 1:]
                if cls._validate_jmbg_checksum(candidate):
                    candidates.add(candidate)

        if len(candidates) == 1:
            return candidates.pop()
        return None

    @classmethod
    def validate_jmbg(cls, value: str) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """
        Validira JMBG (Jedinstveni maticni broj gradana).

        Format: DDMMGGGRRBBBK
        - DD: dan rodjenja (01-31)
        - MM: mesec rodjenja (01-12)
        - GGG: poslednje 3 cifre godine (000-999)
        - RR: regionalni kod (70-79 za Srbiju)
        - BBB: jedinstveni broj (000-499 muski, 500-999 zenski)
        - K: kontrolna cifra (modul 11)

        VAŽNO: kontrolna cifra se koristi SAMO da validira, nikad da tiho
        izmeni OCR pročitanu vrednost. Ako čeksuma ne prolazi, is_valid je
        False i cleaned_value ostaje TAČNO ono što je OCR pročitao.

        Returns:
            (is_valid, cleaned_value, error_message, suggested_value)
            suggested_value je opcioni jednoznačan predlog ispravke (vidi
            suggest_jmbg_correction) - nikad se sam ne primenjuje.
        """
        cleaned = cls.clean_jmbg_ocr(value)

        if len(cleaned) != cls.JMBG_LENGTH:
            return False, cleaned, f"JMBG mora imati {cls.JMBG_LENGTH} cifara, ima {len(cleaned)}", None

        if not cleaned.isdigit():
            return False, cleaned, "JMBG mora sadržati samo cifre", None

        if cls._validate_jmbg_checksum(cleaned):
            return True, cleaned, None, None

        suggestion = cls.suggest_jmbg_correction(cleaned)
        if suggestion:
            return (
                False,
                cleaned,
                "Neispravna kontrolna cifra JMBG-a (postoji mogući predlog ispravke - proveriti ručno)",
                suggestion,
            )
        return False, cleaned, "Neispravna kontrolna cifra JMBG-a", None

    @staticmethod
    def _validate_jmbg_checksum(jmbg: str) -> bool:
        """
        Validira kontrolnu cifru JMBG-a po modulu 11.

        Formula: 11 - ((7*(a+g) + 6*(b+h) + 5*(c+i) + 4*(d+j) + 3*(e+k) + 2*(f+l)) mod 11)
        Gde su a-m cifre JMBG-a od leva na desno.
        """
        if len(jmbg) != 13:
            return False

        try:
            # Tezinski faktori
            weights = [7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]

            # Izracunaj sumu
            total = sum(int(jmbg[i]) * weights[i] for i in range(12))
            remainder = total % 11

            # Odredi ocekivanu kontrolnu cifru
            if remainder == 0:
                expected_check = 0
            elif remainder == 1:
                # JMBG sa ostatkom 1 nije validan
                return False
            else:
                expected_check = 11 - remainder

            return int(jmbg[12]) == expected_check

        except (ValueError, IndexError):
            return False

    @classmethod
    def validate_index(cls, value: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validira broj indeksa studenta.
        Format: GGGG/BBBB (npr. 2023/0342)

        Returns:
            (is_valid, cleaned_value, error_message)
        """
        cleaned = value.strip()

        # Ukloni razmake
        cleaned = re.sub(r'\s+', '', cleaned)

        # Korekcija cestih OCR gresaka
        cleaned = cleaned.replace('l', '1').replace('I', '1')
        cleaned = cleaned.replace('O', '0').replace('o', '0')

        # Proveri standardni format
        if re.match(cls.INDEX_PATTERN, cleaned):
            return True, cleaned, None

        # Pokusaj izvuci cifre i formatirati
        digits = re.sub(r'\D', '', cleaned)
        if len(digits) == 8:
            formatted = f"{digits[:4]}/{digits[4:]}"
            return True, formatted, None

        # Mozda ima samo 7 cifara (godina/broj bez vodece nule)
        if len(digits) == 7:
            formatted = f"{digits[:4]}/{digits[4:].zfill(4)}"
            return True, formatted, None

        return False, cleaned, "Indeks mora biti u formatu GGGG/BBBB (npr. 2023/0342)"

    @classmethod
    def validate_school_year(cls, value: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validira skolsku godinu.
        Format: GGGG/GGGG (npr. 2024/2025)

        Returns:
            (is_valid, cleaned_value, error_message)
        """
        cleaned = value.strip()
        cleaned = re.sub(r'\s+', '', cleaned)

        # Proveri format
        if re.match(cls.SCHOOL_YEAR_PATTERN, cleaned):
            parts = cleaned.split('/')
            year1 = int(parts[0])
            year2 = int(parts[1])

            # Druga godina treba biti year1 + 1
            if year2 == year1 + 1:
                return True, cleaned, None
            else:
                return False, cleaned, f"Nevalidna skolska godina: druga godina mora biti {year1 + 1}"

        # Pokusaj izvuci cifre
        digits = re.sub(r'\D', '', cleaned)
        if len(digits) == 8:
            formatted = f"{digits[:4]}/{digits[4:]}"
            return True, formatted, None

        return False, cleaned, "Skolska godina mora biti u formatu GGGG/GGGG (npr. 2024/2025)"

    @classmethod
    def validate_date(cls, value: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validira datum i vraca u standardnom formatu DD.MM.YYYY.

        Returns:
            (is_valid, cleaned_value, error_message)
        """
        # SMART CLEANING - izvlači datum iz kompleksnih tekstova
        # Primer: "У БЕОГРАДУ, 1.7. 2025. год" -> "01.07.2025"
        cleaned = cls.clean_date_ocr(value)

        # Pokusaj parsirati po poznatim formatima
        for pattern, date_format in cls.DATE_PATTERNS:
            if re.match(pattern, cleaned):
                try:
                    date_obj = datetime.strptime(cleaned, date_format)

                    # Validacija godine
                    if date_obj.year < 1900 or date_obj.year > datetime.now().year + 1:
                        return False, cleaned, f"Nevazeca godina: {date_obj.year}"

                    formatted = date_obj.strftime("%d.%m.%Y")
                    return True, formatted, None

                except ValueError as e:
                    return False, cleaned, f"Nevazeci datum: {e}"

        # Pokusaj izvuci cifre i formatirati
        digits = re.sub(r'\D', '', cleaned)
        if len(digits) == 8:
            try:
                formatted = f"{digits[:2]}.{digits[2:4]}.{digits[4:]}"
                date_obj = datetime.strptime(formatted, "%d.%m.%Y")
                return True, formatted, None
            except ValueError:
                pass

        # Za 6 cifara (kratka godina)
        if len(digits) == 6:
            try:
                year = int(digits[4:6])
                full_year = 2000 + year if year < 50 else 1900 + year
                formatted = f"{digits[:2]}.{digits[2:4]}.{full_year}"
                date_obj = datetime.strptime(formatted, "%d.%m.%Y")
                return True, formatted, None
            except ValueError:
                pass

        return False, cleaned, "Datum mora biti u formatu DD.MM.YYYY"

    @classmethod
    def validate_year(cls, value: str,
                      min_year: int = 1940,
                      max_year: int = None) -> Tuple[bool, str, Optional[str]]:
        """
        Validira godinu.

        Returns:
            (is_valid, cleaned_value, error_message)
        """
        if max_year is None:
            max_year = datetime.now().year + 1

        # Smart Year Search
        # 1. Clean input
        cleaned = re.sub(r'\D', '', value)
        
        # 2. Search for 4-digit year pattern (19xx or 20xx)
        match = re.search(r'(19|20)\d{2}', cleaned)
        if match:
            year_candidate = match.group(0)
            year = int(year_candidate)
            if min_year <= year <= max_year:
                 return True, year_candidate, None
        
        # 3. Fallback logic: check exact match if length is 4
        if len(cleaned) == 4:
            try:
                year = int(cleaned)
                if min_year <= year <= max_year:
                    return True, cleaned, None
                else:
                    return False, cleaned, f"Godina mora biti izmedju {min_year} i {max_year}"
            except ValueError:
                pass
                
        # 4. Fallback for 2-digit years
        if len(cleaned) == 2:
            try:
                year = int(cleaned)
                full_year = 2000 + year if year < 50 else 1900 + year
                if min_year <= full_year <= max_year:
                    return True, str(full_year), None
            except ValueError:
                pass

        return False, cleaned, "Nevalidna godina"

    @classmethod
    def validate_text(cls, value: str,
                      min_length: int = 1,
                      max_length: int = 500,
                      required: bool = True) -> Tuple[bool, str, Optional[str]]:
        """
        Validira tekstualno polje.

        Returns:
            (is_valid, cleaned_value, error_message)
        """
        cleaned = value.strip()

        # Normalizuj razmake
        cleaned = re.sub(r'\s+', ' ', cleaned)

        if required and not cleaned:
            return False, cleaned, "Polje je obavezno"

        if not required and not cleaned:
            return True, cleaned, None

        if len(cleaned) < min_length:
            return False, cleaned, f"Minimalna duzina je {min_length} karaktera"

        if len(cleaned) > max_length:
            return False, cleaned[:max_length], f"Tekst skracen na {max_length} karaktera"

        return True, cleaned, None

    @classmethod
    def validate_numeric(cls, value: str,
                         min_val: Optional[int] = None,
                         max_val: Optional[int] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Validira numericko polje.

        Returns:
            (is_valid, cleaned_value, error_message)
        """
        cleaned = re.sub(r'\D', '', value)

        if not cleaned:
            return False, "", "Ocekuje se numericka vrednost"

        try:
            num_value = int(cleaned)

            if min_val is not None and num_value < min_val:
                return False, cleaned, f"Vrednost mora biti >= {min_val}"

            if max_val is not None and num_value > max_val:
                return False, cleaned, f"Vrednost mora biti <= {max_val}"

            return True, cleaned, None

        except ValueError:
            return False, cleaned, "Nevalidna numericka vrednost"

    @classmethod
    def validate_espb(cls, value: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validira ESPB bodove (0-400).

        Returns:
            (is_valid, cleaned_value, error_message)
        """
        return cls.validate_numeric(value, min_val=0, max_val=400)

    @classmethod
    def validate_field(cls, value: str, field_config: dict) -> dict:
        """
        Validira polje na osnovu konfiguracije iz template-a.

        Args:
            value: Vrednost za validaciju
            field_config: Konfiguracija polja iz template-a

        Returns:
            Dict sa rezultatom validacije
        """
        field_type = field_config.get('type', 'TEXT')
        field_name = field_config.get('name', '')
        required = field_config.get('required', False)
        validation = field_config.get('validation', {})
        suggested_value = None

        if field_type == 'TEXT':
            is_valid, cleaned, error = cls.validate_text(
                value,
                required=required,
                min_length=validation.get('min_length', 1),
                max_length=validation.get('max_length', 500)
            )

        elif field_type == 'NUMERIC':
            # Specijalan slucaj za JMBG
            if validation.get('length') == 13 or 'jmbg' in field_name.lower():
                is_valid, cleaned, error, suggested_value = cls.validate_jmbg(value)
            # Specijalan slucaj za godine (godina_rodjenja, godina_upisa, itd.)
            elif 'godina' in field_name.lower() and 'studija' not in field_name.lower():
                # SMART CLEANING za godine: "2925" -> "2025", "12006" -> "2006"
                cleaned_value = cls.clean_year_ocr(value)
                is_valid, cleaned, error = cls.validate_numeric(
                    cleaned_value,
                    min_val=validation.get('min'),
                    max_val=validation.get('max')
                )
            else:
                is_valid, cleaned, error = cls.validate_numeric(
                    value,
                    min_val=validation.get('min'),
                    max_val=validation.get('max')
                )

        elif field_type == 'ALPHANUMERIC':
            # Specijalan slucaj za indeks
            if 'indeks' in field_name.lower() or 'index' in field_name.lower():
                is_valid, cleaned, error = cls.validate_index(value)
            # Skolska godina
            elif 'skolska_godina' in field_name.lower() or 'school_year' in field_name.lower():
                is_valid, cleaned, error = cls.validate_school_year(value)
            else:
                is_valid, cleaned, error = cls.validate_text(value, required=required)

        elif field_type == 'DATE':
            is_valid, cleaned, error = cls.validate_date(value)

        elif field_type in ('OMR_SINGLE', 'OMR_MULTI'):
            # OMR vrednosti se ne validiraju ovde, vec u OMR procesoru
            is_valid, cleaned, error = True, value, None

        elif field_type == 'SIGNATURE':
            # Potpis se ne validira
            is_valid, cleaned, error = True, value, None

        else:
            # Nepoznat tip - samo basic text validacija
            is_valid, cleaned, error = cls.validate_text(value, required=required)

        return {
            'is_valid': is_valid,
            'original_value': value,
            'cleaned_value': cleaned,
            'error': error,
            'field_type': field_type,
            'suggested_value': suggested_value
        }


def postprocess_text(text: str, operations: List[str]) -> str:
    """
    Primenjuje post-processing operacije na OCR tekst.

    Args:
        text: Originalni tekst
        operations: Lista operacija za primenu

    Returns:
        Obradjen tekst
    """
    result = text

    for op in operations:
        if op == 'capitalize_first':
            # Capitalize samo prvo slovo, ostalo lowercase
            result = result.capitalize() if result else result
        elif op == 'capitalize_words':
            # Capitalize svaku rec
            result = result.title() if result else result
        elif op == 'uppercase':
            result = result.upper()
        elif op == 'lowercase':
            result = result.lower()
        elif op == 'strip_spaces':
            result = result.strip()
        elif op == 'normalize_spaces':
            result = re.sub(r'\s+', ' ', result).strip()
        elif op == 'remove_special':
            # Ukloni specijalne karaktere osim osnovnih
            result = re.sub(r'[^\w\s\-\./]', '', result, flags=re.UNICODE)
        elif op == 'digits_only':
            result = re.sub(r'\D', '', result)

    return result


def correct_common_ocr_errors(text: str, field_type: str = 'TEXT') -> str:
    """
    Ispravlja ceste OCR greske.

    Args:
        text: Originalni tekst
        field_type: Tip polja (TEXT, NUMERIC, etc.)

    Returns:
        Korigovan tekst
    """
    if not text:
        return text

    result = text

    if field_type in ('NUMERIC', 'ALPHANUMERIC'):
        # Zameni slova koja lice na cifre
        replacements = {
            'O': '0', 'o': '0',
            'l': '1', 'I': '1', '|': '1',
            'Z': '2',
            'S': '5', 's': '5',
            'B': '8',
            'g': '9', 'q': '9',
        }
        for old, new in replacements.items():
            result = result.replace(old, new)

    elif field_type == 'TEXT':
        # Zameni cifre koje lice na slova (obrnuto od gornjeg)
        replacements = {
            '0': 'O',
            '1': 'l',
            '5': 'S',
            '8': 'B',
        }
        # Ne primenjuj automatski - moze biti pogresno

    return result


def test_validators():
    """Test funkcija za validatore."""
    print("Testiram Validatore...")

    # Test JMBG
    valid_jmbg = "0101998710029"  # Primer - mozda nije validan checksum
    is_valid, cleaned, error, suggested = FieldValidator.validate_jmbg(valid_jmbg)
    print(f"JMBG '{valid_jmbg}': valid={is_valid}, cleaned={cleaned}, error={error}, suggested={suggested}")

    # Test indeksa
    test_index = "2023/0342"
    is_valid, cleaned, error = FieldValidator.validate_index(test_index)
    print(f"Indeks '{test_index}': valid={is_valid}, cleaned={cleaned}, error={error}")

    # Test indeksa sa greskom
    test_index2 = "2O23/O342"  # O umesto 0
    is_valid, cleaned, error = FieldValidator.validate_index(test_index2)
    print(f"Indeks '{test_index2}': valid={is_valid}, cleaned={cleaned}, error={error}")

    # Test datuma
    test_date = "15.03.2024"
    is_valid, cleaned, error = FieldValidator.validate_date(test_date)
    print(f"Datum '{test_date}': valid={is_valid}, cleaned={cleaned}, error={error}")

    # Test godine
    test_year = "1998"
    is_valid, cleaned, error = FieldValidator.validate_year(test_year)
    print(f"Godina '{test_year}': valid={is_valid}, cleaned={cleaned}, error={error}")

    # Test postprocessing
    test_text = "  vukaSIN   lukIC  "
    processed = postprocess_text(test_text, ['normalize_spaces', 'capitalize_words'])
    print(f"Postprocess '{test_text}': '{processed}'")

    print("Test zavrsen!")


if __name__ == "__main__":
    test_validators()
