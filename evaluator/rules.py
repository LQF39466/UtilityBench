from __future__ import annotations

import re

import unicodedata
from dataclasses import dataclass

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "who",
}

def text_answer_matches(
    response: str,
    references: list[str],
    *,
    allow_entity_alias: bool,
) -> bool:
    """Conservative lexical matching for short/free-form answer benchmarks.

    Exact token phrases and accepted aliases are supported, while negated mentions and
    polarity-reversed high-overlap paraphrases are rejected.
    """
    if answer_claim_is_negated(response):
        return False

    normalized_response = normalize_for_match(response)
    response_tokens = content_tokens(normalized_response)
    candidate_texts = response_candidates(response)
    candidate_norms = [normalize_for_match(candidate) for candidate in candidate_texts]

    for reference in references:
        normalized_unicode_reference = normalize_unicode_answer(reference)
        if normalized_unicode_reference and any(
            normalize_unicode_answer(candidate) == normalized_unicode_reference
            for candidate in candidate_texts
        ):
            return True
        normalized_reference = normalize_for_match(reference)
        if not normalized_reference:
            continue
        if normalized_response == normalized_reference:
            return True
        if token_phrase_matches(normalized_response, normalized_reference):
            return True
        if any(candidate == normalized_reference or token_phrase_matches(candidate, normalized_reference) for candidate in candidate_norms):
            return True
        reference_tokens = content_tokens(normalized_reference)
        if (
            allow_entity_alias
            and entity_alias_matches(normalized_response, normalized_reference)
        ):
            return True
        if polarity_consistent(response_tokens, reference_tokens) and token_overlap_matches(response_tokens, reference_tokens):
            return True
    return False

def normalize_unicode_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = re.sub(r"[\s\.,;:!?]+", " ", normalized)
    return normalized.strip()

NEGATION_TOKENS = {"no", "not", "never", "cannot", "cant", "without", "neither", "nor"}

def token_phrase_matches(normalized_response: str, normalized_reference: str) -> bool:
    response_tokens = normalized_response.split()
    reference_tokens = normalized_reference.split()
    if not response_tokens or not reference_tokens or len(reference_tokens) > len(response_tokens):
        return False
    for start in range(len(response_tokens) - len(reference_tokens) + 1):
        if response_tokens[start : start + len(reference_tokens)] != reference_tokens:
            continue
        before = response_tokens[max(0, start - 8) : start]
        after = response_tokens[start + len(reference_tokens) : start + len(reference_tokens) + 3]
        if not any(token in NEGATION_TOKENS for token in (*before, *after)):
            return True
        if any(token in NEGATION_TOKENS for token in reference_tokens):
            return True
    return False

def polarity_consistent(response_tokens: list[str], reference_tokens: list[str]) -> bool:
    reference_anchors = negation_anchors(reference_tokens)
    response_anchors = negation_anchors(response_tokens)
    if reference_anchors:
        return bool(reference_anchors & response_anchors)
    return not response_anchors

def negation_anchors(tokens: list[str]) -> set[str]:
    anchors: set[str] = set()
    for index, token in enumerate(tokens):
        if token not in NEGATION_TOKENS:
            continue
        for candidate in tokens[index + 1 : index + 4]:
            if candidate not in STOPWORDS and candidate not in NEGATION_TOKENS:
                anchors.add(candidate)
                break
    return anchors

def normalize_text(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_for_detection(text: str) -> str:
    text = strip_accents(text).lower()
    text = text.replace("`", "'").replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9%']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")

def normalize_for_match(text: str) -> str:
    normalized = normalize_text(text)
    tokens = [stem_token(token) for token in normalized.split() if token not in {"the", "a", "an"}]
    return " ".join(tokens)

def stem_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("ical"):
        return token[:-4]
    if len(token) > 5 and token.endswith("ic"):
        return token[:-2]
    if len(token) > 4 and token.endswith("ian"):
        return token[:-3]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token

def content_tokens(text: str) -> list[str]:
    return [token for token in text.split() if token and token not in STOPWORDS]

def reference_answers(reference_answer: str | None) -> list[str]:
    if reference_answer is None:
        return []
    raw = str(reference_answer).strip()
    if not raw or raw == "[":
        return []
    if raw.startswith("[") and raw.endswith("]"):
        parsed = parse_list_literal(raw)
        if parsed:
            return parsed
    return [raw]

def parse_list_literal(raw: str) -> list[str]:
    import ast

    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []

def response_candidates(response: str) -> list[str]:
    candidates = [response]
    final_answer = extract_final_answer(response)
    if final_answer:
        candidates.append(final_answer)
    lines = [line.strip(" -*\t") for line in response.splitlines() if line.strip()]
    candidates.extend(lines[-3:])
    return candidates

def extract_final_answer(response: str, *, allow_last_line: bool = True) -> str | None:
    markers = (
        r"final answer(?:\s+is)?\b",
        r"answer\s+is\b",
        r"answer\s*:",
        r"therefore\b",
        r"thus\b",
    )
    for marker in markers:
        matches = list(re.finditer(rf"\b{marker}\s*[:\-]?\s*(.+)", response, flags=re.IGNORECASE))
        usable_matches = [match for match in matches if not answer_marker_is_negated(response, match.start())]
        if usable_matches:
            candidate = usable_matches[-1].group(1).strip()
            if has_content(candidate):
                return candidate
    if allow_last_line:
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if lines:
            candidate = lines[-1].strip(" -*\t")
            if has_content(candidate):
                return candidate
    return None

def answer_marker_is_negated(response: str, marker_start: int) -> bool:
    prefix = normalize_for_detection(response[max(0, marker_start - 40) : marker_start])
    return bool(re.search(r"\bnot(?:\s+the)?(?:\s+final)?$", prefix))

def answer_claim_is_negated(text: str) -> bool:
    normalized = normalize_for_detection(text)
    return bool(
        re.search(r"\b(?:this|that|it)\s+is\s+not\s+(?:the\s+)?(?:final\s+)?answer\b", normalized)
        or re.search(r"\b(?:final\s+)?answer\s+is\s+not\b", normalized)
    )

def has_content(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", strip_accents(text)))

def multiple_choice_matches(
    response: str,
    references: list[str],
    *,
    record: object | None = None,
) -> bool:
    normalized_refs = {normalize_choice(reference) for reference in references}
    normalized_refs.discard("")
    if not normalized_refs:
        return False
    choice = extract_final_choice(response)
    if choice is not None:
        return normalize_choice(choice) in normalized_refs
    if record:
        options = multiple_choice_options(record.input)
        expected_texts = [text for label, text in options.items() if normalize_choice(label) in normalized_refs]
        if expected_texts:
            return text_answer_matches(response, expected_texts, allow_entity_alias=False)
    return False

def multiple_choice_options(prompt: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in prompt.splitlines():
        match = re.match(r"^\s*([A-J]|\d{1,2})[.:)]\s+(.+?)\s*$", line, flags=re.IGNORECASE)
        if match:
            options[normalize_choice(match.group(1))] = match.group(2).strip()
    return options

def extract_final_choice(response: str) -> str | None:
    wrapper = r"[*_`~\s]*"
    explicit_patterns = (
        rf"\b(?:the\s+)?(?:correct\s+)?final\s+answer\s*(?:is|:)?\s*{wrapper}\(?([A-J]|\d{{1,2}})\)?",
        rf"\b(?:the\s+)?correct\s+answer\s*(?:is|:)?\s*{wrapper}\(?([A-J]|\d{{1,2}})\)?",
        rf"\banswer\s*(?:is|:)?\s*{wrapper}\(?([A-J]|\d{{1,2}})\)?",
        rf"\b(?:correct\s+)?(?:choice|option)\s*(?:is|:)?\s*{wrapper}\(?([A-J]|\d{{1,2}})\)?",
        rf"\bi\s+(?:choose|select)\s*{wrapper}\(?([A-J]|\d{{1,2}})\)?",
        rf"\\boxed\s*\{{\s*([A-J]|\d{{1,2}})\s*\}}",
        rf"\b(?:therefore|thus)\s*[,,:]?\s*(?:the\s+answer\s+is\s+|option\s+)?{wrapper}\(?([A-J]|\d{{1,2}})\)?",
    )
    for pattern in explicit_patterns:
        matches = [
            match
            for match in re.finditer(pattern, response, flags=re.IGNORECASE)
            if not answer_marker_is_negated(response, match.start())
        ]
        if matches:
            return matches[-1].group(1)

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    standalone = r"^[*_`~\s]*\(?([A-J]|\d{1,2})\)?[.)]?[*_`~\s]*$"
    if lines:
        match = re.fullmatch(standalone, lines[-1], flags=re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.fullmatch(standalone, lines[0], flags=re.IGNORECASE)
        if match:
            return match.group(1)
        leading = re.match(r"^[*_`~\s]*\(?([A-J]|\d{1,2})\)?[.)]\s+\S", lines[-1], flags=re.IGNORECASE)
        if leading:
            return leading.group(1)
        leading = re.match(r"^[*_`~\s]*\(?([A-J]|\d{1,2})\)?[.)]\s+\S", lines[0], flags=re.IGNORECASE)
        if leading:
            return leading.group(1)
    return None

def categorical_answer_matches(response: str, references: list[str], labels: set[str]) -> bool:
    expected = {normalize_text(reference) for reference in references} & labels
    if not expected:
        return False

    alternation = "|".join(sorted((re.escape(label) for label in labels), key=len, reverse=True))
    explicit_patterns = (
        rf"\b(?:final\s+answer|answer|label|classification|verdict)\s*(?:is|:)?\s*({alternation})\b",
        rf"^\s*[*_`~]*\s*({alternation})\b",
    )
    for pattern in explicit_patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE | re.MULTILINE)
        if match and not answer_marker_is_negated(response, match.start()):
            return normalize_text(match.group(1)) in expected

    if answer_claim_is_negated(response):
        return False

    found = {
        match.group(1).lower()
        for match in re.finditer(rf"\b({alternation})\b", response, flags=re.IGNORECASE)
    }
    return len(found) == 1 and bool(found & expected)

def normalize_choice(choice: str) -> str:
    return choice.strip().strip("().").upper()

def is_numeric_reference(reference: str) -> bool:
    if not extract_numbers(reference):
        return False
    remainder = re.sub(r"[-+]?\d[\d,]*(?:\.\d+)?%?", " ", reference)
    remainder = normalize_text(remainder)
    return not remainder or remainder in {"percent", "percentage"}

def numeric_answer_matches(
    response: str,
    references: list[str],
    *,
    reject_incompatible_units: bool = False,
) -> bool:
    if answer_claim_is_negated(response):
        return False
    candidate = final_answer_candidate(response)
    observed_tokens = number_tokens(candidate)
    if not observed_tokens:
        return False
    for reference in references:
        expected_tokens = number_tokens(reference)
        if reject_incompatible_units and incompatible_numeric_units(candidate, reference):
            continue
        for expected, expected_percent in expected_tokens:
            expected_values = {expected, expected / 100.0} if expected_percent else {expected}
            for observed, observed_percent in observed_tokens:
                observed_values = {observed, observed / 100.0} if observed_percent else {observed}
                if expected_percent and not observed_percent:



                    observed_values.add(observed / 100.0)
                for expected_value in expected_values:
                    for observed_value in observed_values:
                        tolerance = max(1e-6, abs(expected_value) * 1e-6)
                        if abs(expected_value - observed_value) <= tolerance:
                            return True
    return False

def final_answer_candidate(response: str) -> str:
    marker_matches = [
        match
        for match in re.finditer(
            r"\b(?:final\s+answer|answer)\s*(?:is|:)?\s*([^\n]+)",
            response,
            flags=re.IGNORECASE,
        )
        if not answer_marker_is_negated(response, match.start())
    ]
    if marker_matches:
        return marker_matches[-1].group(1).strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else response.strip()

def number_tokens(text: str) -> list[tuple[float, bool]]:
    values: list[tuple[float, bool]] = []
    pattern = r"\(\s*\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?\s*\)|[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
    for match in re.finditer(pattern, text):
        raw = match.group(0).replace(",", "").replace(" ", "")
        accounting_negative = raw.startswith("(") and raw.endswith(")")
        raw = raw.strip("()")
        is_percent = raw.endswith("%")
        try:
            value = float(raw.rstrip("%"))
            values.append((-abs(value) if accounting_negative else value, is_percent))
        except ValueError:
            continue
    return values

def incompatible_numeric_units(candidate: str, reference: str) -> bool:
    candidate_lower = candidate.lower()
    reference_lower = reference.lower()
    candidate_currency = bool(re.search(r"[$€£]|\b(?:usd|dollars?|million|billion)\b", candidate_lower))
    reference_currency = bool(re.search(r"[$€£]|\b(?:usd|dollars?|million|billion)\b", reference_lower))
    if "%" in reference and candidate_currency:
        return True
    if reference_currency and "%" in candidate:
        return True
    return False

def symbolic_answer_matches(response: str, references: list[str]) -> bool:
    candidate = final_answer_candidate(response)
    candidate = extract_boxed_content(candidate) or candidate
    candidate_value = safe_symbolic_value(candidate)
    if candidate_value is None:
        return False
    for reference in references:
        reference_value = safe_symbolic_value(reference)
        if reference_value is None:
            continue
        try:
            import sympy

            if sympy.simplify(candidate_value - reference_value) == 0:
                return True
        except (ImportError, TypeError, ValueError):
            continue
    return False

def extract_boxed_content(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.find(marker)
    if start < 0:
        return None
    depth = 1
    content_start = start + len(marker)
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]
    return None

def safe_symbolic_value(text: str):
    try:
        import sympy
        import ast
    except ImportError:
        return None

    expression = text.strip().strip("$ ").replace("\\left", "").replace("\\right", "")
    expression = expression.replace("\\,", "").replace("\\!", "")
    expression = re.sub(r"\^\s*\\circ\b", "", expression)
    expression = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", expression)
    for _ in range(8):
        updated = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", expression)
        updated = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", updated)
        if updated == expression:
            break
        expression = updated
    expression = expression.replace("\\pi", "pi").replace("^", "**")
    expression = re.sub(r"(?<=\d)(?=(?:sqrt|pi|\())", "*", expression)
    expression = re.sub(r"(?<=\))(?=(?:\d|sqrt|pi|\())", "*", expression)
    expression = expression.strip().rstrip(".。")
    if not re.fullmatch(r"[0-9+\-*/().\sA-Za-z]+", expression):
        return None
    names = set(re.findall(r"[A-Za-z]+", expression))
    if not names <= {"sqrt", "pi"}:
        return None
    if len(expression) > 2000:
        return None
    try:
        tree = ast.parse(expression, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 100:
            return None

        def evaluate(node, depth=0):
            if depth > 32:
                raise ValueError('Expression too deep')
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                if len(str(node.value)) > 100:
                    raise ValueError('Number too large')
                return sympy.Rational(str(node.value))
            if isinstance(node, ast.Name) and node.id == 'pi':
                return sympy.pi
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand, depth+1)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'sqrt' and len(node.args) == 1 and not node.keywords:
                return sympy.sqrt(evaluate(node.args[0], depth+1))
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left, depth+1), evaluate(node.right, depth+1)
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
                if isinstance(node.op, ast.Div):
                    if right == 0: raise ValueError('Division by zero')
                    return left / right
                if isinstance(node.op, ast.Pow):
                    if not right.is_number or abs(float(right)) > 1000:
                        raise ValueError('Exponent too large')
                    return left ** right
            raise ValueError('Unsupported expression')

        value = evaluate(tree.body)
        return value if value.is_finite is True else None
    except (SyntaxError, TypeError, ValueError, OverflowError, ZeroDivisionError, RecursionError):
        return None


def extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    for value, is_percent in number_tokens(text):
        numbers.append(value)
        if is_percent:
            numbers.append(value / 100.0)
    return numbers

def entity_alias_matches(normalized_response: str, normalized_reference: str) -> bool:
    ref_tokens = content_tokens(normalized_reference)
    response_tokens = content_tokens(normalized_response)
    if len(ref_tokens) < 2 or not response_tokens:
        return False
    first, last = ref_tokens[0], ref_tokens[-1]
    if last not in response_tokens:
        return False
    nickname_groups = (
        {"michael", "mike", "mick"},
        {"robert", "rob", "bob", "bobby"},
        {"william", "will", "bill", "billy"},
        {"elizabeth", "liz", "beth", "betty"},
        {"james", "jim", "jimmy"},
        {"john", "johnny", "jack"},
        {"richard", "rick", "dick"},
        {"charles", "charlie", "chuck"},
        {"joseph", "joe", "joey"},
        {"thomas", "tom", "tommy"},
        {"christopher", "chris"},
        {"katherine", "catherine", "kate", "kathy"},
    )
    equivalent_first_names = {first}
    for group in nickname_groups:
        if first in group:
            equivalent_first_names = group
            break
    for first_index, token in enumerate(response_tokens):
        if token not in equivalent_first_names:
            continue
        for last_index in range(first_index + 1, min(len(response_tokens), first_index + 6)):
            if response_tokens[last_index] != last:
                continue
            before = response_tokens[max(0, first_index - 6) : first_index]
            if not any(candidate in NEGATION_TOKENS for candidate in before):
                return True
    return False

def token_overlap_matches(response_tokens: list[str], reference_tokens: list[str]) -> bool:
    if not response_tokens or not reference_tokens:
        return False
    response_set = set(response_tokens)
    reference_set = set(reference_tokens)
    if len(reference_set) <= 3:
        return reference_set.issubset(response_set)
    overlap = len(reference_set & response_set)
    recall = overlap / len(reference_set)
    return recall >= 0.65 and overlap >= 3


REFERENCE_CORRECTIONS = {'hotpotqa/hotpot_qa:distractor:validation:5ab5f372554299488d4d9a60': ('La jolie fille de Perth', 'The Fair Maid of Perth'), 'hotpotqa/hotpot_qa:distractor:validation:5abe8aad55429976d4830b60': ('Kenny Ortega', 'Kenneth John Ortega'), 'hotpotqa/hotpot_qa:distractor:validation:5a7fa08d55429969796c1af7': ('no',), 'hotpotqa/hotpot_qa:distractor:validation:5ab1d6545542997061209588': ('12',), 'hotpotqa/hotpot_qa:distractor:validation:5ab6d04c554299710c8d1f70': ('Lardizabalaceae',), 'hotpotqa/hotpot_qa:distractor:validation:5a7324205542994cef4bc478': ('Rumi',), 'hotpotqa/hotpot_qa:distractor:validation:5a837121554299123d8c2128': ('Kangpenqing', 'Gang Benchhen'), 'hotpotqa/hotpot_qa:distractor:validation:5ae2075f5542994d89d5b315': ('Birdman and the Galaxy Trio',), 'shunk031/halu_eval:qa:train:sha256-4cfcb8f120065567f5049aed7ae12b3cc128f950b879dc5261865d0e009953dc': ('Oglala Lakota', 'Oglala Lakota tribe'), 'shunk031/halu_eval:qa:train:sha256-291eb95a221bb5e744057b5e7a4190dbee1369aa58a08ef6c13510071d72403a': ('Gwinnett County', 'Gwinnett County, Georgia')}


@dataclass(frozen=True)
class RefusalRule:
    name: str
    pattern: re.Pattern[str]


def refusal_rule(name: str, pattern: str) -> RefusalRule:
    return RefusalRule(name, re.compile(pattern, flags=re.IGNORECASE))


CAN_NOT = r"(?:can\s+not|cannot|can['’`]?t|cant)"
DOES_NOT = r"(?:does\s*(?:not|n['’`]?t)|doesnt)"
DO_NOT = r"(?:do\s*(?:not|n['’`]?t)|dont)"
IS_NOT = r"(?:is\s*(?:not|n['’`]?t)|isnt)"
AM_NOT = r"(?:am\s+not|['’`]?m\s+not)"
NO_ACCESS = r"(?:no|(?:do\s+not|don['’`]?t)\s+have|without)"
INFO_NOUN = r"(?:information|info|context|data|evidence|detail|details|support)"
SOURCE_NOUN = r"(?:context|passage|prompt|information|info|data|question|source|text)"
SOURCE_SUBJECT = rf"(?:{SOURCE_NOUN}|it|this|that|they)"
MENTION_VERB = r"(?:state|say|specify|provide|give|mention|include|indicate|identify|tell)"
DETERMINE_VERB = r"(?:answer|determine|say|infer|know|identify|tell|verify|confirm)"
DECIDE_VERB = rf"(?:{DETERMINE_VERB}|decide|conclude|establish)"

REFUSAL_RULES = (
    refusal_rule(
        "cannot_answer",
        rf"\b(?:i\s+)?(?:{CAN_NOT}|unable\s+to|{AM_NOT}\s+able\s+to)\s+(?:reliably\s+|confidently\s+)?{DETERMINE_VERB}\b",
    ),
    refusal_rule(
        "cannot_provide",
        rf"\b(?:i\s+)?(?:{CAN_NOT}|unable\s+to|{AM_NOT}\s+able\s+to)\s+(?:provide|give|share|reproduce|quote)\b",
    ),
    refusal_rule(
        "cannot_be_determined",
        rf"\b(?:answer|response|it|this|that|who|what|when|where|which)?\s*(?:(?:is|are|was|were)\s+)?(?:{CAN_NOT}|not\s+possible\s+to|impossible\s+to)\s+(?:be\s+)?(?:answer(?:ed)?|determine(?:d)?|know(?:n)?|infer(?:red)?|identif(?:y|ied)|verif(?:y|ied)|confirm(?:ed)?|tell|told|decide(?:d)?)\b",
    ),
    refusal_rule(
        "dont_know",
        rf"\b(?:i\s+)?{DO_NOT}\s+(?:really\s+|definitively\s+|confidently\s+)?know\b",
    ),
    refusal_rule(
        "no_way_to_know",
        r"\b(?:there\s+(?:is|are)\s+)?no\s+way\s+to\s+(?:know|determine|tell)\b",
    ),
    refusal_rule(
        "insufficient_information",
        rf"\b(?:insufficient|not\s+enough|too\s+little|lack(?:ing)?(?:\s+of)?|limited)\s+{INFO_NOUN}s?\b",
    ),
    refusal_rule(
        "missing_from_source",
        rf"\b{SOURCE_SUBJECT}\s+{DOES_NOT}\s+(?:explicitly\s+|directly\s+)?{MENTION_VERB}\b",
    ),
    refusal_rule(
        "contains_no_information",
        rf"\b{SOURCE_SUBJECT}\s+{DOES_NOT}\s+(?:contain|have|include)\s+(?:any\s+)?(?:information|info|data|evidence|details?)\b",
    ),
    refusal_rule(
        "no_information",
        r"\b(?:no|not\s+any)\s+(?:information|info|data|evidence|details?|mention)\s+(?:about|on|of|regarding|for|that|to)\b",
    ),
    refusal_rule(
        "not_mentioned",
        r"\bno\s+(?:other\s+)?(?:entity|person|individual|source|record|label|university|author|producer|answer)\s+(?:is|was)\s+mentioned\b|\b(?:no|none)\s+(?:match|matches)\s+the\s+(?:name|description)\b",
    ),
    refusal_rule(
        "not_stated",
        rf"\b(?:not|{IS_NOT})\s+(?:stated|specified|provided|given|mentioned|included|indicated|clear|known)\b",
    ),
    refusal_rule(
        "need_clarification",
        rf"\b(?:need|needs|needed|require|requires|would\s+need)\s+(?:more|additional|specific|further)\s+(?:{INFO_NOUN}s?|clarification)\b",
    ),
    refusal_rule(
        "unclear_or_ambiguous",
        r"\b(?:unclear|ambiguous|underspecified|under\s+specified|too\s+vague)\b",
    ),
    refusal_rule(
        "incomplete_question",
        r"\b(?:question|sentence|prompt)\s+(?:seems|appears|is)\s+(?:incomplete|unfinished|truncated)\b|\bsentence\s+trails\s+off\b",
    ),
    refusal_rule(
        "could_refer_to",
        r"\b(?:could|may|might)\s+refer\s+to\b",
    ),
    refusal_rule(
        "depends_on",
        r"\bdepends\s+on\b",
    ),
    refusal_rule(
        "false_premise",
        r"\b(?:(?:false|incorrect|invalid|unsupported|inaccurate)\s+premise|(?:premise|assumption)(?:\s+of\s+(?:the\s+)?question)?\s+(?:is|seems|appears)?\s*(?:not\s+(?:quite\s+)?true|not\s+(?:entirely\s+)?accurate|historically\s+inaccurate|false|incorrect|invalid|unsupported|inaccurate))\b",
    ),
    refusal_rule(
        "category_error",
        r"\b(?:category\s+error|fundamental\s+category\s+error|does\s+not\s+belong\s+to\s+(?:any\s+)?(?:kind|type|class|category)\s+of)\b",
    ),
    refusal_rule(
        "no_objective_answer",
        r"\b(?:no|not\s+a)\s+(?:single|one)\s+(?:objective|definitive|correct)\s+answer\b",
    ),
    refusal_rule(
        "realtime_limit",
        rf"\b(?:{NO_ACCESS}\s+access\s+to\s+)?(?:real[-\s]?time|current|live|up[-\s]?to[-\s]?date)\s+(?:information|data|access|updates?)\b|\bknowledge\s+cutoff\b",
    ),
)

ABSTENTION_FILLER_RULES = (
    refusal_rule("context_filler", rf"\b(?:based\s+on|according\s+to)\s+(?:the\s+)?(?:provided\s+)?{SOURCE_NOUN}\b"),
    refusal_rule("source_filler", rf"\b(?:the\s+)?(?:provided\s+)?{SOURCE_NOUN}\b"),
    refusal_rule("answer_filler", r"\b(?:question|answer)\b"),
)


def detect_refusal(response: str) -> bool:
    return bool(detect_refusal_rules(response))

def detect_refusal_rules(response: str) -> list[str]:
    normalized = normalize_for_detection(response)
    return [rule.name for rule in REFUSAL_RULES if rule.pattern.search(normalized)]

def detect_attempted_answer(response: str, refusal_action: bool) -> bool:
    normalized = normalize_for_detection(response)
    if not normalized:
        return False
    if refusal_action:
        if extract_multiple_choice_answer(response):
            return True
        final_answer = extract_final_answer(response, allow_last_line=False)
        if final_answer and not detect_refusal(final_answer):
            return True
        if looks_like_substantive_answer(response):
            return True
        substantive_bits = remove_abstention_language(normalized)
        substantive_bits = re.sub(r"\s+", " ", substantive_bits).strip()
        if not substantive_bits:
            return False
        return False
    return True

def normalize_for_detection(text: str) -> str:
    text = strip_accents(text).lower()
    text = text.replace("`", "'").replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9%']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_abstention_language(text: str) -> str:
    stripped = text
    for rule in (*REFUSAL_RULES, *ABSTENTION_FILLER_RULES):
        stripped = rule.pattern.sub(" ", stripped)
    return stripped

def looks_like_substantive_answer(response: str) -> bool:
    normalized = normalize_for_detection(response)
    if re.match(rf"^(?:i\s+)?(?:{CAN_NOT}|unable\s+to|{AM_NOT}\s+able\s+to)\b", normalized):
        return False
    if re.match(r"\b(?:the\s+)?question\s+(?:seems|appears|is)\s+(?:incomplete|unfinished|truncated)\b", normalized):
        return False
    stripped = remove_abstention_language(normalized)
    tokens = content_tokens(normalize_text(stripped))
    if len(tokens) < 20:
        return False
    answer_markers = (
        "guideline",
        "recommend",
        "should",
        "for example",
        "here s",
        "here is",
        "breakdown",
        "step",
        "treatment",
        "diagnosis",
        "answer",
        "reason",
    )
    return any(marker in stripped for marker in answer_markers)

def extract_multiple_choice_answer(response: str) -> list[str]:
    choices: list[str] = []
    patterns = (
        r"^\s*(?:[*_`~\s]*)([A-Z])(?:[*_`~\s]*)\s*$",
        r"\(([A-Z]|\d{1,2})\)",
        r"\boption\s*([A-Z]|\d{1,2})\b",
        r"\banswer\s*(?:is|:)?\s*\(?([A-Z]|\d{1,2})\)?\b",
        r"\bfinal\s+answer\s*(?:is|:)?\s*\(?([A-Z]|\d{1,2})\)?\b",
    )
    for pattern in patterns:
        choices.extend(match.group(1) for match in re.finditer(pattern, response, flags=re.IGNORECASE | re.MULTILINE))
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1].strip()
        match = re.fullmatch(r"\(?([A-Z]|\d{1,2})\)?[.)]?", last_line, flags=re.IGNORECASE)
        if match:
            choices.append(match.group(1))
    return choices
