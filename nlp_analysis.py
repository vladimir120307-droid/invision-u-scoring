import re
import math
from collections import Counter

import textstat

from models import NLPMetrics


POSITIVE_WORDS_RU = {
    "отлично", "прекрасно", "замечательно", "успех", "достижение", "победа",
    "развитие", "рост", "вдохновение", "мотивация", "стремление", "энергия",
    "радость", "счастье", "творчество", "инновация", "лидерство", "команда",
    "цель", "мечта", "возможность", "потенциал", "прогресс", "улучшение",
    "благодарность", "уверенность", "сила", "решительность", "амбиции",
    "страсть", "энтузиазм", "преданность", "упорство", "целеустремлённость",
    "вклад", "помощь", "поддержка", "сотрудничество", "партнёрство",
}

NEGATIVE_WORDS_RU = {
    "плохо", "неудача", "провал", "разочарование", "проблема", "трудность",
    "конфликт", "стресс", "тревога", "страх", "сомнение", "неуверенность",
    "слабость", "ошибка", "потеря", "кризис", "депрессия", "усталость",
    "безразличие", "апатия", "невозможно", "бесполезно", "бессмысленно",
}

POSITIVE_WORDS_EN = {
    "excellent", "great", "amazing", "success", "achievement", "victory",
    "development", "growth", "inspiration", "motivation", "aspiration",
    "energy", "joy", "happiness", "creativity", "innovation", "leadership",
    "team", "goal", "dream", "opportunity", "potential", "progress",
    "improvement", "gratitude", "confidence", "strength", "determination",
    "ambition", "passion", "enthusiasm", "dedication", "perseverance",
    "contribution", "support", "collaboration", "partnership", "impact",
}

NEGATIVE_WORDS_EN = {
    "bad", "failure", "disappointment", "problem", "difficulty", "conflict",
    "stress", "anxiety", "fear", "doubt", "weakness", "mistake", "loss",
    "crisis", "depression", "fatigue", "indifference", "apathy", "impossible",
    "useless", "meaningless", "struggle", "obstacle", "setback",
}

LEADERSHIP_KEYWORDS = {
    "лидер", "руководил", "управлял", "организовал", "координировал",
    "возглавил", "инициировал", "создал", "основал", "вёл", "наставник",
    "менторство", "делегировал", "мотивировал", "вдохновил", "команда",
    "leader", "led", "managed", "organized", "coordinated", "founded",
    "initiated", "created", "mentor", "delegated", "motivated", "inspired",
}

GROWTH_KEYWORDS = {
    "научился", "развил", "улучшил", "вырос", "преодолел", "освоил",
    "изменился", "трансформация", "эволюция", "адаптация", "прогресс",
    "learned", "developed", "improved", "grew", "overcame", "mastered",
    "changed", "transformation", "evolution", "adaptation", "progress",
}

MOTIVATION_KEYWORDS = {
    "мечта", "цель", "стремлюсь", "хочу", "верю", "миссия", "призвание",
    "страсть", "вдохновение", "будущее", "вижу", "представляю", "планирую",
    "dream", "goal", "aspire", "believe", "mission", "calling", "passion",
    "inspiration", "future", "envision", "plan", "purpose", "vision",
}


def detect_language(text: str) -> str:
    cyrillic_count = len(re.findall(r'[а-яёА-ЯЁ]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    if cyrillic_count > latin_count:
        return "ru"
    return "en"


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'[а-яёa-z]+', text)
    return tokens


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def compute_sentiment(text: str, language: str = "auto") -> tuple[float, str, dict[str, float]]:
    if language == "auto":
        language = detect_language(text)

    tokens = tokenize(text)
    if not tokens:
        return 0.0, "neutral", {}

    if language == "ru":
        pos_words = POSITIVE_WORDS_RU
        neg_words = NEGATIVE_WORDS_RU
    else:
        pos_words = POSITIVE_WORDS_EN
        neg_words = NEGATIVE_WORDS_EN

    pos_count = sum(1 for t in tokens if t in pos_words)
    neg_count = sum(1 for t in tokens if t in neg_words)
    total = len(tokens)

    pos_ratio = pos_count / total
    neg_ratio = neg_count / total

    score = (pos_ratio - neg_ratio) / max(pos_ratio + neg_ratio, 0.001)
    score = max(-1.0, min(1.0, score))

    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"

    emotional_tone = {
        "позитив": round(pos_ratio * 100, 1),
        "негатив": round(neg_ratio * 100, 1),
        "нейтральность": round((1 - pos_ratio - neg_ratio) * 100, 1),
    }

    return score, label, emotional_tone


def compute_complexity(text: str) -> tuple[float, dict]:
    tokens = tokenize(text)
    sentences = split_sentences(text)

    if not tokens or not sentences:
        return 0.0, {}

    avg_word_length = sum(len(t) for t in tokens) / len(tokens)
    avg_sentence_length = len(tokens) / len(sentences)

    unique_tokens = set(tokens)
    vocabulary_richness = len(unique_tokens) / len(tokens) if tokens else 0

    long_words = [t for t in tokens if len(t) > 8]
    long_word_ratio = len(long_words) / len(tokens)

    word_freq = Counter(tokens)
    hapax_legomena = sum(1 for count in word_freq.values() if count == 1)
    hapax_ratio = hapax_legomena / len(unique_tokens) if unique_tokens else 0

    complexity = (
        normalize_value(avg_word_length, 3.0, 8.0) * 0.20
        + normalize_value(avg_sentence_length, 5.0, 30.0) * 0.25
        + vocabulary_richness * 0.25
        + long_word_ratio * 0.15
        + hapax_ratio * 0.15
    )

    complexity = min(1.0, max(0.0, complexity))

    details = {
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "vocabulary_richness": round(vocabulary_richness, 3),
        "long_word_ratio": round(long_word_ratio, 3),
        "hapax_ratio": round(hapax_ratio, 3),
        "unique_words": len(unique_tokens),
        "total_words": len(tokens),
        "total_sentences": len(sentences),
    }

    return complexity, details


def compute_readability(text: str, language: str = "auto") -> float:
    if language == "auto":
        language = detect_language(text)

    if language == "en":
        try:
            grade = textstat.flesch_kincaid_grade(text)
            return min(20.0, max(0.0, grade))
        except Exception:
            pass

    tokens = tokenize(text)
    sentences = split_sentences(text)
    if not tokens or not sentences:
        return 0.0

    avg_word_len = sum(len(t) for t in tokens) / len(tokens)
    avg_sent_len = len(tokens) / len(sentences)

    grade = 0.5 * avg_sent_len + 0.3 * avg_word_len - 2.0
    return min(20.0, max(0.0, grade))


def detect_ai_generated(text: str) -> tuple[float, str]:
    tokens = tokenize(text)
    sentences = split_sentences(text)

    if len(tokens) < 20:
        return 0.0, "insufficient_data"

    signals = []

    sent_lengths = [len(tokenize(s)) for s in sentences]
    if len(sent_lengths) > 2:
        mean_len = sum(sent_lengths) / len(sent_lengths)
        variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_len if mean_len > 0 else 0
        uniformity = 1.0 - min(1.0, cv)
        signals.append(("sentence_uniformity", uniformity, 0.25))

    word_freq = Counter(tokens)
    freq_values = sorted(word_freq.values(), reverse=True)
    if len(freq_values) > 5:
        top_5_freq = sum(freq_values[:5]) / len(tokens)
        if top_5_freq < 0.15:
            signals.append(("low_repetition", 0.7, 0.15))
        else:
            signals.append(("low_repetition", 0.2, 0.15))

    transition_words = {
        "однако", "тем не менее", "более того", "кроме того", "следовательно",
        "таким образом", "в результате", "в заключение", "во-первых", "во-вторых",
        "however", "moreover", "furthermore", "consequently", "therefore",
        "additionally", "nevertheless", "in conclusion", "firstly", "secondly",
    }
    transition_count = sum(1 for t in tokens if t in transition_words)
    transition_density = transition_count / len(sentences) if sentences else 0
    if transition_density > 0.5:
        signals.append(("high_transitions", min(1.0, transition_density), 0.20))
    else:
        signals.append(("high_transitions", transition_density * 0.5, 0.20))

    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    bigram_freq = Counter(bigrams)
    unique_bigram_ratio = len(bigram_freq) / len(bigrams) if bigrams else 1
    if unique_bigram_ratio > 0.92:
        signals.append(("bigram_uniqueness", (unique_bigram_ratio - 0.85) * 5, 0.20))
    else:
        signals.append(("bigram_uniqueness", 0.1, 0.20))

    para_starts = []
    paragraphs = text.split("\n")
    for p in paragraphs:
        p = p.strip()
        if p:
            first_word = tokenize(p)
            if first_word:
                para_starts.append(first_word[0])
    if len(para_starts) > 2:
        unique_starts = len(set(para_starts)) / len(para_starts)
        if unique_starts > 0.9:
            signals.append(("paragraph_variety", 0.3, 0.10))
        else:
            signals.append(("paragraph_variety", 0.1, 0.10))

    exclamations = text.count("!") + text.count("?")
    ellipses = text.count("...") + text.count("…")
    personal = sum(1 for t in tokens if t in {"я", "мне", "мой", "меня", "i", "my", "me"})
    personality_score = (exclamations + ellipses + personal) / len(tokens)
    lack_of_personality = 1.0 - min(1.0, personality_score * 10)
    signals.append(("lack_personality", lack_of_personality, 0.10))

    total_weight = sum(w for _, _, w in signals)
    if total_weight == 0:
        return 0.0, "likely_human"

    ai_score = sum(s * w for _, s, w in signals) / total_weight
    ai_score = min(1.0, max(0.0, ai_score))

    if ai_score > 0.65:
        label = "likely_ai"
    elif ai_score > 0.40:
        label = "uncertain"
    else:
        label = "likely_human"

    return ai_score, label


def compute_authenticity(text: str) -> float:
    tokens = tokenize(text)
    if len(tokens) < 10:
        return 0.5

    personal_pronouns = {"я", "мне", "мой", "моя", "моё", "мои", "меня", "мною",
                         "i", "my", "me", "mine", "myself"}
    personal_count = sum(1 for t in tokens if t in personal_pronouns)
    personal_ratio = personal_count / len(tokens)

    specific_markers = 0
    year_pattern = re.findall(r'\b(19|20)\d{2}\b', text)
    specific_markers += len(year_pattern)

    number_pattern = re.findall(r'\b\d+\b', text)
    specific_markers += min(len(number_pattern), 5)

    proper_nouns = re.findall(r'[А-ЯA-Z][а-яa-z]{2,}', text)
    specific_markers += min(len(proper_nouns) // 3, 3)

    specificity = min(1.0, specific_markers / 8)

    emotional_markers = text.count("!") + text.count("?") + text.count("...")
    emotional_ratio = min(1.0, emotional_markers / max(len(split_sentences(text)), 1) * 0.5)

    authenticity = (
        min(1.0, personal_ratio * 8) * 0.35
        + specificity * 0.35
        + emotional_ratio * 0.15
        + (1.0 - detect_ai_generated(text)[0]) * 0.15
    )

    return min(1.0, max(0.0, authenticity))


def extract_keyword_density(text: str, keyword_sets: dict[str, set]) -> dict[str, float]:
    tokens = tokenize(text)
    if not tokens:
        return {k: 0.0 for k in keyword_sets}

    result = {}
    for category, keywords in keyword_sets.items():
        count = sum(1 for t in tokens if t in keywords)
        result[category] = round(count / len(tokens) * 100, 2)
    return result


def analyze_essay(text: str, language: str = "auto") -> NLPMetrics:
    if not text or len(text.strip()) < 10:
        return NLPMetrics()

    if language == "auto":
        language = detect_language(text)

    sentiment_score, sentiment_label, emotional_tone = compute_sentiment(text, language)
    complexity_score, complexity_details = compute_complexity(text)
    readability = compute_readability(text, language)
    ai_score, ai_label = detect_ai_generated(text)
    authenticity = compute_authenticity(text)

    keyword_density = extract_keyword_density(text, {
        "лидерство": LEADERSHIP_KEYWORDS,
        "рост": GROWTH_KEYWORDS,
        "мотивация": MOTIVATION_KEYWORDS,
    })

    return NLPMetrics(
        sentiment_score=round(sentiment_score, 3),
        sentiment_label=sentiment_label,
        complexity_score=round(complexity_score, 3),
        vocabulary_richness=complexity_details.get("vocabulary_richness", 0.0),
        avg_sentence_length=complexity_details.get("avg_sentence_length", 0.0),
        readability_grade=round(readability, 1),
        ai_detection_score=round(ai_score, 3),
        ai_detection_label=ai_label,
        authenticity_score=round(authenticity, 3),
        keyword_density=keyword_density,
        emotional_tone=emotional_tone,
    )


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
