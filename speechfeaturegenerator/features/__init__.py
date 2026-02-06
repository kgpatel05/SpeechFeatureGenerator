"""Feature extraction modules: phoneme, diphone, entropy, surprisal, word, syllable, lexical, wordfreq, glove, phonotactic."""

from speechfeaturegenerator.features.diphone import diphone, generate_diphone_features
from speechfeaturegenerator.features.entropy import entropy, generate_entropy_features
from speechfeaturegenerator.features.phoneme import phoneme, generate_phoneme_features
from speechfeaturegenerator.features.surprisal import surprisal, generate_surprisal_features
from speechfeaturegenerator.features.word import word, generate_word_features
from speechfeaturegenerator.features.syllable import syllable, generate_syllable_features
from speechfeaturegenerator.features.lexical import lexical, generate_lexical_features
from speechfeaturegenerator.features.wordfreq import wordfreq, generate_wordfreq_features
from speechfeaturegenerator.features.glove import glove, generate_glove_features
from speechfeaturegenerator.features.phonotactic import phonotactic, generate_phonotactic_features

__all__ = [
    "diphone",
    "generate_diphone_features",
    "entropy",
    "generate_entropy_features",
    "phoneme",
    "generate_phoneme_features",
    "surprisal",
    "generate_surprisal_features",
    "word",
    "generate_word_features",
    "syllable",
    "generate_syllable_features",
    "lexical",
    "generate_lexical_features",
    "wordfreq",
    "generate_wordfreq_features",
    "glove",
    "generate_glove_features",
    "phonotactic",
    "generate_phonotactic_features",
]
