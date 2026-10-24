package li.joye.yakuyomi.engine

/**
 * [MTYOMI] SCOWL-style English OCR noise filter (the xpmanga approach): OCR engines
 * hallucinate letter-like strings out of screen tones, clothing folds and background
 * art. A region whose text contains NO word from the bundled English word list
 * (SCOWL, via hunspell-en_US) is almost certainly such noise, not real lettering —
 * real text drawn on the art (skill names, shouts, SFX) is made of real words and
 * survives. Judged per region, before inpainting and before the LLM, so noise costs
 * no tokens and is never "translated".
 *
 * Conservative by design: a region is noise only when it has at least one Latin
 * token of length >= 2 and NONE of its tokens (or their light stems) are in the
 * list. Regions with no Latin tokens at all (numbers, symbols, CJK) are kept.
 */
internal object NoiseFilter {

    private val TOKEN_SPLIT = Regex("[^a-z']+")
    private val SUFFIXES = listOf("'s", "ing", "ed", "es", "er", "est", "ly", "s")

    fun isNoise(text: String, words: Set<String>): Boolean {
        if (words.isEmpty()) return false
        val tokens = text.lowercase().split(TOKEN_SPLIT).filter { it.length >= 2 }
        if (tokens.isEmpty()) return false
        return tokens.none { tok -> tok in words || stem(tok) in words }
    }

    /** Light suffix stemmer: the SCOWL/hunspell list stores base forms, so inflected
     *  OCR words (running, walked, swords) must be reduced before the lookup. */
    private fun stem(w: String): String {
        var t = w
        for (suf in SUFFIXES) {
            if (t.endsWith(suf) && t.length - suf.length >= 3) {
                t = t.dropLast(suf.length)
                break
            }
        }
        // undouble a final consonant left by -ing/-ed stripping (runn -> run, stopp -> stop)
        if (t.length >= 4 && t[t.length - 1] == t[t.length - 2] && t.last() !in "aeiou") {
            t = t.dropLast(1)
        }
        return t
    }
}
