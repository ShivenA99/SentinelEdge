package com.sentineledge

/**
 * Extracts 18 handcrafted linguistic features from a single sentence of
 * transcript text.
 *
 * This is a Kotlin port of the Python feature extractor at
 * `sentinel_edge/features/handcrafted.py`.  The same word lists, regex
 * patterns, and feature ordering are used so that the feature vector is
 * compatible with the ONNX model consumed by [FraudClassifier].
 *
 * Feature order (indices 0-17):
 * ```
 *  0  urgency_count
 *  1  action_count
 *  2  financial_count
 *  3  impersonation_count
 *  4  has_url              (0/1)
 *  5  has_shortened_url    (0/1)
 *  6  has_verify_pattern   (0/1)
 *  7  has_threat           (0/1)
 *  8  has_prize            (0/1)
 *  9  has_account_ref      (0/1)
 * 10  dollar_sign          (0/1)
 * 11  has_phone_number     (0/1)
 * 12  char_count
 * 13  word_count
 * 14  avg_word_len
 * 15  url_count
 * 16  exclamation_count
 * 17  caps_ratio
 * ```
 */
class FeatureExtractor {

    // ------------------------------------------------------------------
    // Word lists (mirrors Python `_*_WORDS`)
    // ------------------------------------------------------------------

    companion object {
        /** Number of features produced by [extract]. */
        const val FEATURE_COUNT = 18

        private val URGENCY_WORDS = listOf(
            "urgent", "immediately", "now", "act", "deadline", "expire",
            "hurry", "quickly", "asap", "right away", "limited time", "last chance"
        )

        private val ACTION_WORDS = listOf(
            "click", "verify", "confirm", "validate", "login", "update",
            "provide", "submit", "enter", "activate", "download", "install"
        )

        private val FINANCIAL_WORDS = listOf(
            "bank", "account", "credit", "payment", "wire", "transfer",
            "routing", "deposit", "withdrawal", "balance", "transaction", "invoice"
        )

        private val IMPERSONATION_WORDS = listOf(
            "irs", "fbi", "ssa", "microsoft", "apple", "amazon",
            "google", "police", "government", "federal", "agent", "department",
            "officer"
        )

        // ------------------------------------------------------------------
        // Pre-compiled regex patterns (mirrors Python patterns)
        // ------------------------------------------------------------------

        private val URL_PATTERN =
            Regex("""https?://\S+|www\.\S+""", RegexOption.IGNORE_CASE)

        private val SHORTENED_URL_PATTERN = Regex(
            """(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly""" +
                """|rebrand\.ly|cutt\.ly|shorturl\.at)[/\s]?""",
            RegexOption.IGNORE_CASE
        )

        private val VERIFY_PATTERN =
            Regex("""\b(?:verify|confirm|validate)\b""", RegexOption.IGNORE_CASE)

        private val THREAT_PATTERN = Regex(
            """\b(?:arrest|suspend|terminate|cancel|lock|seize|revoke|prosecute)""",
            RegexOption.IGNORE_CASE
        )

        private val PRIZE_PATTERN = Regex(
            """\b(?:won|winner|prize|congratulations|selected|lucky|reward)\b""",
            RegexOption.IGNORE_CASE
        )

        private val ACCOUNT_REF_PATTERN = Regex(
            """\b(?:account|password|pin|ssn|social\s+security|credentials)\b""",
            RegexOption.IGNORE_CASE
        )

        private val PHONE_NUMBER_PATTERN = Regex(
            """(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}"""
        )
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Extract 18 handcrafted features from [text].
     *
     * @param text A single sentence (or short segment) from a transcript.
     * @return [FloatArray] of length [FEATURE_COUNT] ready for [FraudClassifier.predict].
     */
    fun extract(text: String): FloatArray {
        val lower = text.lowercase()

        // Count-based features
        val urgencyCount = countWordOccurrences(lower, URGENCY_WORDS)
        val actionCount = countWordOccurrences(lower, ACTION_WORDS)
        val financialCount = countWordOccurrences(lower, FINANCIAL_WORDS)
        val impersonationCount = countWordOccurrences(lower, IMPERSONATION_WORDS)

        // Binary pattern features
        val urlMatches = URL_PATTERN.findAll(text).toList()
        val hasUrl = if (urlMatches.isNotEmpty()) 1.0f else 0.0f
        val hasShortenedUrl = if (SHORTENED_URL_PATTERN.containsMatchIn(lower)) 1.0f else 0.0f
        val hasVerifyPattern = if (VERIFY_PATTERN.containsMatchIn(text)) 1.0f else 0.0f
        val hasThreat = if (THREAT_PATTERN.containsMatchIn(text)) 1.0f else 0.0f
        val hasPrize = if (PRIZE_PATTERN.containsMatchIn(text)) 1.0f else 0.0f
        val hasAccountRef = if (ACCOUNT_REF_PATTERN.containsMatchIn(text)) 1.0f else 0.0f
        val dollarSign = if ('$' in text) 1.0f else 0.0f
        val hasPhoneNumber = if (PHONE_NUMBER_PATTERN.containsMatchIn(text)) 1.0f else 0.0f

        // Structural / statistical features
        val charCount = text.length.toFloat()
        val words = text.split(Regex("""\s+""")).filter { it.isNotEmpty() }
        val wordCount = words.size.toFloat()
        val avgWordLen = if (words.isNotEmpty()) {
            words.sumOf { it.length }.toFloat() / words.size
        } else {
            0.0f
        }
        val urlCount = urlMatches.size.toFloat()
        val exclamationCount = text.count { it == '!' }.toFloat()

        val alphaChars = text.filter { it.isLetter() }
        val capsRatio = if (alphaChars.isNotEmpty()) {
            alphaChars.count { it.isUpperCase() }.toFloat() / alphaChars.length
        } else {
            0.0f
        }

        return floatArrayOf(
            urgencyCount,       // 0
            actionCount,        // 1
            financialCount,     // 2
            impersonationCount, // 3
            hasUrl,             // 4
            hasShortenedUrl,    // 5
            hasVerifyPattern,   // 6
            hasThreat,          // 7
            hasPrize,           // 8
            hasAccountRef,      // 9
            dollarSign,         // 10
            hasPhoneNumber,     // 11
            charCount,          // 12
            wordCount,          // 13
            avgWordLen,         // 14
            urlCount,           // 15
            exclamationCount,   // 16
            capsRatio,          // 17
        )
    }

    /**
     * Extract features and return them as a named map (useful for debugging).
     */
    fun extractNamed(text: String): Map<String, Float> {
        val features = extract(text)
        return mapOf(
            "urgency_count" to features[0],
            "action_count" to features[1],
            "financial_count" to features[2],
            "impersonation_count" to features[3],
            "has_url" to features[4],
            "has_shortened_url" to features[5],
            "has_verify_pattern" to features[6],
            "has_threat" to features[7],
            "has_prize" to features[8],
            "has_account_ref" to features[9],
            "dollar_sign" to features[10],
            "has_phone_number" to features[11],
            "char_count" to features[12],
            "word_count" to features[13],
            "avg_word_len" to features[14],
            "url_count" to features[15],
            "exclamation_count" to features[16],
            "caps_ratio" to features[17],
        )
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    /**
     * Count how many times any word in [wordList] appears in [textLower].
     *
     * Multi-word phrases are matched as substrings.  Single words are
     * matched on word boundaries to avoid partial matches.
     */
    private fun countWordOccurrences(textLower: String, wordList: List<String>): Float {
        var count = 0
        for (word in wordList) {
            if (' ' in word) {
                // Multi-word phrase: count substring occurrences
                var startIndex = 0
                while (true) {
                    val idx = textLower.indexOf(word, startIndex)
                    if (idx == -1) break
                    count++
                    startIndex = idx + 1
                }
            } else {
                // Single word: word-boundary match
                val pattern = Regex("""\b${Regex.escape(word)}\b""")
                count += pattern.findAll(textLower).count()
            }
        }
        return count.toFloat()
    }
}
