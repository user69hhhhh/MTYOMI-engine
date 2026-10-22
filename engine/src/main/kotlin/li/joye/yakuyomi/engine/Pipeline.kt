package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Typeface
import android.util.Log
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.coroutineScope
import kotlin.math.roundToInt

/**
 * 單頁翻譯結果（§11：成功才覆蓋+marker、略過不覆蓋、失敗不覆蓋待重試）。
 * 引擎只回結果，不碰檔案——覆蓋/marker/resume 由呼叫端（下載 worker）依此處理（§3、§12.6）。
 */
sealed interface PageResult {
    /** 成功：可覆蓋原檔 + 寫「已翻譯」marker。 */
    data class Translated(val page: Bitmap, val stats: PageStats, val analysis: PageAnalysis? = null) : PageResult

    /** 沒東西可翻（偵測不到字 / OCR 全空 / 譯文全被過濾）：保留原圖、標記略過、**不覆蓋**。 */
    data class Skipped(val reason: String, val stats: PageStats) : PageResult

    /** 出錯（網路/429 重試後仍失敗/例外）：保留原圖、**不標記**、之後可重試。 */
    data class Failed(val reason: String) : PageResult
}

/**
 * 重繪素材（給「最低成本切換去字方法」用）：seg 文字遮罩 + regions（含 quad/角度/onArt/源文/譯文）。
 * 原圖由呼叫端持有（translatePage 的輸入）、去字方法由呼叫端決定，故不在此。
 * 序列化/落地（含 mask 轉文字塞 json）由呼叫端（reader）負責。
 */
data class PageAnalysis(val mask: Bitmap, val regions: List<TextRegion>)

/** 逐階段計時與計數（除錯/效能用）。 */
data class PageStats(
    val lines: Int,
    val regions: Int,
    val kept: Int,
    val detectMs: Long,
    val ocrMs: Long,
    val translateMs: Long,
    val inpaintMs: Long,
    val renderMs: Long,
    val wallMs: Long = 0,   // 實際牆鐘時間（去字‖翻譯重疊 ⇒ 通常 < totalMs；省下的＝重疊掉的）
    val promptTokens: Int = 0,      // 本頁 LLM 請求的 prompt token（無 LLM/代理不回＝0）。供統計：用量只記、不計價。
    val completionTokens: Int = 0,  // 本頁 LLM 請求的 completion token。
) {
    /** 各階段純計算時間之和（不含重疊修正）；實際耗時看 [wallMs]。 */
    val totalMs: Long get() = detectMs + ocrMs + translateMs + inpaintMs + renderMs
}

/**
 * 引擎主 pipeline：單頁 偵測→OCR→分群→翻譯→過濾→去字→排版。
 * 順序對齊 manga_translator.py 主流程（§5 順序＝第一層）；orchestration＝第二層。
 *
 * **§11 不變式焊進此處：永不用比原圖更糟的東西覆蓋。**
 *   - 偵測不到字 / OCR 全空 / 譯文全失敗 → [PageResult.Skipped]（保留原圖、不覆蓋；去字若已並發跑出來也丟棄）。
 *   - 單 block 翻譯失敗 → **去字後重貼 OCR 原文**（日文）——非「保留源圖」。讓去字不必等翻譯 ⇒ 去字(CPU)與翻譯(網路)並發重疊。
 *   - 任一階段拋例外（網路/429 重試後仍失敗等）→ [PageResult.Failed]（保留原圖、不覆蓋、可重試；丟棄已並發的去字）。
 *
 * **去字‖翻譯重疊**：兩者只依賴 OCR、互不爭資源（網路 vs CPU）⇒ 同時跑，[PageStats.wallMs] < [PageStats.totalMs]（省下重疊掉的）。
 *
 * 模型由呼叫端建好傳入；本類不碰檔案、不管跨頁批次與 resume。
 * **生命週期**：[close] 會收掉傳入的 detector/ocr/inpainter 的原生 session ——
 * 走 [Yakuyomi.create] 時這三顆由工廠建、歸本 pipeline 所有，`use { }` 即可。
 * 進階：若你注入「想重用、共享」的元件，請自己管生命週期、別呼叫本 [close]（否則會把共享元件一起關掉）。
 */
class Pipeline(
    private val detector: Detector,
    private val ocr: OcrEngine,
    private val translator: Translator?, // null＝不翻譯（純偵測/OCR 除錯用）
    private val inpainter: Inpainter,
    private val cfg: EngineConfig = EngineConfig(),
    private val typeface: Typeface? = null,
) : TranslationEngine {

    override suspend fun translatePage(page: Bitmap): PageResult = coroutineScope {
        val tWall = System.currentTimeMillis()
        EngineTrace.log("pipe.page.enter ${page.width}x${page.height}")
        // 偵測
        val tDet = System.currentTimeMillis()
        val detection = try {
            detector.detect(page)
        } catch (t: Throwable) {
            Log.e(TAG, "偵測失敗", t); return@coroutineScope PageResult.Failed("detect: ${t.message}")
        }
        val lines = detection.lines
        EngineTrace.log("pipe.detect.done lines=${lines.size}")
        val detectMs = System.currentTimeMillis() - tDet
        if (lines.isEmpty()) {
            return@coroutineScope PageResult.Skipped("偵測不到文字", PageStats(0, 0, 0, detectMs, 0, 0, 0, 0))
        }

        // OCR + 分群
        val tOcr = System.currentTimeMillis()
        EngineTrace.log("pipe.ocr.enter lines=${lines.size}")
        try {
            ocr.recognize(page, lines)
        } catch (t: Throwable) {
            Log.e(TAG, "OCR 失敗", t); return@coroutineScope PageResult.Failed("ocr: ${t.message}")
        }
        EngineTrace.log("pipe.ocr.exit")
        // 實驗 secondPass：弱行（讀不到字／低分短讀）裁出 2× 放大再 OCR 一次（分群前、只採用更好的結果）。
        if (cfg.ocr.secondPass) {
            try {
                secondPassOcr(page, lines)
            } catch (t: Throwable) {
                Log.w(TAG, "OCR 二輪整體失敗（略過）: ${t.message}")
            }
        }
        val regions = Grouping.group(lines)
        val ocrMs = System.currentTimeMillis() - tOcr
        // 去字集＝有 OCR 原文的區（空白＝疑似誤偵測，不去字、保畫面）。此集翻譯前就確定 ⇒ 去字可與翻譯並發。
        val textRegions = regions.filter { it.sourceText.isNotBlank() }
        // 實驗 matchTextColor：去字前從原圖取樣每區的筆畫色（去字後原字像素就沒了，必須在此量）。
        if (cfg.render.matchTextColor) {
            for (r in textRegions) r.srcTextColor = sampleTextColor(page, detection.textMask, r)
        }
        if (textRegions.isEmpty()) {
            return@coroutineScope PageResult.Skipped("OCR 全空", PageStats(lines.size, regions.size, 0, detectMs, ocrMs, 0, 0, 0))
        }

        // ★ 去字（CPU）‖ 翻譯（網路）並發：兩者只依賴 OCR，可同時跑（網路等待時 CPU 去字、互不爭資源）。
        // §11 改：失敗區不再「保留源圖」，而是「去字後重貼 OCR 原文」(使用者拍板：重貼成本低、不需源圖狀態) ⇒ 去字與翻譯解耦。
        // 不限去字方法（boxfill/lama 皆可重疊）。boxfill(~0.5s)整段藏進翻譯(~2.7s)＝免費。
        var inpaintMs = 0L
        val inpaintJob = async {
            val t0 = System.currentTimeMillis()
            EngineTrace.log("pipe.inpaint.enter regions=${textRegions.size}")
            val r = inpainter.inpaint(page, textRegions, detection.textMask)
            EngineTrace.log("pipe.inpaint.exit")
            inpaintMs = System.currentTimeMillis() - t0
            r
        }

        var translateMs = 0L
        var promptTok = 0
        var completionTok = 0
        // per-call 診斷 metadata（區域變數、不讀 LlmTranslator 的共享單值欄位 → 跨頁併發各頁拿自己的、不 race）。
        var llmError: String? = null
        var llmRaw: String? = null
        val llm = translator as? LlmTranslator
        if (translator != null) {
            val tTr = System.currentTimeMillis()
            EngineTrace.log("pipe.translate.enter n=${textRegions.size}")
            val cht = try {
                if (llm != null) {
                    // 走 translateDetailed → per-call 拿 translations + usage + error + raw（跨頁併發安全）。
                    val r = llm.translateDetailed(textRegions.map { it.sourceText })
                    r.usage?.let { promptTok = it.promptTokens; completionTok = it.completionTokens }
                    llmError = r.error
                    llmRaw = r.raw
                    r.translations
                } else {
                    translator.translate(textRegions.map { it.sourceText })
                }
            } catch (t: Throwable) {
                Log.e(TAG, "翻譯失敗", t)
                inpaintJob.cancelAndJoin() // 翻譯掛 → 丟棄去字、留原圖（§11；native run 不可中斷，cancel 實為等它跑完再丟）
                return@coroutineScope PageResult.Failed("translate: ${t.message}")
            }
            textRegions.forEachIndexed { j, r -> r.translatedText = cht.getOrElse(j) { r.sourceText } }
            translateMs = System.currentTimeMillis() - tTr
            EngineTrace.log("pipe.translate.exit err=$llmError")
        } else {
            textRegions.forEach { it.translatedText = it.sourceText } // 無 key debug：排版原文
        }

        // 判定每區譯文有效性（空白/數字/regex/譯==原＝失敗）。整頁全失敗 → 留原圖（Skipped、丟棄去字）。
        val kept = if (translator != null) TextFilter.apply(textRegions, cfg.translator.filterText) else textRegions
        if (kept.isEmpty()) {
            val aligned = textRegions.count { it.translatedText.isNotBlank() && it.translatedText != it.sourceText }
            val dbg = textRegions.take(2).joinToString(" ‖ ") { "${it.sourceText.take(8)}→${it.translatedText.take(8)}" }
            Log.w(TAG, "全數過濾 對齊$aligned/${textRegions.size} err=$llmError 回應=$llmRaw")
            inpaintJob.cancelAndJoin()
            // §11 盲點修正：分辨「網路/格式軟失敗」vs「真的全不可譯」。
            // LlmTranslator 對網路/HTTP 例外是「catch + 回傳原文」（不丟例外）→ 全頁 translated==source → 落到這裡全數過濾。
            // 若一律回 Skipped(標記略過、算已處理)，網路失敗的頁會被當「已翻」、整章不變紅（正是此盲點）。改用 error 分流（per-call、不 race）：
            //  - error != null（例外〔網路/HTTP〕或部分解析）→ Failed：不標記、之後重試、整章變紅（呼叫端 drain 標 ERROR）。
            //  - error == null（LLM 正常全解析、但內容全被過濾，如整頁狀聲詞被原樣回 translated==source）→ Skipped：略過、不無限重試。
            return@coroutineScope if (llmError != null) {
                PageResult.Failed("全數過濾(LLM 失敗 $llmError)｜回應=${llmRaw?.take(80)}")
            } else {
                PageResult.Skipped(
                    "全數過濾 對齊$aligned/${textRegions.size}｜回應=$llmRaw｜$dbg",
                    PageStats(
                        lines.size, regions.size, 0, detectMs, ocrMs, translateMs, 0, 0,
                        promptTokens = promptTok, completionTokens = completionTok,
                    ),
                )
            }
        }
        // 失敗的區（不在 kept）→ 譯文改回原文＝去字後重貼 OCR 日文（TextRegion 無 equals override→HashSet 走 identity）。
        // 實驗 eraseUntranslated：失敗區改為「只去字、不回貼」（多為被 LLM 原樣回傳的狀聲詞 SFX）⇒ 排版清單濾掉它們。
        val keptSet = kept.toHashSet()
        val renderRegions: List<TextRegion>
        if (cfg.render.eraseUntranslated) {
            textRegions.forEach { if (it !in keptSet) it.translatedText = "" }
            renderRegions = textRegions.filter { it.translatedText.isNotBlank() }
        } else {
            textRegions.forEach { if (it !in keptSet) it.translatedText = it.sourceText }
            renderRegions = textRegions
        }

        // 等去字完成（多半已與翻譯重疊跑完）
        EngineTrace.log("pipe.inpaint.await")
        val cleaned = try {
            inpaintJob.await()
        } catch (t: Throwable) {
            Log.e(TAG, "去字失敗", t); return@coroutineScope PageResult.Failed("inpaint: ${t.message}")
        }

        // 排版（renderRegions：kept 貼譯文、失敗區貼原文；eraseUntranslated 開時失敗區不入列＝只去字不回貼）
        val tRn = System.currentTimeMillis()
        EngineTrace.log("pipe.render.enter")
        val report = Renderer.renderWithReport(cleaned, renderRegions, cfg.render, typeface)
        var finalPage = report.bitmap
        var renderMs = System.currentTimeMillis() - tRn
        // 實驗 fitRetranslate：字級到底仍溢出的 kept 區 → 帶「改短」指示向 LLM 重翻一次（只收更短的新譯文）、重排一次。
        if (cfg.translator.fitRetranslate && llm != null && report.overflow.isNotEmpty()) {
            val targets = report.overflow.filter { it in keptSet }
            if (targets.isNotEmpty()) {
                EngineTrace.log("pipe.fit-retry n=${targets.size}")
                val retry = try {
                    llm.translateDetailed(targets.map { it.sourceText }, FIT_SHORTER_INSTRUCTION)
                } catch (t: Throwable) {
                    Log.w(TAG, "fit 重翻失敗，保留原譯文：${t.message}")
                    null
                }
                if (retry != null) {
                    retry.usage?.let { promptTok += it.promptTokens; completionTok += it.completionTokens }
                    var changed = false
                    targets.forEachIndexed { j, r ->
                        val nt = retry.translations.getOrNull(j)
                        if (nt != null && nt.isNotBlank() && nt != r.sourceText && nt.length < r.translatedText.length) {
                            r.translatedText = nt
                            changed = true
                        }
                    }
                    if (changed) {
                        val tR2 = System.currentTimeMillis()
                        finalPage = Renderer.render(cleaned, renderRegions, cfg.render, typeface)
                        renderMs += System.currentTimeMillis() - tR2
                    }
                }
            }
        }
        EngineTrace.log("pipe.page.done")

        PageResult.Translated(
            finalPage,
            PageStats(
                lines.size, regions.size, kept.size, detectMs, ocrMs, translateMs, inpaintMs, renderMs,
                System.currentTimeMillis() - tWall, promptTok, completionTok,
            ),
            PageAnalysis(detection.textMask, textRegions),
        )
    }

    /**
     * 單緒暖機：對 detector / OCR / 去字三個原生 session 各空跑一次推論，完成首次 lazy 初始化。
     * 建構後、放行跨頁併發前呼叫一次（見介面說明）。三者依序（單緒），best-effort（各自 catch）。
     */
    override fun warmUp() {
        EngineTrace.log("warmup.detector.enter")
        detector.warmUp()
        EngineTrace.log("warmup.ocr.enter")
        ocr.warmUp()
        EngineTrace.log("warmup.inpainter.enter")
        inpainter.warmUp()
        EngineTrace.log("warmup.done")
    }

    /**
     * 實驗 secondPass OCR（[OcrConfig.secondPass]）：弱行＝首輪 text 空白，或偵測分數 < 0.75 且只讀到 ≤2 字。
     * 把該行 bbox（外擴 6px）從原圖裁出、2× 放大，quad 同步平移＋縮放後走同一 [OcrEngine.recognize] 再讀一次；
     * 只在二輪「非空且（首輪空 或 更長）」時採用（連 direction 一起）。逐行 best-effort，失敗不影響主流程。
     */
    private suspend fun secondPassOcr(page: Bitmap, lines: List<TextLine>) {
        val weak = lines.filter { it.text.isBlank() || (it.score < 0.75f && it.text.length <= 2) }
        if (weak.isEmpty()) return
        EngineTrace.log("pipe.ocr.secondpass n=${weak.size}")
        for (line in weak) {
            try {
                val pad = 6f
                val xi = (line.quad.minOf { it.x } - pad).coerceAtLeast(0f).roundToInt()
                val yi = (line.quad.minOf { it.y } - pad).coerceAtLeast(0f).roundToInt()
                val w = ((line.quad.maxOf { it.x } + pad).roundToInt() - xi).coerceAtMost(page.width - xi)
                val h = ((line.quad.maxOf { it.y } + pad).roundToInt() - yi).coerceAtMost(page.height - yi)
                if (w < 8 || h < 8) continue
                val crop = Bitmap.createBitmap(page, xi, yi, w, h)
                val big = Bitmap.createScaledBitmap(crop, w * 2, h * 2, true)
                try {
                    val nl = TextLine(
                        line.quad.map { Pt((it.x - xi) * 2f, (it.y - yi) * 2f) },
                        line.score,
                    )
                    ocr.recognize(big, listOf(nl))
                    if (nl.text.isNotBlank() && (line.text.isBlank() || nl.text.length > line.text.length)) {
                        line.text = nl.text
                        line.direction = nl.direction
                    }
                } finally {
                    crop.recycle()
                    big.recycle()
                }
            } catch (t: Throwable) {
                Log.w(TAG, "OCR 二輪單行失敗（略過）: ${t.message}")
            }
        }
    }

    /**
     * 原文字色取樣：區 bbox ∩ seg 筆畫遮罩內的原圖像素，逐通道取中位數（中位數抗抗鋸齒邊緣的混色像素）。
     * 筆畫像素 < 8 → null（Renderer 退回 auto 色）。直方圖法（256 bins/通道），不建大清單。
     */
    private fun sampleTextColor(page: Bitmap, textMask: Bitmap, region: TextRegion): Int? {
        val w = page.width
        val h = page.height
        val x0 = region.x0.toInt().coerceIn(0, w - 1)
        val y0 = region.y0.toInt().coerceIn(0, h - 1)
        val x1 = region.x1.toInt().coerceIn(x0 + 1, w)
        val y1 = region.y1.toInt().coerceIn(y0 + 1, h)
        val bw = x1 - x0
        val bh = y1 - y0
        val px = IntArray(bw * bh)
        page.getPixels(px, 0, bw, x0, y0, bw, bh)
        val mp = IntArray(bw * bh)
        textMask.getPixels(mp, 0, bw, x0, y0, bw, bh)
        val hr = IntArray(256); val hg = IntArray(256); val hb = IntArray(256)
        var n = 0
        for (i in px.indices) {
            if ((mp[i] and 0xFF) <= 127) continue
            val p = px[i]
            hr[(p shr 16) and 0xFF]++; hg[(p shr 8) and 0xFF]++; hb[p and 0xFF]++; n++
        }
        if (n < 8) return null
        fun median(hist: IntArray): Int {
            var acc = 0
            val half = n / 2
            for (v in 0..255) { acc += hist[v]; if (acc > half) return v }
            return 255
        }
        return android.graphics.Color.rgb(median(hr), median(hg), median(hb))
    }

    /** 釋放 detector/ocr/inpainter 的原生 ONNX session（見類別說明的生命週期注意事項）。 */
    override fun close() {
        runCatching { detector.close() }
        runCatching { ocr.close() }
        runCatching { inpainter.close() }
    }

    companion object {
        private const val TAG = "Pipeline"

        /** fit 改短重翻的附加指示（只進該次呼叫的 system prompt，見 [LlmTranslator.translateDetailed]）。 */
        private const val FIT_SHORTER_INSTRUCTION =
            "IMPORTANT: The previous translations of the following lines were too long to fit inside their speech bubbles. " +
                "Translate each line into a much shorter, more concise version — cut every word that is not essential — " +
                "while keeping the core meaning."
    }
}
