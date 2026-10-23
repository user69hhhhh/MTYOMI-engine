package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * 日文／中文 OCR（PaddleOCR PP-OCRv6 rec → NCNN）——fork 新增的第三條 OCR 路徑，來源語言＝日文或中文
 * （簡／繁）且使用者選 v6 時啟用（[OcrConfig.engine]＝"paddle_v6_small"／"paddle_v6_medium"，選擇邏輯
 * 在 app 端；引擎只照 config 建）。單一模型同時讀日文＋簡中＋繁中＋英文（字典 18,708 字），漫畫與漫畫
 * （manhua）共用，不必依書換模型。模型檔不包進 APK：由 app 的「下載模型」選單抓到 extra_models 目錄、
 * 路徑經 [ModelSet.ocrPaddleV6Small]／[ModelSet.ocrPaddleV6Medium] 傳入（比照 LaMa 的 app 端解析模式）。
 *
 * 與 [PaddleOcr]（PP-OCRv3 阿拉伯文）的差異：
 *  - 轉換契約同樣是 pnnx 固定畫布 [CANVAS_W]×[TEXT_H]（320×48 逐寬實測：與 ONNX 輸出 maxdiff ~7e-05、
 *    解碼逐字一致，2026-10-23 桌面驗證）；輸出 [T]=40 步 × [CLASSES]=18,710（0=blank、1..18708＝字典、
 *    18709＝空白——PP-OCR CTC 的 use_space_char 尾字元，與桌面解碼對過）。
 *  - **精度分級**：small 的 .bin 是 fp16 儲存（10.6MB；桌面 NCNN 驗證解碼全對）；medium 的 fp16 儲存
 *    實測輸出 NaN（權重範圍溢位）→ medium 一律 fp32（76.5MB），兩者載入旗標都關 fp16 storage/arith
 *    （NCNN 對已存 fp16 的權重會自動上轉，pyncnn 預設選項實測可載）。
 *  - **直書**：偵測框高＞寬的行視為直書，裁切先逆時針轉 90° 再餵（桌面實測：直條轉 CCW 後逐字讀對、
 *    轉 CW 讀不出；Android Matrix 正角是順時針，故用 -90°），direction 標 "v"；其餘標 "h"。
 *  - 裁切、畫布、CTC 收合（[Ocr.ctcCollapse]）、信心門檻（[OcrConfig.minProb]）與 [PaddleOcr] 同。
 */
class PaddleOcrV6(
    modelPath: String,
    dictLines: List<String>,
    private val cfg: OcrConfig = OcrConfig(),
    tier: String = "small",
) : OcrEngine {

    override val backend: String = "NCNN-paddle-v6-$tier"
    private var ncnnHandle: Long = 0L

    /** class id → 字元：0=blank 佔位、1..18708＝[dictLines]、18709＝' '。 */
    private val classDict: List<String> = listOf("") + dictLines + listOf(" ")

    init {
        require(modelPath.endsWith(".param")) { "Paddle OCR v6 模型只支援 NCNN `.param`：$modelPath" }
        require(dictLines.size == DICT_SIZE) { "PP-OCRv6 字典應有 $DICT_SIZE 行，實際 ${dictLines.size}" }
        check(NcnnBackend.available) { "NCNN 原生庫未載入，無法 OCR" }
        val bin = modelPath.removeSuffix(".param") + ".bin"
        ncnnHandle = NcnnBackend.createNetEx(modelPath, bin, numThreads = 0, fp16Storage = false, fp16Arith = false)
        check(ncnnHandle != 0L) { "NCNN Paddle OCR v6 模型載入失敗：$modelPath（bin=$bin）" }
        EngineTrace.log("ocr.load $backend ${java.io.File(modelPath).name}")
        Log.i(TAG, "NCNN paddle-v6 ocr loaded $modelPath")
    }

    override suspend fun recognize(page: Bitmap, lines: List<TextLine>): Unit = coroutineScope {
        if (cfg.concurrent && lines.size > 1) {
            val sem = Semaphore(cfg.concurrency.coerceAtLeast(1))
            lines.map { line ->
                async(Dispatchers.Default) { sem.withPermit { recognizeOne(page, line) } }
            }.awaitAll()
        } else {
            for (line in lines) recognizeOne(page, line)
        }
    }

    override fun warmUp() {
        val blank = Bitmap.createBitmap(CANVAS_W, TEXT_H, Bitmap.Config.ARGB_8888)
        try {
            blank.eraseColor(Color.WHITE)
            infer(blank)
        } catch (t: Throwable) {
            Log.w(TAG, "Paddle OCR v6 暖機失敗：${t.message}")
        } finally {
            blank.recycle()
        }
    }

    /** 單行 OCR：bbox 裁切（直書先轉正）→ canvas → CTC → 填 text。只寫自己的 line。 */
    private fun recognizeOne(page: Bitmap, line: TextLine) {
        val pad = cfg.stripPad
        val x0 = (line.quad.minOf { it.x } - pad).roundToInt().coerceIn(0, page.width - 1)
        val y0 = (line.quad.minOf { it.y } - pad).roundToInt().coerceIn(0, page.height - 1)
        val x1 = (line.quad.maxOf { it.x } + pad).roundToInt().coerceIn(0, page.width - 1)
        val y1 = (line.quad.maxOf { it.y } + pad).roundToInt().coerceIn(0, page.height - 1)
        val bw = x1 - x0 + 1
        val bh = y1 - y0 + 1
        if (bw < 2 || bh < 2) return
        val raw = Bitmap.createBitmap(page, x0, y0, bw, bh)
        // 直書（高＞寬）：逆時針轉 90° 成橫書再讀（桌面實測 CCW 才讀得對）；direction 同步標記給排版用。
        val vertical = bh > bw
        line.direction = if (vertical) "v" else "h"
        val crop = if (vertical) rotate(raw, -90f) else raw
        try {
            val (text, prob) = infer(crop)
            if (prob >= cfg.minProb) line.text = text
        } catch (t: Throwable) {
            Log.w(TAG, "Paddle OCR v6 單行失敗：${t.message}")
        } finally {
            crop.recycle()
            if (crop !== raw) raw.recycle()
        }
    }

    private fun rotate(src: Bitmap, degrees: Float): Bitmap {
        val m = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    /**
     * 一條裁切 → (text, prob)：等比縮到高 [TEXT_H]（寬 > [CANVAS_W] 再等比縮到寬 [CANVAS_W]），
     * 貼白 canvas 左上，(x−127.5)/127.5、CHW——與 PaddleOCR 原前處理一致、也是桌面 ONNX/NCNN
     * 驗證時的餵法。輸出 [T] 步逐歩 argmax＋ln(p)，交 [Ocr.ctcCollapse] 收合。
     */
    private fun infer(crop: Bitmap): Pair<String, Float> {
        var sw = (crop.width * (TEXT_H.toFloat() / crop.height)).roundToInt().coerceAtLeast(1)
        var sh = TEXT_H
        if (sw > CANVAS_W) {
            sh = (sh * (CANVAS_W.toFloat() / sw)).roundToInt().coerceAtLeast(1)
            sw = CANVAS_W
        }
        val canvas = Bitmap.createBitmap(CANVAS_W, TEXT_H, Bitmap.Config.ARGB_8888)
        canvas.eraseColor(Color.WHITE)
        val scaled = Bitmap.createScaledBitmap(crop, sw, sh, true)
        Canvas(canvas).drawBitmap(scaled, 0f, 0f, Paint(Paint.FILTER_BITMAP_FLAG))
        if (scaled !== crop) scaled.recycle()
        val px = IntArray(CANVAS_W * TEXT_H)
        canvas.getPixels(px, 0, CANVAS_W, 0, 0, CANVAS_W, TEXT_H)
        canvas.recycle()

        val area = CANVAS_W * TEXT_H
        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = px[i]
            chw[i] = (((p shr 16) and 0xFF) - 127.5f) / 127.5f
            chw[area + i] = (((p shr 8) and 0xFF) - 127.5f) / 127.5f
            chw[2 * area + i] = ((p and 0xFF) - 127.5f) / 127.5f
        }
        val out = FloatArray(T * CLASSES)
        val rc = NcnnBackend.extract(ncnnHandle, chw, CANVAS_W, TEXT_H, 3, arrayOf(OUT_BLOB), arrayOf(out))
        check(rc == 0) { "NCNN paddle OCR v6 失敗 rc=$rc" }
        val idx = IntArray(T)
        val logp = FloatArray(T)
        for (t in 0 until T) {
            val base = t * CLASSES
            var best = 0
            var bestV = out[base]
            for (c in 1 until CLASSES) {
                val v = out[base + c]
                if (v > bestV) {
                    bestV = v
                    best = c
                }
            }
            idx[t] = best
            logp[t] = ln(max(bestV, 1e-6f))
        }
        return Ocr.ctcCollapse(idx, logp, T, classDict)
    }

    override fun close() {
        if (ncnnHandle != 0L) {
            NcnnBackend.releaseNet(ncnnHandle)
            ncnnHandle = 0L
        }
    }

    companion object {
        private const val TAG = "PaddleOcrV6"
        private const val TEXT_H = 48    // PP-OCRv6 rec 固定輸入高
        private const val CANVAS_W = 320 // ★ pnnx 轉換寬度：模型只在此寬度正確（同 PaddleOcr 的教訓），不可改值不重轉
        private const val T = 40         // CANVAS_W / 8（桌面實測輸出形狀 (1, 40, 18710)）
        private const val CLASSES = 18710 // blank + 18,708 字典 + 空白
        private const val DICT_SIZE = 18708
        private const val OUT_BLOB = "out0"
    }
}
