package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
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
 * 阿拉伯文 OCR（PaddleOCR PP-OCRv3 arabic rec → NCNN）——fork 新增的第二條 OCR 路徑，來源語言＝阿拉伯文且
 * 使用者選 paddle 時啟用（[OcrConfig.engine]＝"paddle"，選擇邏輯在 app 端；引擎只照 config 建）。
 *
 * 與主路徑 [Ocr]（48px CTC transformer＋位置編碼第二輸入、JNI 內 argmax）的差異：
 *  - 本模型（SVTR_LCNet＋CTC）經 pnnx 轉 NCNN 後**只在轉換寬度 [CANVAS_W] 正確**：寬度相關的常數被 pnnx
 *    烤進圖裡，其他寬度輸出直接亂碼（逐寬實測：320 與 ONNX 逐字一致、480/640 亂碼）。故每一條一律貼到
 *    [CANVAS_W]×[TEXT_H] 的白底 canvas 上、以原生比例縮放餵入（寬超過 canvas 才等比縮小、高隨之變矮），
 *    絕不以其他寬度呼叫模型。
 *  - 精度用 fp32 param/bin（fp16 實測 maxdiff 0.98、argmax 會翻——同主路徑 transformer 吃 fp16 的教訓，不用）。
 *  - 推論走通用抽取（[NcnnBackend.extract]，out0＝[T]×[CLASSES] 機率、尾層已 Softmax）；argmax＋logp＋CTC
 *    收合在 Kotlin（共用 [Ocr.ctcCollapse]）。字典以 class id 索引：0=blank、1..161＝阿拉伯字典、162＝空白
 *    （PP-OCRv3 arabic 的 use_space_char 尾字元）。
 *  - 裁切走 quad 的 axis-aligned bbox（＋[OcrConfig.stripPad] 外擴）：阿拉伯文橫書、氣泡框近矩形；不沿用主
 *    路徑的透視 warp／bicubic／unsharp——那些是為直書日文小假名調的。解码結果為**邏輯順序**（模型訓練即如此，
 *    與 ONNX 逐字對過），direction 一律 "h"。
 *  - 信心門檻沿用 [OcrConfig.minProb]；SFX 跳過（ignoreBubble）不移植（那是日文彩色 SFX 的判準）。
 */
class PaddleOcr(
    modelPath: String,
    dictLines: List<String>,
    private val cfg: OcrConfig = OcrConfig(),
) : OcrEngine {

    override val backend: String = "NCNN-paddle-ar"
    private var ncnnHandle: Long = 0L

    /** class id → 字元：0=blank 佔位、1..161＝[dictLines]、162＝' '。 */
    private val classDict: List<String> = listOf("") + dictLines + listOf(" ")

    init {
        require(modelPath.endsWith(".param")) { "Paddle OCR 模型只支援 NCNN `.param`：$modelPath" }
        require(dictLines.size == DICT_SIZE) { "阿拉伯字典應有 $DICT_SIZE 行，實際 ${dictLines.size}" }
        check(NcnnBackend.available) { "NCNN 原生庫未載入，無法 OCR" }
        val bin = modelPath.removeSuffix(".param") + ".bin"
        // fp16 storage/arith 全關：權重與中間值都留 fp32（本模型的 fp16 實測會翻 argmax，見類別註解）。
        ncnnHandle = NcnnBackend.createNetEx(modelPath, bin, numThreads = 0, fp16Storage = false, fp16Arith = false)
        check(ncnnHandle != 0L) { "NCNN Paddle OCR 模型載入失敗：$modelPath（bin=$bin）" }
        EngineTrace.log("ocr.load $backend ${java.io.File(modelPath).name}")
        Log.i(TAG, "NCNN paddle-ar ocr loaded $modelPath")
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
            Log.w(TAG, "Paddle OCR 暖機失敗：${t.message}")
        } finally {
            blank.recycle()
        }
    }

    /** 單行 OCR：bbox 裁切 → canvas → CTC → 填 text。只寫自己的 line；[NcnnBackend.extract] 自帶全域鎖，可並發呼叫。 */
    private fun recognizeOne(page: Bitmap, line: TextLine) {
        line.direction = "h" // 阿拉伯文只有橫書
        val pad = cfg.stripPad
        val x0 = (line.quad.minOf { it.x } - pad).roundToInt().coerceIn(0, page.width - 1)
        val y0 = (line.quad.minOf { it.y } - pad).roundToInt().coerceIn(0, page.height - 1)
        val x1 = (line.quad.maxOf { it.x } + pad).roundToInt().coerceIn(0, page.width - 1)
        val y1 = (line.quad.maxOf { it.y } + pad).roundToInt().coerceIn(0, page.height - 1)
        val bw = x1 - x0 + 1
        val bh = y1 - y0 + 1
        if (bw < 2 || bh < 2) return
        val crop = Bitmap.createBitmap(page, x0, y0, bw, bh)
        try {
            val (text, prob) = infer(crop)
            if (prob >= cfg.minProb) line.text = text
        } catch (t: Throwable) {
            Log.w(TAG, "Paddle OCR 單行失敗：${t.message}")
        } finally {
            crop.recycle()
        }
    }

    /**
     * 一條裁切 → (text, prob)：等比縮到高 [TEXT_H]（寬 > [CANVAS_W] 再等比縮到寬 [CANVAS_W]、高隨之變矮），
     * 貼白 canvas 左上（其餘留白＝CTC 的 blank context），(x−127.5)/127.5、CHW——與 PaddleOCR 原前處理一致、
     * 也是本模型 ONNX 驗證時的餵法。輸出 [T] 步逐歩 argmax＋ln(p)，交 [Ocr.ctcCollapse] 收合。
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
        check(rc == 0) { "NCNN paddle OCR 失敗 rc=$rc" }
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
        private const val TAG = "PaddleOcr"
        private const val TEXT_H = 48    // PP-OCRv3 rec 固定輸入高
        private const val CANVAS_W = 320 // ★ pnnx 轉換寬度：模型只在此寬度正確（見類別註解），不可改值不重轉
        private const val T = 40         // CANVAS_W / 8（SVTR 三段下採樣後的時間步數）
        private const val CLASSES = 163  // blank + 161 字典 + 空白
        private const val DICT_SIZE = 161
        private const val OUT_BLOB = "out0"
    }
}
