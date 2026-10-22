package li.joye.yakuyomi.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import java.nio.FloatBuffer

/**
 * LaMa 去字（max quality 門別）——走 **ONNX Runtime**，不走 NCNN。
 *
 * 為何另開 runtime：LaMa 的 FFC 層核心是 FFT，NCNN 沒有 FFT/DFT layer、pnnx 轉檔會靜默丟 op（2026-10 MTYOMI
 * 實測三路全滅：動態圖 bad_alloc、固定 512 圖 pnnx OOM、DFT 單 op 被 ignore）。ORT 直接跑現成的
 * `lama_fp32.onnx`（Carve/LaMa-ONNX，big-lama 權重）：雙輸入 `image`[1,3,H,W]∈[0,1]、`mask`[1,1,H,W]∈{0,1}，
 * 輸出 [1,3,H,W]∈[0,1]。圖是動態尺寸，但餵固定 [TILE] 方磚（與 [Inpainter] 的逐區磚路徑配合：
 * 逐區近原生解析度、避開整頁縮小必糊）。
 *
 * 代價（使用者知情下自選的品質門別）：模型 ~208MB（fp32）、CPU 每磚秒級。session 執行緒安全（ORT 保證），
 * lazy 建一次、全頁共用；建不起來回 false → [Inpainter] 退回 AOT-GAN [OG]。
 */
class LamaOrtInpainter(private val modelPath: String) : AutoCloseable {

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var failed = false

    /** lazy 建 session（冪等）；成功回 true。失敗標記後不再重試（每頁都試只會刷 log）。 */
    @Synchronized
    fun ensureSession(): Boolean {
        if (session != null) return true
        if (failed) return false
        return try {
            val e = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 8))
                setInterOpNumThreads(1)
            }
            session = e.createSession(modelPath, opts)
            env = e
            Log.i(TAG, "LaMa ORT session 建好：$modelPath（inputs=${session?.inputNames}）")
            true
        } catch (t: Throwable) {
            Log.w(TAG, "LaMa ORT session 建立失敗：${t.message}")
            failed = true
            false
        }
    }

    /** 暖機：建 session + 對空白磚跑一次（首次推論的 lazy 初始化在暖機緒完成）。best-effort。 */
    fun warmUp() {
        if (!ensureSession()) return
        val bmp = Bitmap.createBitmap(TILE, TILE, Bitmap.Config.ARGB_8888)
        val mask = Bitmap.createBitmap(TILE, TILE, Bitmap.Config.ARGB_8888)
        try {
            runTile(bmp, mask)?.recycle()
        } catch (t: Throwable) {
            Log.w(TAG, "LaMa 暖機失敗：${t.message}")
        } finally {
            bmp.recycle(); mask.recycle()
        }
    }

    /**
     * 單磚推論：裁切＋遮罩縮到 [TILE] 方磚 → ORT → 放回裁切尺寸回傳。失敗回 null（呼叫端該區退回 AOT 結果）。
     * image 餵**原圖**（不歸零洞——LaMa 是 image+mask 雙輸入、模型自己處理洞）。
     */
    fun runTile(crop: Bitmap, cropMask: Bitmap): Bitmap? {
        if (!ensureSession()) return null
        val e = env ?: return null
        val s = session ?: return null
        val t = TILE
        val imgScaled = Bitmap.createScaledBitmap(crop, t, t, true)
        val maskScaled = Bitmap.createScaledBitmap(cropMask, t, t, false)
        return try {
            val area = t * t
            val ip = IntArray(area); imgScaled.getPixels(ip, 0, t, 0, 0, t, t)
            val mp = IntArray(area); maskScaled.getPixels(mp, 0, t, 0, 0, t, t)
            val img = FloatArray(3 * area)
            val msk = FloatArray(area)
            for (i in 0 until area) {
                val p = ip[i]
                img[i] = ((p shr 16) and 0xFF) / 255f
                img[area + i] = ((p shr 8) and 0xFF) / 255f
                img[2 * area + i] = (p and 0xFF) / 255f
                msk[i] = if ((mp[i] and 0xFF) > 127) 1f else 0f
            }
            val imgTensor = OnnxTensor.createTensor(e, FloatBuffer.wrap(img), longArrayOf(1, 3, t.toLong(), t.toLong()))
            val mskTensor = OnnxTensor.createTensor(e, FloatBuffer.wrap(msk), longArrayOf(1, 1, t.toLong(), t.toLong()))
            val outBitmap = imgTensor.use { it ->
                mskTensor.use { mk ->
                    s.run(mapOf("image" to it, "mask" to mk)).use { result ->
                        @Suppress("UNCHECKED_CAST")
                        val out = result[0].value as Array<Array<Array<FloatArray>>>
                        arrToBitmap(out[0], t)
                    }
                }
            }
            val back = Bitmap.createScaledBitmap(outBitmap, crop.width, crop.height, true)
            if (back !== outBitmap) outBitmap.recycle()
            back
        } catch (e2: Throwable) {
            Log.w(TAG, "LaMa 磚推論失敗：${e2.message}"); null
        } finally {
            imgScaled.recycle(); maskScaled.recycle()
        }
    }

    /** ORT 輸出 [3][H][W]∈[0,1] → Bitmap。 */
    private fun arrToBitmap(chw: Array<Array<FloatArray>>, n: Int): Bitmap {
        val px = IntArray(n * n)
        for (y in 0 until n) {
            val row = y * n
            for (x in 0 until n) {
                val r = (chw[0][y][x] * 255f).toInt().coerceIn(0, 255)
                val g = (chw[1][y][x] * 255f).toInt().coerceIn(0, 255)
                val b = (chw[2][y][x] * 255f).toInt().coerceIn(0, 255)
                px[row + x] = Color.rgb(r, g, b)
            }
        }
        return Bitmap.createBitmap(px, n, n, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        runCatching { session?.close() }
        session = null
        env = null
    }

    companion object {
        private const val TAG = "LamaOrt"
        private const val TILE = 512  // LaMa 方磚邊長（逐區裁切縮放進出；與 Inpainter 的磚路徑共用此尺寸概念）
    }
}
