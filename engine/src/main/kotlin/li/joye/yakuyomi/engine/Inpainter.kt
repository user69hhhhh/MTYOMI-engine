package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import kotlin.math.roundToInt
import kotlinx.coroutines.coroutineScope

/**
 * 去字（text removal）。真機 A/B 定案＝兩門別，皆純 NCNN（`.param`/`.bin`）：
 *  - **boxfill（快速去字）**：逐區把去字遮罩像素平塗成背景色。瞬間、不跑模型；乾淨白泡完美、壓畫面是平色塊。
 *  - **aot（AI 去字·預設）**：AOT-GAN（m-i-t 漫畫權重）整頁一次縮到 [InpainterConfig.tileSize]（768）重建背景、全區重建。
 *    全卷積 → 任意尺寸；768＝畫質/記憶體/藏在翻譯下（§8 去字‖翻譯重疊）的甜蜜點。
 *
 * AOT I/O 契約（對齊 parity/inpaint_parity.py）：img∈[-1,1] 且洞歸零（m-i-t `img*(1-mask)`）、mask∈{0,1}(1=擦)、輸出∈[-1,1]。
 * LaMa（整頁縮 512 必糊）與 AOT 逐格（原生解析度·CPU 太貴）皆已退役移除；GPU/Vulkan 實測算不對 AOT-GAN（見 memory ncnn-vulkan-fp16）。
 */
class Inpainter(
    modelPath: String,
    private val cfg: InpainterConfig = InpainterConfig(),
    private val miganModelPath: String? = null,
    private val lamaOnnxPath: String? = null,
) : AutoCloseable {

    private var ncnnHandle: Long = 0L
    private var miganHandle: Long = 0L
    private var lamaOrt: LamaOrtInpainter? = null
    /** 實際生效的後端；無 adb 時由呼叫端寫進 log/圖確認。 */
    val ep: String = "NCNN-CPU"

    init {
        check(modelPath.endsWith(".param")) { "去字需 NCNN `.param` 模型（AOT-GAN）：$modelPath" }
        check(NcnnBackend.available) { "NCNN 原生庫未載入，無法去字" }
        val bin = modelPath.removeSuffix(".param") + ".bin"
        ncnnHandle = NcnnBackend.createNet(modelPath, bin)
        check(ncnnHandle != 0L) { "NCNN AOT 模型載入失敗：$modelPath" }
    }

    suspend fun inpaint(page: Bitmap, regions: List<TextRegion>, textMask: Bitmap): Bitmap = coroutineScope {
        val w = page.width
        val h = page.height
        val result = page.copy(Bitmap.Config.ARGB_8888, true)
        // seg 細筆畫遮罩 ∩ 保留區 bbox（外擴 bboxPad）再膨脹：只動筆畫、限制在翻譯過的區（SFX/未譯留原圖，§11）。
        val maskPx = buildSegMask(regions, textMask, w, h)

        if (cfg.method == "boxfill") {
            // 逐區平塗背景色：白泡乾淨無殘留、忙碌區是平色塊（要品質用 aot）。
            val px = IntArray(w * h); result.getPixels(px, 0, w, 0, 0, w, h)
            val tightPx = IntArray(w * h); textMask.getPixels(tightPx, 0, w, 0, 0, w, h)
            for (r in regions) {
                val s = bgStats(px, tightPx, r, w, h)
                r.onArt = false; r.dbgStd = s.std; r.dbgWhite = s.meanLum // dbg 值給 sandbox 去背比較標框
                flatFill(result, maskPx, r, s.color, cfg.bboxPad, w, h)
            }
            return@coroutineScope result
        }

        // 模型重建（aot 整頁 [OG] / lama 逐區磚）：結果合成進 result 的遮罩像素；標 onArt 讓 Renderer 給黑字粗白邊。
        regions.forEach { it.onArt = true }
        val maskBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        maskBmp.setPixels(maskPx, 0, w, 0, 0, w, h)
        val tiledDone = when (cfg.method) {
            "migan" -> {
                val net = miganNet()
                net != 0L && runTiledRegions(page, maskBmp, regions, result, maskPx, w, h) { c, m -> runMiganTile(net, c, m) }
            }
            "lama" -> {
                val ort = lamaOrtRunner()
                ort != null && runTiledRegions(page, maskBmp, regions, result, maskPx, w, h) { c, m -> ort.runTile(c, m) }
            }
            else -> false
        }
        if (!tiledDone) {
            runWholeAot(page, maskBmp, w, h)?.let { compositePixels(result, maskPx, it) }
        }
        maskBmp.recycle()
        // 平坦區路由：背景夠平（std 低＝純色泡泡）的區，AOT 重建反而可能留影/糊化 → 改平塗原圖取樣的背景色
        // 蓋掉該區遮罩像素（像素級乾淨），並撤 onArt（平坦底走 Renderer auto 配色、不需粗白邊）。
        if (cfg.flatFillFlatRegions) {
            val origPx = IntArray(w * h); page.getPixels(origPx, 0, w, 0, 0, w, h)
            val tightPx = IntArray(w * h); textMask.getPixels(tightPx, 0, w, 0, 0, w, h)
            for (r in regions) {
                val s = bgStats(origPx, tightPx, r, w, h)
                r.dbgStd = s.std; r.dbgWhite = s.meanLum
                if (s.std < cfg.flatMaxStd) {
                    flatFill(result, maskPx, r, s.color, cfg.bboxPad, w, h)
                    r.onArt = false
                }
            }
        }
        result
    }

    /**
     * 暖機：aot 方法對空白小圖跑一次 NCNN AOT session（首次 lazy 初始化在單緒完成）。
     * boxfill 只平塗、不跑該 session → 無需暖（也不會冷撞）。併發翻多頁前先呼叫一次。
     */
    fun warmUp() {
        if (cfg.method == "boxfill") return
        if (cfg.method == "lama") { runCatching { lamaOrtRunner()?.warmUp() }; return }
        if (cfg.method == "migan") { miganNet(); return }
        val w = 64
        val h = 64
        val page = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        try {
            runWholeAot(page, mask, w, h)
        } catch (t: Throwable) {
            Log.w(TAG, "去字暖機失敗：${t.message}")
        } finally {
            page.recycle()
            mask.recycle()
        }
    }

    /** 去字遮罩 Bitmap（白＝要去字）。給重繪素材/視覺化用；與 inpaint 同一份 seg 細遮罩。 */
    fun buildMask(page: Bitmap, regions: List<TextRegion>, textMask: Bitmap): Bitmap {
        val w = page.width; val h = page.height
        val maskPx = buildSegMask(regions, textMask, w, h)
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply { setPixels(maskPx, 0, w, 0, 0, w, h) }
    }

    /**
     * seg 細遮罩＝seg 細筆畫 ∩ 已保留區的「區域 bbox 矩形（外擴 bboxPad）」（allow），再膨脹。回傳 ARGB 像素（白＝要去字）。
     * ★ 用 bbox 矩形(非緊的文字行框)＝涵蓋漢字旁的注音假名；pad 再外擴涵蓋貼 bbox 邊界的假名（桌面 auto_diag.py 實證）。
     */
    private fun buildSegMask(regions: List<TextRegion>, textMask: Bitmap, w: Int, h: Int): IntArray {
        val pad = cfg.bboxPad
        val allow = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(allow).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
            for (region in regions) drawRect(region.x0 - pad, region.y0 - pad, region.x1 + pad, region.y1 + pad, p)
        }
        val mask = IntArray(w * h)
        allow.getPixels(mask, 0, w, 0, 0, w, h)
        allow.recycle()
        val seg = IntArray(w * h)
        textMask.getPixels(seg, 0, w, 0, 0, w, h)
        for (i in mask.indices) {
            mask[i] = if ((mask[i] and 0xFF) > 127 && (seg[i] and 0xFF) > 127) Color.WHITE else Color.BLACK
        }
        dilate(mask, w, h, (cfg.maskDilate / 2f).roundToInt().coerceAtLeast(1))
        return mask
    }

    /** 二值遮罩可分離膨脹（先橫後縱 max-filter），radius 像素。覆蓋筆畫抗鋸齒邊緣、給去字餘裕。 */
    private fun dilate(px: IntArray, w: Int, h: Int, radius: Int) {
        if (radius <= 0) return
        val tmp = IntArray(px.size)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var on = false; var k = -radius
                while (k <= radius) { val xx = x + k; if (xx in 0 until w && (px[row + xx] and 0xFF) > 127) { on = true; break }; k++ }
                tmp[row + x] = if (on) Color.WHITE else Color.BLACK
            }
        }
        for (x in 0 until w) {
            for (y in 0 until h) {
                var on = false; var k = -radius
                while (k <= radius) { val yy = y + k; if (yy in 0 until h && (tmp[yy * w + x] and 0xFF) > 127) { on = true; break }; k++ }
                px[y * w + x] = if (on) Color.WHITE else Color.BLACK
            }
        }
    }

    private class BgStat(val meanLum: Float, val std: Float, val color: Int)

    /**
     * 區 bbox 內「非文字(背景)」像素的亮度均值+std+平均色。tightPx＝未膨脹 textMask（量得到筆畫間的白）。
     * boxfill 用 [BgStat.color] 平塗；std/meanLum 只給 sandbox 去背比較標框（對照用）。用行框多邊形局部遮罩避開軸對齊 bbox 角落雜訊。
     */
    private fun bgStats(px: IntArray, tightPx: IntArray, region: TextRegion, w: Int, h: Int): BgStat {
        val x0 = region.x0.toInt().coerceIn(0, w - 1)
        val y0 = region.y0.toInt().coerceIn(0, h - 1)
        val x1 = region.x1.toInt().coerceIn(x0 + 1, w)
        val y1 = region.y1.toInt().coerceIn(y0 + 1, h)
        val bw = x1 - x0; val bh = y1 - y0
        val qmBmp = Bitmap.createBitmap(bw, bh, Bitmap.Config.ARGB_8888)
        Canvas(qmBmp).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
            for (line in region.lines) {
                val q = line.quad
                if (q.size < 4) continue
                val path = android.graphics.Path().apply {
                    moveTo(q[0].x - x0, q[0].y - y0)
                    for (i in 1..3) lineTo(q[i].x - x0, q[i].y - y0)
                    close()
                }
                drawPath(path, p)
            }
        }
        val qm = IntArray(bw * bh)
        qmBmp.getPixels(qm, 0, bw, 0, 0, bw, bh)
        qmBmp.recycle()
        var n = 0; var sl = 0.0; var sl2 = 0.0; var sr = 0L; var sg = 0L; var sb = 0L
        for (y in 0 until bh) {
            for (x in 0 until bw) {
                if ((qm[y * bw + x] and 0xFF) <= 127) continue
                val gi = (y0 + y) * w + (x0 + x)
                if ((tightPx[gi] and 0xFF) > 127) continue // 文字像素
                val p = px[gi]
                val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
                val lum = 0.299 * r + 0.587 * g + 0.114 * b
                sl += lum; sl2 += lum * lum; sr += r; sg += g; sb += b; n++
            }
        }
        if (n < 16) return BgStat(255f, 0f, Color.WHITE)
        val mean = sl / n
        val std = kotlin.math.sqrt((sl2 / n - mean * mean).coerceAtLeast(0.0))
        return BgStat(mean.toFloat(), std.toFloat(), Color.rgb((sr / n).toInt(), (sg / n).toInt(), (sb / n).toInt()))
    }

    /** 白泡去字＝把區域 bbox(外擴 pad)內的去字遮罩像素直接平塗成背景色。均勻白泡保證無殘留。 */
    private fun flatFill(result: Bitmap, maskPx: IntArray, region: TextRegion, color: Int, pad: Int, w: Int, h: Int) {
        val x0 = (region.x0.toInt() - pad).coerceIn(0, w - 1)
        val y0 = (region.y0.toInt() - pad).coerceIn(0, h - 1)
        val x1 = (region.x1.toInt() + pad).coerceIn(x0 + 1, w)
        val y1 = (region.y1.toInt() + pad).coerceIn(y0 + 1, h)
        val bw = x1 - x0; val bh = y1 - y0
        val sub = IntArray(bw * bh)
        result.getPixels(sub, 0, bw, x0, y0, bw, bh)
        for (y in 0 until bh) {
            val mrow = (y0 + y) * w + x0
            val row = y * bw
            for (x in 0 until bw) if ((maskPx[mrow + x] and 0xFF) > 127) sub[row + x] = color
        }
        result.setPixels(sub, 0, bw, x0, y0, bw, bh)
    }

    private class WinOut(val x0: Int, val y0: Int, val ww: Int, val wh: Int, val px: IntArray)

    /**
     * 整頁跑一次 NCNN AOT-GAN：整張縮到方形 [cfg.tileSize] → 推論 → 放回原尺寸。只讀 page/maskBmp ⇒ 併發安全。
     * NCNN net 固定方形輸入、同尺寸復用安全（跨尺寸 reuse 會崩，故一律整頁 tileSize）。
     */
    private fun runWholeAot(page: Bitmap, maskBmp: Bitmap, w: Int, h: Int): WinOut? {
        val t = cfg.tileSize
        val imgScaled = Bitmap.createScaledBitmap(page, t, t, true)
        val maskScaled = Bitmap.createScaledBitmap(maskBmp, t, t, false)
        return try {
            val imgChw = aotImageChw(imgScaled, maskScaled, t)
            val maskArr = maskArr(maskScaled, t)
            val out = FloatArray(3 * t * t)
            val rc = NcnnBackend.inpaintAot(ncnnHandle, imgChw, maskArr, t, out)
            check(rc == 0) { "NCNN AOT 推論失敗 rc=$rc" }
            val resScaled = aotArrToBitmap(out, t)
            val resWin = Bitmap.createScaledBitmap(resScaled, w, h, true)
            val px = IntArray(w * h)
            resWin.getPixels(px, 0, w, 0, 0, w, h)
            if (resWin !== resScaled) resScaled.recycle()
            resWin.recycle()
            WinOut(0, 0, w, h, px)
        } catch (t2: Throwable) {
            Log.w(TAG, "AOT 去字失敗：${t2.message}"); null
        } finally {
            if (imgScaled !== page) imgScaled.recycle()
            if (maskScaled !== maskBmp) maskScaled.recycle()
        }
    }

    /** AOT 影像的裸 NCHW 陣列 [3*n*n]：RGB→[-1,1]，遮罩處歸零（m-i-t `img*(1-mask)`）。 */
    private fun aotImageChw(imgBmp: Bitmap, maskBmp: Bitmap, n: Int): FloatArray {
        val area = n * n
        val px = IntArray(area); imgBmp.getPixels(px, 0, n, 0, 0, n, n)
        val mp = IntArray(area); maskBmp.getPixels(mp, 0, n, 0, 0, n, n)
        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            if ((mp[i] and 0xFF) > 127) continue // 洞＝0
            val p = px[i]
            chw[i] = ((p shr 16) and 0xFF) / 127.5f - 1f
            chw[area + i] = ((p shr 8) and 0xFF) / 127.5f - 1f
            chw[2 * area + i] = (p and 0xFF) / 127.5f - 1f
        }
        return chw
    }

    /** 遮罩的裸陣列 [n*n]（1=擦）。 */
    private fun maskArr(bmp: Bitmap, n: Int): FloatArray {
        val px = IntArray(n * n); bmp.getPixels(px, 0, n, 0, 0, n, n)
        val m = FloatArray(n * n)
        for (i in px.indices) m[i] = if ((px[i] and 0xFF) > 127) 1f else 0f
        return m
    }

    /** AOT 輸出陣列 [3*n*n]∈[-1,1] → Bitmap（(x+1)*127.5）。 */
    private fun aotArrToBitmap(arr: FloatArray, n: Int): Bitmap {
        val area = n * n
        val px = IntArray(area)
        for (i in 0 until area) {
            val r = ((arr[i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val g = ((arr[area + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val b = ((arr[2 * area + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, n, n, Bitmap.Config.ARGB_8888)
    }

    /** 把 AOT 輸出貼回 result，只換遮罩內像素（序列呼叫、寫入安全）。 */
    private fun compositePixels(result: Bitmap, maskPx: IntArray, o: WinOut) {
        val w = result.width
        val cur = IntArray(o.ww * o.wh)
        result.getPixels(cur, 0, o.ww, o.x0, o.y0, o.ww, o.wh)
        for (y in 0 until o.wh) {
            val maskRow = (o.y0 + y) * w + o.x0
            val row = y * o.ww
            for (x in 0 until o.ww) if ((maskPx[maskRow + x] and 0xFF) > 127) cur[row + x] = o.px[row + x]
        }
        result.setPixels(cur, 0, o.ww, o.x0, o.y0, o.ww, o.wh)
    }

    /** MI-GAN net（lazy）：method=migan 且模型在才載；載不到回 0 → 呼叫端退回 AOT [OG]。fp32（品質路徑、不冒 fp16 風險）。 */
    private fun miganNet(): Long {
        if (miganHandle != 0L) return miganHandle
        val path = miganModelPath ?: return 0L
        if (!path.endsWith(".param")) return 0L
        val bin = path.removeSuffix(".param") + ".bin"
        miganHandle = runCatching {
            NcnnBackend.createNetEx(path, bin, numThreads = 0, fp16Storage = false, fp16Arith = false)
        }.getOrDefault(0L)
        if (miganHandle == 0L) Log.w(TAG, "MI-GAN 模型載入失敗：$path")
        return miganHandle
    }

    /** LaMa ORT runner（lazy）：method=lama 且 ONNX 模型在才建；建不起來回 null → 退回 AOT [OG]。 */
    private fun lamaOrtRunner(): LamaOrtInpainter? {
        lamaOrt?.let { return it }
        val path = lamaOnnxPath ?: return null
        if (!path.endsWith(".onnx")) return null
        val ort = LamaOrtInpainter(path)
        lamaOrt = ort
        return ort
    }

    /**
     * 逐區磚去字（migan NCNN / lama ORT 共用骨架）：每區取 bbox 外擴 margin 的裁切 → [tile] 推論（內部縮到
     * 固定方磚再放回裁切尺寸）→ 只合成該區遮罩像素（[compositePixels] 的窗口版）。逐區近原生解析度＝避開
     * 「整頁縮小必糊」（上游當年退役整頁 LaMa 的原因）；代價是 CPU 時間（品質門別、使用者自選）。
     * 全部磚失敗回 false（呼叫端整頁退 AOT）。
     */
    private fun runTiledRegions(
        page: Bitmap, maskBmp: Bitmap, regions: List<TextRegion>, result: Bitmap,
        maskPx: IntArray, w: Int, h: Int, tile: (Bitmap, Bitmap) -> Bitmap?,
    ): Boolean {
        var any = false
        for (r in regions) {
            val bw = (r.x1 - r.x0).toInt(); val bh = (r.y1 - r.y0).toInt()
            val margin = maxOf(48, maxOf(bw, bh) / 2)
            val cx0 = (r.x0.toInt() - margin).coerceIn(0, w - 1)
            val cy0 = (r.y0.toInt() - margin).coerceIn(0, h - 1)
            val cx1 = (r.x1.toInt() + margin).coerceIn(cx0 + 1, w)
            val cy1 = (r.y1.toInt() + margin).coerceIn(cy0 + 1, h)
            val cw = cx1 - cx0; val ch = cy1 - cy0
            val crop = Bitmap.createBitmap(page, cx0, cy0, cw, ch)
            val cropMask = Bitmap.createBitmap(maskBmp, cx0, cy0, cw, ch)
            val out = tile(crop, cropMask)
            crop.recycle(); cropMask.recycle()
            if (out != null) {
                val px = IntArray(cw * ch)
                out.getPixels(px, 0, cw, 0, 0, cw, ch)
                out.recycle()
                compositePixels(result, maskPx, WinOut(cx0, cy0, cw, ch, px))
                any = true
            }
        }
        if (!any) Log.w(TAG, "逐區磚全失敗，退回 AOT")
        return any
    }

    /**
     * 單磚 MI-GAN 推論：裁切＋遮罩縮到 [MIGAN_TILE] 方磚 → 通用 extract → 放回裁切尺寸。失敗回 null。
     * I/O 契約（轉檔定案）：in0＝4 通道 CHW（RGB∈[0,1] 原圖、mask∈{0,1}）、out0＝3 通道 CHW∈[0,1]。
     */
    private fun runMiganTile(net: Long, crop: Bitmap, cropMask: Bitmap): Bitmap? {
        val t = MIGAN_TILE
        val imgScaled = Bitmap.createScaledBitmap(crop, t, t, true)
        val maskScaled = Bitmap.createScaledBitmap(cropMask, t, t, false)
        return try {
            val area = t * t
            val ip = IntArray(area); imgScaled.getPixels(ip, 0, t, 0, 0, t, t)
            val mp = IntArray(area); maskScaled.getPixels(mp, 0, t, 0, 0, t, t)
            val chw = FloatArray(4 * area)
            for (i in 0 until area) {
                val p = ip[i]
                chw[i] = ((p shr 16) and 0xFF) / 255f
                chw[area + i] = ((p shr 8) and 0xFF) / 255f
                chw[2 * area + i] = (p and 0xFF) / 255f
                chw[3 * area + i] = if ((mp[i] and 0xFF) > 127) 1f else 0f
            }
            val out = FloatArray(3 * area)
            val rc = NcnnBackend.extract(net, chw, t, t, 4, arrayOf("out0"), arrayOf(out))
            check(rc == 0) { "NCNN MI-GAN 推論失敗 rc=$rc" }
            val resScaled = miganArrToBitmap(out, t)
            val back = Bitmap.createScaledBitmap(resScaled, crop.width, crop.height, true)
            if (back !== resScaled) resScaled.recycle()
            back
        } catch (e: Throwable) {
            Log.w(TAG, "MI-GAN 磚失敗：${e.message}"); null
        } finally {
            imgScaled.recycle(); maskScaled.recycle()
        }
    }

    /** MI-GAN 輸出陣列 [3*n*n]∈[0,1] → Bitmap。 */
    private fun miganArrToBitmap(arr: FloatArray, n: Int): Bitmap {
        val area = n * n
        val px = IntArray(area)
        for (i in 0 until area) {
            val r = (arr[i] * 255f).toInt().coerceIn(0, 255)
            val g = (arr[area + i] * 255f).toInt().coerceIn(0, 255)
            val b = (arr[2 * area + i] * 255f).toInt().coerceIn(0, 255)
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, n, n, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        if (ncnnHandle != 0L) { NcnnBackend.releaseNet(ncnnHandle); ncnnHandle = 0L }
        if (miganHandle != 0L) { NcnnBackend.releaseNet(miganHandle); miganHandle = 0L }
        lamaOrt?.close(); lamaOrt = null
    }

    companion object {
        private const val TAG = "Inpainter"
        private const val MIGAN_TILE = 512  // MI-GAN 方磚邊長（轉檔固定輸入；逐區裁切縮放進出）
    }
}
