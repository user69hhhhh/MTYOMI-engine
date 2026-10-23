package li.joye.yakuyomi.engine

/**
 * 引擎要的模型本機檔案路徑——**全 NCNN**、五個角色：翻譯必備三顆（偵測 DBNet、OCR 48px CTC、去字 AOT-GAN）
 * ＋夜讀選配兩顆（人物分割 YOLO11-seg、CartoonSegmentation RTMDet-Ins），各一組 `.param`/`.bin`。
 * 用 [resolve] 從一堆 (檔名, 本機路徑) 比對出來——把「哪個檔是哪顆模型」的命名知識收進引擎。
 *
 * 路徑必須是**本機檔案路徑**（非 SAF/content uri）：模型走 native 記憶體載入
 * （勿 `readBytes()` 進 JVM heap，512MB 上限會 OOM；§10）。SAF 來源請先複製到 filesDir 再給路徑。
 * ORT 已整個從引擎拔除（2026-09-26；偵測/去字備援與 LaMa 更早退役）——產品 arm64、NCNN 必在，沒有第二條推論路徑。
 */
data class ModelSet(
    /**
     * 48px CTC OCR 的 NCNN `.param`（PE 當第二輸入繞開寬度牆；parity/export_ocr_ncnn.py 轉出）。
     * 兩份 param 共用同一份 `<name>.ncnn.bin`：原版 `<name>.ncnn.param` 與混合精度 `<name>_mixed.ncnn.param`
     * （backbone fp16、transformer+char_pred fp32）。這裡給哪一份都行——[Ocr] 的 pickParam 依 [OcrConfig.ncnnMixed]
     * ＋fp16 storage＋CPU asimdhp 決定實際載哪份；[resolve] 優先回原版（沒原版時才回 mixed）。
     */
    val ocr: String,
    /** DBNet（m-i-t default 偵測器）的 NCNN 版（`.param`，同名 `.bin` 需在旁）。偵測純 NCNN（手機 CPU 的 NEON/Winograd 核心）。 */
    val detectorNcnn: String? = null,
    /** AOT-GAN 去字的 NCNN 版（`.param`，同名 `.bin` 需在旁）。去字純 NCNN（整頁固定 tile 768）。 */
    val aotInpainterNcnn: String? = null,
    /**
     * 夜讀人物分割 YOLO11-seg 的 NCNN 版（`manga_seg_s.ncnn.param`，同名 `.bin` 需在旁）。**選配**：缺了翻譯照常，
     * 只是夜讀少一顆（兩顆都缺＝夜讀不可用）；給 :nightread-android 的 `NightReadRenderer.charSegmenter`。
     */
    val charSegYoloNcnn: String? = null,
    /** 夜讀人物分割 CartoonSegmentation（RTMDet-Ins）的 NCNN 版（`cartoonseg.ncnn.param`，同名 `.bin` 需在旁）。**選配**，同上。 */
    val charSegCsegNcnn: String? = null,
    /**
     * 阿拉伯文 OCR（PaddleOCR PP-OCRv3 arabic → NCNN，`paddle_ar_rec.ncnn.param`，同名 `.bin` 需在旁）。**選配**：
     * fork 內建於 assets、由 app 端複製進 models 目錄後在此被認出；缺了只是 [PaddleOcr] 不可用（主 OCR 照常）。
     * 檔名關鍵字 `paddle` 與其他角色不撞（不含 `ocr`/`dbnet`/`aot`/`manga_seg`/`cartoonseg`）。
     */
    val ocrPaddleArabic: String? = null,
    /**
     * 日／中 OCR（PaddleOCR PP-OCRv6 small → NCNN，`ppocrv6_small_rec.ncnn.param`，同名 `.bin` 需在旁）。**選配**：
     * fork 的下載型模型（app「下載模型」選單抓到 extra_models），缺了 [PaddleOcrV6] 不可用、退回主 OCR。
     * 此欄不由 [resolve] 填（理由同 [lamaInpaintOnnx]，且檔名含 `ocr` 會撞 [resolve] 的 OCR 關鍵字比對），
     * 由 app 端解析 extra_models 後直接給。字典（`ppocrv6_dict.txt`，18,708 字）由 app 讀成行經
     * [Yakuyomi.create] 的 paddleV6Alphabet 傳入，small／medium 共用。
     */
    val ocrPaddleV6Small: String? = null,
    /** 同 [ocrPaddleV6Small]，medium 版（`ppocrv6_medium_rec.ncnn.param`，fp32、76.5MB）。 */
    val ocrPaddleV6Medium: String? = null,
    /**
     * MI-GAN 去字（`migan_inpaint.ncnn.param`，同名 `.bin` 需在旁）。**選配**：balanced 品質去字門別，
     * 缺了 [Inpainter] 自動退回 AOT-GAN [OG]。檔名關鍵字 `migan` 與其他角色不撞
     * （不含 `ocr`/`dbnet`/`aot`/`manga_seg`/`cartoonseg`/`paddle`/`lama`）。
     */
    val miganInpainterNcnn: String? = null,
    /**
     * LaMa 去字 ONNX（`lama_fp32.onnx`；走 **ONNX Runtime**、非 NCNN——LaMa 的 FFT 核心 NCNN 跑不了，
     * 2026-10 MTYOMI 實測）。**選配**：max quality 去字門別，缺了退回 AOT-GAN [OG]。
     * 此欄不由 [resolve] 填（它只掃 `.param`），由 app 端解析 `.onnx` 後直接給。
     */
    val lamaInpaintOnnx: String? = null,
) {
    companion object {
        /**
         * 從 (檔名, 本機路徑) 清單比對出模型；缺 OCR / 偵測 / 去字任一 → 回 null（未備齊，呼叫端略過翻譯）。
         * 比對不分大小寫、只認 `.param`（`.onnx` 不再接受）：ocr＝含 `ocr`（多份時優先檔名**不含** `mixed` 的原版；
         * 只有 mixed 也接受，由 [Ocr] pickParam 決定能不能用）；偵測＝含 `dbnet`；去字＝含 `aot`。
         * 夜讀兩顆＝含 `manga_seg`（yolo）／`cartoonseg`（cseg），**缺不回 null**——翻譯就緒不受夜讀模型影響。
         * 關鍵字彼此不撞：`cartoonseg`／`manga_seg` 都不含 `ocr`、`aot`、`dbnet`，翻譯三顆的檔名也不含夜讀關鍵字（有測試守著）。
         */
        fun resolve(files: List<Pair<String, String>>): ModelSet? {
            fun matches(name: String, vararg keys: String): Boolean {
                val n = name.lowercase()
                return n.endsWith(".param") && keys.any { n.contains(it) }
            }
            fun find(vararg keys: String): String? = files.firstOrNull { (name, _) -> matches(name, *keys) }?.second
            val ocrParams = files.filter { (name, _) -> matches(name, "ocr") }
            val ocr = (ocrParams.firstOrNull { !it.first.lowercase().contains("mixed") } ?: ocrParams.firstOrNull())?.second
                ?: return null
            val detNcnn = find("dbnet") ?: return null
            val aotNcnn = find("aot") ?: return null
            return ModelSet(
                ocr = ocr,
                detectorNcnn = detNcnn,
                aotInpainterNcnn = aotNcnn,
                charSegYoloNcnn = find("manga_seg"),
                charSegCsegNcnn = find("cartoonseg"),
                ocrPaddleArabic = find("paddle"),
                miganInpainterNcnn = find("migan"),
            )
        }
    }
}
