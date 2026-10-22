package li.joye.yakuyomi.engine

import android.graphics.Bitmap

/**
 * OCR 後端共同介面：[Ocr]（48px CTC，日文主路徑）與 [PaddleOcr]（PaddleOCR 阿拉伯文，fork 新增）互換；
 * [Pipeline] 只依賴此介面，由 [Yakuyomi.create] 依 [OcrConfig.engine]＋[ModelSet] 決定建哪一顆。
 */
interface OcrEngine : AutoCloseable {
    /** 實際後端標籤（進診斷紀錄用，讓無 adb 時也能確認產品載的是哪顆 OCR）。 */
    val backend: String

    /** 對每條文字行做 OCR，就地填入 direction 與 text。 */
    suspend fun recognize(page: Bitmap, lines: List<TextLine>)

    /** 暖機：跑一次空白推論，吃掉首次 forward 的一次性配置（併發翻多頁前先呼叫一次）。 */
    fun warmUp()
}
