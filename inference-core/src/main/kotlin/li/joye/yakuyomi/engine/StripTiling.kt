package li.joye.yakuyomi.engine

import kotlin.math.max
import kotlin.math.min

/**
 * [MTYOMI] Long-strip (webtoon) detection tiling — pure geometry, no Android types,
 * so it is unit-testable on the JVM.
 *
 * Why: DBNet resizes the page so its long side equals the detector input size
 * (1024). On a full strip (e.g. 800x12000) that shrinks 30px text to ~2.6px and
 * detection finds nothing. Tiling keeps every tile's long side near the input
 * size, so text stays detectable.
 *
 * The tiles exist ONLY for detection. Boxes are mapped back to full-strip
 * coordinates and merged; OCR, erase and render all run on the whole strip, so
 * no seam ever appears in the output image.
 *
 * Safe cuts: a cut through a text line would split it into two fragments that
 * no merge step can rejoin, so cuts go through quiet bands (panel gutters,
 * blank transitions) found from a per-row gradient-energy scan. Where no quiet
 * band exists within reach, the cut is forced at the tile-height cap with an
 * overlap, and the merge step dedupes the double detections in the overlap.
 */
object StripTiling {

    /** One tile: half-open y-range [y0, y1) in full-strip coordinates. */
    data class Tile(val y0: Int, val y1: Int)

    /** One tile's detected lines, in tile-local coordinates. */
    data class TileLines(val y0: Int, val lines: List<TextLine>)

    /** One quiet band: half-open y-range [y0, y1) in full-strip coordinates. */
    data class Band(val y0: Int, val y1: Int)

    /** Overlap (full-res px) kept after every cut, safe or forced. */
    const val OVERLAP = 160

    /** Minimum tile height: avoids sliver tiles at the strip end. */
    const val MIN_TILE_H = 900

    /** Minimum quiet-band height (full-res px) to trust as text-free. */
    const val MIN_BAND_H = 40

    /**
     * Strip trigger. A normal manga page is ~1.4:1 and under ~2600px tall;
     * a strip is several times taller than wide AND tall enough that the
     * detector's long-side resize would shrink text below readability
     * (inputSize / height < ~1/3).
     */
    fun isStrip(pageW: Int, pageH: Int, inputSize: Int): Boolean =
        pageH >= 3 * inputSize && pageH >= 2 * pageW

    /**
     * Tile height cap: ~1.75x the detector input size, so after the tile's own
     * long-side resize, 30px source text is still ~17px — comfortably
     * detectable, and the tile count stays modest (a 12000px strip -> 7 tiles).
     */
    fun maxTileHeight(inputSize: Int): Int = (inputSize * 1.75f).toInt()

    /**
     * Find quiet bands from per-row gradient energy (one value per downscaled
     * row; [rowScale] = full-res px per downscaled row). A row is quiet when
     * its energy is at most 30% of the strip's 90th-percentile row energy:
     * text rows are edge-dense, gutters and flat art are not. A band only
     * counts if it runs at least [MIN_BAND_H] full-res px — shorter quiet runs
     * are just line spacing inside a bubble, never a safe cut.
     */
    fun findSafeBands(rowEnergy: FloatArray, rowScale: Float): List<Band> {
        if (rowEnergy.isEmpty()) return emptyList()
        val sorted = rowEnergy.sorted()
        val p90 = sorted[(sorted.size * 0.9f).toInt().coerceIn(0, sorted.size - 1)]
        val threshold = p90 * 0.30f
        val minRows = max(1, (MIN_BAND_H / rowScale).toInt())
        val bands = ArrayList<Band>()
        var runStart = -1
        for (y in rowEnergy.indices) {
            if (rowEnergy[y] <= threshold) {
                if (runStart < 0) runStart = y
            } else if (runStart >= 0) {
                if (y - runStart >= minRows) {
                    bands.add(Band((runStart * rowScale).toInt(), (y * rowScale).toInt()))
                }
                runStart = -1
            }
        }
        if (runStart >= 0 && rowEnergy.size - runStart >= minRows) {
            bands.add(Band((runStart * rowScale).toInt(), (rowEnergy.size * rowScale).toInt()))
        }
        return bands
    }

    /**
     * Cut the strip into tiles. Walk from the top: the ideal cut is one
     * [maxTileH] down; take the quiet band whose center is nearest the ideal
     * cut, provided the tile stays >= [MIN_TILE_H]. With no band in reach, cut
     * at [maxTileH] anyway (forced) — the overlap + merge step covers it.
     * Every cut keeps [OVERLAP] px of overlap with the next tile, so a line
     * straddling a cut still appears whole in at least one tile (text lines
     * are far shorter than the overlap). A trailing sliver shorter than
     * [MIN_TILE_H] is folded back into the previous tile.
     */
    fun computeTiles(
        pageH: Int,
        maxTileH: Int,
        bands: List<Band>,
        overlap: Int = OVERLAP,
        minTileH: Int = MIN_TILE_H,
    ): List<Tile> {
        if (pageH <= maxTileH) return listOf(Tile(0, pageH))
        val cuts = ArrayList<Int>()
        var y = 0
        while (pageH - y > maxTileH) {
            val ideal = y + maxTileH
            // Cut at a band's center, so the band only qualifies when its
            // center keeps the tile within [minTileH, maxTileH]. A band
            // straddling the ideal cut still qualifies via its center.
            val band = bands
                .filter { (it.y0 + it.y1) / 2 in (y + minTileH)..ideal }
                .minByOrNull { bandCenterDistance(it, ideal) }
            val cut = if (band != null) (band.y0 + band.y1) / 2 else ideal
            cuts.add(cut)
            y = cut - overlap
            if (y <= cuts.dropLast(1).lastOrNull() ?: -1) break // safety: never loop
        }
        if (cuts.isNotEmpty() && pageH - cuts.last() < minTileH) cuts.removeAt(cuts.lastIndex)
        val tiles = ArrayList<Tile>()
        var start = 0
        for (cut in cuts) {
            tiles.add(Tile(start, cut))
            start = max(0, cut - overlap)
        }
        tiles.add(Tile(start, pageH))
        return tiles
    }

    private fun bandCenterDistance(band: Band, ideal: Int): Int {
        val center = (band.y0 + band.y1) / 2
        return kotlin.math.abs(center - ideal)
    }

    /**
     * Map every tile's lines back to strip coordinates and dedupe across
     * tiles. Two lines from DIFFERENT tiles are the same physical line when
     * their axis-aligned boxes overlap heavily (IoU > 0.45) or one contains
     * the other (intersection / smaller area > 0.7 — a whole line in one tile
     * plus its fragment in the neighbour). Keep the higher score. Lines from
     * the same tile are never merged: the detector's own output stands.
     */
    private data class Aabb(val x0: Float, val y0: Float, val x1: Float, val y1: Float) {
        val area: Float get() = max(0f, x1 - x0) * max(0f, y1 - y0)
    }

    private data class Shifted(val line: TextLine, val tile: Int, val box: Aabb)

    private fun boxOf(line: TextLine, yOff: Int): Aabb {
        var x0 = Float.MAX_VALUE
        var y0 = Float.MAX_VALUE
        var x1 = -Float.MAX_VALUE
        var y1 = -Float.MAX_VALUE
        for (p in line.quad) {
            x0 = min(x0, p.x); y0 = min(y0, p.y + yOff)
            x1 = max(x1, p.x); y1 = max(y1, p.y + yOff)
        }
        return Aabb(x0, y0, x1, y1)
    }

    fun mergeTiles(tiles: List<TileLines>): List<TextLine> {
        val entries = ArrayList<Shifted>()
        for ((ti, t) in tiles.withIndex()) {
            for (line in t.lines) {
                val shifted = TextLine(line.quad.map { Pt(it.x, it.y + t.y0) }, line.score)
                entries.add(Shifted(shifted, ti, boxOf(line, t.y0)))
            }
        }
        val dropped = BooleanArray(entries.size)
        for (i in entries.indices) {
            if (dropped[i]) continue
            for (j in i + 1 until entries.size) {
                if (dropped[j] || entries[j].tile == entries[i].tile) continue
                val a = entries[i].box
                val b = entries[j].box
                val ix0 = max(a.x0, b.x0)
                val iy0 = max(a.y0, b.y0)
                val ix1 = min(a.x1, b.x1)
                val iy1 = min(a.y1, b.y1)
                val inter = max(0f, ix1 - ix0) * max(0f, iy1 - iy0)
                if (inter <= 0f) continue
                val union = a.area + b.area - inter
                val iou = if (union > 0f) inter / union else 0f
                val containment = inter / min(a.area, b.area)
                val areaRatio = if (max(a.area, b.area) > 0f) min(a.area, b.area) / max(a.area, b.area) else 0f
                if (iou > 0.45f && areaRatio > 0.7f) {
                    // Same line twice (overlap zone, near-identical boxes):
                    // keep the higher score.
                    if (entries[i].line.score >= entries[j].line.score) dropped[j] = true
                    else { dropped[i] = true; break }
                } else if (containment > 0.7f) {
                    // Whole line vs its fragment (split at a cut — a half-size
                    // fragment still scores IoU 0.5, so IoU alone cannot tell
                    // them apart): keep the LARGER box, never the fragment —
                    // a partial component can carry a higher mean score than
                    // the whole line.
                    if (a.area >= b.area) dropped[j] = true
                    else { dropped[i] = true; break }
                }
            }
        }
        return entries.filterIndexed { idx, _ -> !dropped[idx] }.map { it.line }
    }
}
