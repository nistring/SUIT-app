package com.quicinc.semanticsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import java.util.concurrent.atomic.AtomicReference

class VideoFileFragment : Fragment() {
    private var segmentor: TfLiteSegmentor? = null
    private var videoUri: Uri? = null
    private var fragmentRender: FragmentRender? = null

    // Keep array of hidden states across frames for the running video (may contain 4 hidden tensors)
    private var hiddenState: Array<FloatArray>? = null

    // --- Added: three-stage pipeline state ---
    private var inputThread: HandlerThread? = null
    private var inputHandler: Handler? = null
    private var inferThread: HandlerThread? = null
    private var inferHandler: Handler? = null
    private val latestPre = AtomicReference<TfLiteSegmentor.Preprocessed?>()
    @Volatile private var pipelineRunning = false
    // -----------------------------------------

    companion object{
        fun create(seg: TfLiteSegmentor, uri: Uri) = VideoFileFragment().apply {
            segmentor = seg
            videoUri = uri
        }
    }

    // Allow swapping the segmentor at runtime (restart inferencer only; keep input stream)
    fun setSegmentor(seg: TfLiteSegmentor) {
        val wasRunning = pipelineRunning
        segmentor = seg
        // Drop any stale frames and hidden state
        latestPre.set(null)
        hiddenState = null
        if (wasRunning) {
            // Restart only the inferencer loop; keep producer/input thread alive
            inferHandler?.removeCallbacksAndMessages(null)
            inferThread?.quitSafely()
            inferThread = HandlerThread("VideoInfer").also { it.start() }
            inferHandler = Handler(inferThread!!.looper)
            inferHandler?.post(object : Runnable {
                override fun run() {
                    if (!pipelineRunning) return
                    val segLoc = segmentor ?: return
                    val pre = latestPre.getAndSet(null)
                    if (pre != null) {
                        try {
                            val inf = segLoc.infer(pre, hiddenState)
                            hiddenState = inf.newHidden
                            val fr = fragmentRender ?: return
                            val outForDisplay: Bitmap = if (pre.cropRectInOriginal != null && pre.originalForDisplay != null) {
                                val rect: Rect = pre.cropRectInOriginal
                                val cropSeg = segLoc.postprocessToBitmap(inf, rect.width(), rect.height(), pre.sensorOrientation)
                                val composed = Bitmap.createBitmap(pre.originalForDisplay.width, pre.originalForDisplay.height, Bitmap.Config.ARGB_8888)
                                val canvas = Canvas(composed)
                                canvas.drawBitmap(pre.originalForDisplay, 0f, 0f, null)
                                canvas.drawBitmap(cropSeg, rect.left.toFloat(), rect.top.toFloat(), null)
                                composed
                            } else {
                                segLoc.postprocessToBitmap(inf, pre.viewW, pre.viewH, pre.sensorOrientation)
                            }
                            fr.post { fr.render(outForDisplay, 0f, segLoc.lastInferTime, segLoc.lastPreTime, segLoc.lastPostTime) }
                        } catch (_: Exception) { }
                    }
                    inferHandler?.post(this)
                }
            })
        }
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        return inflater.inflate(R.layout.fragment_camera, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        fragmentRender = view.findViewById(R.id.fragmentRender)
    }

    override fun onResume() {
        super.onResume()
        startPipeline()
    }

    override fun onPause() {
        super.onPause()
        stopPipeline()
    }

    private fun startPipeline() {
        val seg = segmentor ?: return
        val uri = videoUri ?: return
        val ctx: Context = requireContext()
        hiddenState = null
        if (pipelineRunning) return
        pipelineRunning = true

        // Threads
        inputThread = HandlerThread("VideoInput").also { it.start() }
        inputHandler = Handler(inputThread!!.looper)
        inferThread = HandlerThread("VideoInfer").also { it.start() }
        inferHandler = Handler(inferThread!!.looper)

        // Producer
        inputHandler?.post {
            val mmr = MediaMetadataRetriever()
            try {
                mmr.setDataSource(ctx, uri)
                val durationMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                val stepMs = 100L
                var t = 0L
                while (pipelineRunning && t < durationMs) {
                    try {
                        if (latestPre.get() == null) { // drop if infer not consumed
                            mmr.getFrameAtTime(t * 1000, MediaMetadataRetriever.OPTION_CLOSEST)?.let { bmp ->
                                val fr = fragmentRender
                                val cropWithRect = fr?.getCroppedBitmapWithRect(bmp)
                                if (cropWithRect != null) {
                                    val (cropped, rect) = cropWithRect
                                    latestPre.set(seg.preprocess(cropped, 0, originalForDisplay = bmp, cropRectInOriginal = rect))
                                } else {
                                    latestPre.set(seg.preprocess(bmp, 0, originalForDisplay = bmp, cropRectInOriginal = null))
                                }
                            }
                        }
                    } catch (_: Exception) { }
                    t += stepMs
                    try { Thread.sleep(stepMs) } catch (_: InterruptedException) { }
                }
            } catch (_: Exception) { }
            finally { try { mmr.release() } catch (_: Exception) {} }
        }

        // Inferencer + postprocess on same thread
        inferHandler?.post(object : Runnable {
            override fun run() {
                if (!pipelineRunning) return
                val pre = latestPre.getAndSet(null)
                if (pre != null) {
                    try {
                        val inf = seg.infer(pre, hiddenState)
                        hiddenState = inf.newHidden
                        val fr = fragmentRender ?: return
                        val outForDisplay: Bitmap = if (pre.cropRectInOriginal != null && pre.originalForDisplay != null) {
                            val rect: Rect = pre.cropRectInOriginal
                            val cropSeg = seg.postprocessToBitmap(inf, rect.width(), rect.height(), pre.sensorOrientation)
                            val composed = Bitmap.createBitmap(pre.originalForDisplay.width, pre.originalForDisplay.height, Bitmap.Config.ARGB_8888)
                            val canvas = Canvas(composed)
                            canvas.drawBitmap(pre.originalForDisplay, 0f, 0f, null)
                            canvas.drawBitmap(cropSeg, rect.left.toFloat(), rect.top.toFloat(), null)
                            composed
                        } else {
                            seg.postprocessToBitmap(inf, pre.viewW, pre.viewH, pre.sensorOrientation)
                        }
                        fr.post { fr.render(outForDisplay, 0f, seg.lastInferTime, seg.lastPreTime, seg.lastPostTime) }
                    } catch (_: Exception) { }
                }
                inferHandler?.post(this)
            }
        })
    }

    private fun stopPipeline() {
        pipelineRunning = false
        // Drain queued tasks to prevent duplicate processing after model swap
        inputHandler?.removeCallbacksAndMessages(null)
        inferHandler?.removeCallbacksAndMessages(null)
        // Clear any pending preprocessed frame and hidden state
        latestPre.set(null)
        hiddenState = null
        // Now stop threads
        inputThread?.quitSafely(); inputThread = null; inputHandler = null
        inferThread?.quitSafely(); inferThread = null; inferHandler = null
    }
}
