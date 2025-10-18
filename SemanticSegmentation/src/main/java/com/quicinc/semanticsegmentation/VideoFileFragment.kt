package com.quicinc.semanticsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
// Using FFmpeg for frame-by-frame decoding
import org.bytedeco.javacv.FFmpegFrameGrabber
import org.bytedeco.javacv.AndroidFrameConverter
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import java.io.FileInputStream
import kotlin.math.max

class VideoFileFragment : Fragment() {
    private var segmentor: TfLiteSegmentor? = null
    private var videoUri: Uri? = null
    private var fragmentRender: FragmentRender? = null

    // Keep array of hidden states across frames for the running video (may contain 4 hidden tensors)
    private var hiddenState: Array<FloatArray>? = null

    // Carry producer timing with the frame
    private data class FrameItem(
        val pre: TfLiteSegmentor.Preprocessed,
        val producerElapsedNanos: Long
    )

    @Volatile private var pipelineRunning = false
    // FFmpeg resources
    private var ffmpegGrabber: FFmpegFrameGrabber? = null
    private var ffmpegPfd: ParcelFileDescriptor? = null
    private var pipelineScope: CoroutineScope? = null
    private var producerJob: Job? = null
    private var consumerJob: Job? = null
    // change channel type to FrameItem
    private var frameChan: Channel<FrameItem>? = null

    companion object{
        fun create(seg: TfLiteSegmentor, uri: Uri) = VideoFileFragment().apply {
            segmentor = seg
            videoUri = uri
        }
    }

    // Allow swapping the segmentor at runtime (restart consumer only; keep producer alive)
    fun setSegmentor(seg: TfLiteSegmentor) {
        val wasRunning = pipelineRunning
        segmentor = seg
        hiddenState = null
        if (wasRunning) {
            // Restart only the consumer job; keep producer and channel alive
            consumerJob?.cancel()
            consumerJob = pipelineScope?.launch(Dispatchers.Default) {
                val ch = frameChan ?: return@launch
                while (isActive) {
                    val item = try { ch.receive() } catch (_: Exception) { break }
                    val segLoc = segmentor ?: continue
                    val pre = item.pre
                    val producerElapsed = item.producerElapsedNanos
                    try {
                        val tStart = System.nanoTime()
                        val inf = segLoc.infer(pre, hiddenState)
                        hiddenState = inf.newHidden
                        val fr = fragmentRender ?: continue
                        val full = fr.isFullMode()
                        val outForDisplay: Bitmap =
                            if (pre.cropRectInOriginal != null && pre.originalForDisplay != null) {
                                val rect: Rect = pre.cropRectInOriginal
                                val cropSeg = segLoc.postprocessToBitmap(inf, rect.width(), rect.height(), pre.sensorOrientation)
                                if (full) {
                                    cropSeg
                                } else {
                                    val composed = Bitmap.createBitmap(pre.originalForDisplay.width, pre.originalForDisplay.height, Bitmap.Config.ARGB_8888)
                                    val canvas = Canvas(composed)
                                    canvas.drawBitmap(pre.originalForDisplay, 0f, 0f, null)
                                    canvas.drawBitmap(cropSeg, rect.left.toFloat(), rect.top.toFloat(), null)
                                    composed
                                }
                            } else {
                                segLoc.postprocessToBitmap(inf, pre.viewW, pre.viewH, pre.sensorOrientation)
                            }
                        val consumerElapsed = System.nanoTime() - tStart
                        // Pass original frame for the renderer to optionally show
                        val orig = pre.originalForDisplay
                        fr.post { fr.render(outForDisplay, orig, 0f, consumerElapsed, producerElapsed) }
                    } catch (_: Exception) { }
                }
            }
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
        val uri = videoUri ?: return
        val ctx: Context = requireContext()
        if (pipelineRunning) return
        pipelineRunning = true
        hiddenState = null

        pipelineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        frameChan = Channel(capacity = 1, onBufferOverflow = BufferOverflow.DROP_OLDEST)

        // Producer: decode + preprocess at ~video FPS using FFmpeg (sequential grab), measure producer time
        producerJob = pipelineScope?.launch(Dispatchers.IO) {
            // Open the content Uri as an InputStream for FFmpeg
            val pfd = try { ctx.contentResolver.openFileDescriptor(uri, "r") } catch (_: Exception) { null }
            if (pfd == null) return@launch
            ffmpegPfd = pfd
            val fis = FileInputStream(pfd.fileDescriptor)
            val grabber = FFmpegFrameGrabber(fis)
            // Do NOT force scaling here — use the video's natural width/height
            val converter = AndroidFrameConverter()
            try {
                grabber.start()
                ffmpegGrabber = grabber
                // Rotation from metadata if present (e.g., "90")
                val rotStr = try { grabber.getVideoMetadata("rotate") } catch (_: Exception) { null }
                val sensorOrientation = rotStr?.toIntOrNull() ?: 0
                // Use natural FPS reported by the grabber, fallback to 10 if unknown
                val srcFps = if (grabber.frameRate > 0.0) grabber.frameRate else 10.0
                val frameIntervalMillis = (1000.0 / srcFps).toLong()

                while (isActive) {
                    val tStart = System.nanoTime()
                    val frame = try { grabber.grabImage() } catch (_: Exception) { null } ?: break
                    val bmp: Bitmap? = try { converter.convert(frame) } catch (_: Exception) { null }
                    if (bmp != null) {
                        val s = segmentor ?: continue
                        val fr = fragmentRender
                        // Update reference size for proper bbox mapping in full mode
                        fr?.updateReferenceSize(bmp.width, bmp.height)
                        val pre = try {
                            fr?.getCroppedBitmapWithRect(bmp)?.let { (cropped, rect) ->
                                s.preprocess(cropped, sensorOrientation, originalForDisplay = bmp, cropRectInOriginal = rect)
                            } ?: s.preprocess(bmp, sensorOrientation, originalForDisplay = bmp, cropRectInOriginal = null)
                        } catch (_: Exception) { continue }
                        val producerElapsed = System.nanoTime() - tStart
                        frameChan?.trySend(FrameItem(pre, producerElapsed))

                        // Sleep roughly 1/FPS minus time already spent producing this frame
                        val elapsedMillis = (System.nanoTime() - tStart) / 1_000_000
                        val sleepMillis = frameIntervalMillis - elapsedMillis
                        if (sleepMillis > 0) {
                            try { delay(sleepMillis) } catch (_: Exception) { /* ignore cancellation */ }
                        }
                    }
                }
            } catch (_: Exception) {
                // ignore
            } finally {
                try { grabber.stop() } catch (_: Exception) { }
                try { grabber.release() } catch (_: Exception) { }
                try { fis.close() } catch (_: Exception) { }
                try { pfd.close() } catch (_: Exception) { }
                ffmpegGrabber = null
                ffmpegPfd = null
            }
        }

        // Consumer: infer + postprocess + render, measure consumer time
        consumerJob = pipelineScope?.launch(Dispatchers.Default) {
            val ch = frameChan ?: return@launch
            while (isActive) {
                val item = try { ch.receive() } catch (_: Exception) { break }
                val s = segmentor ?: continue
                val pre = item.pre
                val producerElapsed = item.producerElapsedNanos
                try {
                    val tStart = System.nanoTime()
                    val inf = s.infer(pre, hiddenState)
                    hiddenState = inf.newHidden
                    val fr = fragmentRender ?: continue
                    val full = fr.isFullMode()
                    val outForDisplay: Bitmap =
                        if (pre.cropRectInOriginal != null && pre.originalForDisplay != null) {
                            val rect: Rect = pre.cropRectInOriginal
                            val cropSeg = s.postprocessToBitmap(inf, rect.width(), rect.height(), pre.sensorOrientation)
                            if (full) {
                                cropSeg
                            } else {
                                val composed = Bitmap.createBitmap(pre.originalForDisplay.width, pre.originalForDisplay.height, Bitmap.Config.ARGB_8888)
                                val canvas = Canvas(composed)
                                canvas.drawBitmap(pre.originalForDisplay, 0f, 0f, null)
                                canvas.drawBitmap(cropSeg, rect.left.toFloat(), rect.top.toFloat(), null)
                                composed
                            }
                        } else {
                            s.postprocessToBitmap(inf, pre.viewW, pre.viewH, pre.sensorOrientation)
                        }
                    val consumerElapsed = System.nanoTime() - tStart
                    // Pass original frame for the renderer to optionally show
                    val orig = pre.originalForDisplay
                    fr.post { fr.render(outForDisplay, orig, 0f, consumerElapsed, producerElapsed) }
                } catch (_: Exception) { }
            }
        }
    }

    private fun stopPipeline() {
        pipelineRunning = false
        // Cancel coroutines and close channel
        consumerJob?.cancel(); consumerJob = null
        producerJob?.cancel(); producerJob = null
        pipelineScope?.cancel(); pipelineScope = null
        try { frameChan?.close() } catch (_: Exception) { }
        frameChan = null
        // Clear state and release retriever
        hiddenState = null
        try { ffmpegGrabber?.stop() } catch (_: Exception) {}
        try { ffmpegGrabber?.release() } catch (_: Exception) {}
        ffmpegGrabber = null
        try { ffmpegPfd?.close() } catch (_: Exception) {}
        ffmpegPfd = null
    }
}
