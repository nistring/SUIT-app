package com.quicinc.semanticsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
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

class VideoFileFragment : Fragment() {
    private var segmentor: TfLiteSegmentor? = null
    private var videoUri: Uri? = null
    private var fragmentRender: FragmentRender? = null
    private var hiddenState: Array<FloatArray>? = null

    private data class FrameItem(
        val pre: TfLiteSegmentor.Preprocessed,
        val producerElapsedNanos: Long
    )

    @Volatile private var pipelineRunning = false
    @Volatile private var latestOriginalBitmap: Bitmap? = null  // NEW: Store latest frame
    private var ffmpegGrabber: FFmpegFrameGrabber? = null
    private var ffmpegPfd: ParcelFileDescriptor? = null
    private var pipelineScope: CoroutineScope? = null
    private var producerJob: Job? = null
    private var consumerJob: Job? = null
    private var frameChan: Channel<FrameItem>? = null

    companion object {
        fun create(seg: TfLiteSegmentor, uri: Uri) = VideoFileFragment().apply {
            segmentor = seg
            videoUri = uri
        }
    }

    fun setSegmentor(seg: TfLiteSegmentor) {
        val wasRunning = pipelineRunning
        segmentor = seg
        hiddenState = null
        if (wasRunning) {
            consumerJob?.cancel()
            consumerJob = pipelineScope?.launch(Dispatchers.Default) { runConsumer() }
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
        fragmentRender?.resetBBoxInitialization()

        pipelineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        frameChan = Channel(capacity = 1, onBufferOverflow = BufferOverflow.DROP_OLDEST)

        producerJob = pipelineScope?.launch(Dispatchers.IO) {
            val pfd = try { ctx.contentResolver.openFileDescriptor(uri, "r") } catch (_: Exception) { null } ?: return@launch
            ffmpegPfd = pfd
            val grabber = FFmpegFrameGrabber(FileInputStream(pfd.fileDescriptor))
            val converter = AndroidFrameConverter()
            try {
                grabber.start()
                ffmpegGrabber = grabber
                val sensorOrientation = try { grabber.getVideoMetadata("rotate")?.toIntOrNull() ?: 0 } catch (_: Exception) { 0 }
                val frameIntervalMillis = (1000.0 / if (grabber.frameRate > 0.0) grabber.frameRate else 10.0).toLong()

                // REMOVED: var isFirstFrame = true
                while (isActive) {
                    val tStart = System.nanoTime()
                    val frame = try { grabber.grabImage() } catch (_: Exception) { null } ?: break
                    val bmp: Bitmap? = try { converter.convert(frame) } catch (_: Exception) { null }
                    if (bmp != null) {
                        val s = segmentor ?: continue
                        fragmentRender?.updateReferenceSize(bmp.width, bmp.height)
                        
                        // REMOVED: Auto-initialize bounding box on first frame
                        latestOriginalBitmap = bmp  // NEW: Store for manual init
                        
                        val pre = try {
                            fragmentRender?.getCroppedBitmapWithRect(bmp)?.let { (cropped, rect) ->
                                s.preprocess(cropped, sensorOrientation, originalForDisplay = bmp, cropRectInOriginal = rect)
                            } ?: s.preprocess(bmp, sensorOrientation, originalForDisplay = bmp, cropRectInOriginal = null)
                        } catch (_: Exception) { continue }
                        val producerElapsed = System.nanoTime() - tStart
                        frameChan?.trySend(FrameItem(pre, producerElapsed))

                        val sleepMillis = frameIntervalMillis - (System.nanoTime() - tStart) / 1_000_000
                        if (sleepMillis > 0) try { delay(sleepMillis) } catch (_: Exception) { }
                    }
                }
            } finally {
                safeClose { grabber.stop() }
                safeClose { grabber.release() }
                safeClose { FileInputStream(pfd.fileDescriptor).close() }
                safeClose { pfd.close() }
                ffmpegGrabber = null
                ffmpegPfd = null
            }
        }

        consumerJob = pipelineScope?.launch(Dispatchers.Default) { runConsumer() }
    }

    private suspend fun runConsumer() {
        val ch = frameChan ?: return
        val scope = pipelineScope ?: return
        while (scope.isActive) {
            val item = try { ch.receive() } catch (_: Exception) { break }
            val s = segmentor ?: continue
            val pre = item.pre
            val producerElapsed = item.producerElapsedNanos
            try {
                val tStart = System.nanoTime()
                val inf = s.infer(pre, hiddenState)
                hiddenState = inf.newHidden
                val fr = fragmentRender ?: continue
                val outForDisplay = composeOutputBitmap(s, inf, pre, fr.isFullMode())
                val consumerElapsed = System.nanoTime() - tStart
                fr.post { fr.render(outForDisplay, pre.originalForDisplay, 0f, consumerElapsed, producerElapsed) }
            } catch (_: Exception) { }
        }
    }

    private fun composeOutputBitmap(s: TfLiteSegmentor, inf: TfLiteSegmentor.InferResult, pre: TfLiteSegmentor.Preprocessed, fullMode: Boolean): Bitmap {
        return if (pre.cropRectInOriginal != null && pre.originalForDisplay != null) {
            val rect: Rect = pre.cropRectInOriginal
            val cropSeg = s.postprocessToBitmap(inf, rect.width(), rect.height(), pre.sensorOrientation)
            if (fullMode) cropSeg else {
                Bitmap.createBitmap(pre.originalForDisplay.width, pre.originalForDisplay.height, Bitmap.Config.ARGB_8888).apply {
                    Canvas(this).apply {
                        drawBitmap(pre.originalForDisplay, 0f, 0f, null)
                        drawBitmap(cropSeg, rect.left.toFloat(), rect.top.toFloat(), null)
                    }
                }
            }
        } else {
            s.postprocessToBitmap(inf, pre.viewW, pre.viewH, pre.sensorOrientation)
        }
    }

    private fun stopPipeline() {
        pipelineRunning = false
        consumerJob?.cancel()
        producerJob?.cancel()
        pipelineScope?.cancel()
        safeClose { frameChan?.close() }
        hiddenState = null
        safeClose { ffmpegGrabber?.stop() }
        safeClose { ffmpegGrabber?.release() }
        safeClose { ffmpegPfd?.close() }
        ffmpegGrabber = null
        ffmpegPfd = null
    }

    fun requestBBoxInit() {
        latestOriginalBitmap?.let { bmp ->
            fragmentRender?.post {
                val success = fragmentRender?.initializeBBoxFromOriginal(bmp)
                if (success == true) {
                    android.widget.Toast.makeText(requireContext(), "BBox initialized", android.widget.Toast.LENGTH_SHORT).show()
                } else {
                    android.widget.Toast.makeText(requireContext(), "Failed to detect contour", android.widget.Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private inline fun safeClose(block: () -> Unit) {
        try { block() } catch (_: Exception) { }
    }
}
