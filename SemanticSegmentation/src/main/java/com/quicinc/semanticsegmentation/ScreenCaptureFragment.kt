package com.quicinc.semanticsegmentation

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.DisplayMetrics
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import java.util.concurrent.atomic.AtomicReference

class ScreenCaptureFragment : Fragment() {
    private var segmentor: TfLiteSegmentor? = null
    private var projection: MediaProjection? = null
    private var imageReader: ImageReader? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var bgThread: HandlerThread? = null
    private var bgHandler: Handler? = null
    private var fragmentRender: FragmentRender? = null

    // Single hidden state array carried across frames (may contain multiple tensors)
    private var hiddenState: Array<FloatArray>? = null

    // --- Added: three-stage pipeline state ---
    private val latestPre = AtomicReference<TfLiteSegmentor.Preprocessed?>()
    private var inferThread: HandlerThread? = null
    private var inferHandler: Handler? = null
    @Volatile private var pipelineRunning = false
    // -----------------------------------------

    private val projectionLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val mgr = requireContext().getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            projection = mgr.getMediaProjection(result.resultCode, result.data!!)
            startCapture()
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
        startBg()
        requestProjection()
    }

    override fun onPause() {
        stopCapture()
        stopBg()
        super.onPause()
    }

    private fun startBg() { bgThread = HandlerThread("ScreenCap").also { it.start() }; bgHandler = Handler(bgThread!!.looper) }
    private fun stopBg() { bgThread?.quitSafely(); bgThread = null; bgHandler = null }

    private fun requestProjection() {
        val mgr = requireContext().getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        projectionLauncher.launch(mgr.createScreenCaptureIntent())
    }

    private fun startCapture() {
        val seg = segmentor ?: return
        // Reset hidden state for a new capture session
        hiddenState = null

        val metrics = DisplayMetrics()
        requireActivity().windowManager.defaultDisplay.getRealMetrics(metrics)
        val width = metrics.widthPixels
        val height = metrics.heightPixels
        val density = metrics.densityDpi

        // Start pipeline threads
        startPipeline()

        imageReader = ImageReader.newInstance(width, height, android.graphics.PixelFormat.RGBA_8888, 2)
        imageReader?.setOnImageAvailableListener({ reader ->
            val img = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            try {
                if (latestPre.get() == null) { // drop if pending
                    val bmp = imageToBitmap(img)
                    // Update reference size for mapping rect
                    fragmentRender?.updateReferenceSize(bmp.width, bmp.height)
                    // Try to crop via FragmentRender
                    val pre = fragmentRender?.getCroppedBitmapWithRect(bmp)?.let { (cropped, rect) ->
                        seg.preprocess(cropped, 0, originalForDisplay = bmp, cropRectInOriginal = rect)
                    } ?: seg.preprocess(bmp, 0, originalForDisplay = bmp, cropRectInOriginal = null)
                    latestPre.set(pre)
                }
            } finally { img.close() }
        }, bgHandler)

        virtualDisplay = projection?.createVirtualDisplay(
            "SegScreen",
            width,
            height,
            density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader!!.surface,
            null,
            bgHandler
        )
    }

    private fun stopCapture() {
        virtualDisplay?.release(); virtualDisplay = null
        imageReader?.close(); imageReader = null
        projection?.stop(); projection = null
        // clear hidden state on stop
        hiddenState = null
        // also clear any pending preprocessed frame
        latestPre.set(null)
        stopPipeline()
    }

    private fun startPipeline() {
        if (pipelineRunning) return
        pipelineRunning = true
        inferThread = HandlerThread("ScreenInfer").also { it.start() }
        inferHandler = Handler(inferThread!!.looper)

        inferHandler?.post(object : Runnable {
            override fun run() {
                if (!pipelineRunning) return
                val seg = segmentor ?: return
                val pre = latestPre.getAndSet(null)
                if (pre != null) {
                    try {
                        val inf = seg.infer(pre, hiddenState)
                        hiddenState = inf.newHidden
                        val fr = fragmentRender
                        if (fr != null) {
                            val full = fr.isFullMode()
                            val outBmp: Bitmap =
                                if (pre.cropRectInOriginal != null && pre.originalForDisplay != null) {
                                    val rect: Rect = pre.cropRectInOriginal
                                    val cropSeg = seg.postprocessToBitmap(inf, rect.width(), rect.height(), pre.sensorOrientation)
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
                                    seg.postprocessToBitmap(inf, pre.viewW, pre.viewH, pre.sensorOrientation)
                                }
                            // Pass original frame as well
                            val orig = pre.originalForDisplay
                            fragmentRender?.post { fragmentRender?.render(outBmp, orig, 0f, 0L, 0L) }
                        }
                    } catch (_: Exception) { }
                }
                inferHandler?.post(this)
            }
        })
    }

    private fun stopPipeline() {
        pipelineRunning = false
        // Drain queued tasks to prevent duplicate processing after model swap
        inferHandler?.removeCallbacksAndMessages(null)
        inferThread?.quitSafely(); inferThread = null; inferHandler = null
    }

    private fun imageToBitmap(image: Image): Bitmap {
        val plane = image.planes[0]
        val width = image.width
        val height = image.height
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * width
        val bitmap = Bitmap.createBitmap(width + rowPadding / pixelStride, height, Bitmap.Config.ARGB_8888)
        val buffer = plane.buffer
        buffer.rewind()
        bitmap.copyPixelsFromBuffer(buffer)
        return Bitmap.createBitmap(bitmap, 0, 0, width, height)
    }

    // Allow swapping the segmentor at runtime (restart pipeline if running)
    fun setSegmentor(seg: TfLiteSegmentor) {
        val wasRunning = pipelineRunning
        segmentor = seg
        // Drop any stale work and restart only the inferencer
        inferHandler?.removeCallbacksAndMessages(null)
        latestPre.set(null)
        hiddenState = null
        stopPipeline() // stops only infer thread
        if (wasRunning && isResumed) startPipeline()
    }
}
