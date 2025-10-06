package com.quicinc.semanticsegmentation

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
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
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment

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

    companion object {
        fun create(seg: TfLiteSegmentor) = ScreenCaptureFragment().apply { segmentor = seg }
    }

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

        imageReader = ImageReader.newInstance(width, height, android.graphics.PixelFormat.RGBA_8888, 2)
        imageReader?.setOnImageAvailableListener({ reader ->
            val img = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            try {
                val bmp = imageToBitmap(img)
                // Pass and update hidden state array
                val (outBmp, newHidden) = seg.predict(bmp, 0, hiddenState)
                hiddenState = newHidden
                fragmentRender?.post { fragmentRender?.render(outBmp, 0f, seg.lastInferTime, seg.lastPreTime, seg.lastPostTime) }
            } finally {
                img.close()
            }
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
}
