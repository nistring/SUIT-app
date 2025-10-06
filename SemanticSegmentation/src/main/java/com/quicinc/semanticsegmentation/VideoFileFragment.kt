package com.quicinc.semanticsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment

class VideoFileFragment : Fragment() {
    private var segmentor: TfLiteSegmentor? = null
    private var videoUri: Uri? = null
    private var fragmentRender: FragmentRender? = null
    private var bgThread: HandlerThread? = null
    private var bgHandler: Handler? = null

    // Keep array of hidden states across frames for the running video (may contain 4 hidden tensors)
    private var hiddenState: Array<FloatArray>? = null

    companion object {
        fun create(seg: TfLiteSegmentor, uri: Uri) = VideoFileFragment().apply {
            segmentor = seg
            videoUri = uri
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
        startProcessing()
    }

    override fun onPause() {
        super.onPause()
        stopBg()
    }

    private fun startBg() { bgThread = HandlerThread("VideoFile").also { it.start() }; bgHandler = Handler(bgThread!!.looper) }
    private fun stopBg() { bgThread?.quitSafely(); bgThread = null; bgHandler = null }

    private fun startProcessing() {
        val seg = segmentor ?: return
        val uri = videoUri ?: return
        val ctx: Context = requireContext()
        hiddenState = null

        bgHandler?.post {
            val mmr = MediaMetadataRetriever()
            try {
                mmr.setDataSource(ctx, uri)
                val durationMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                val stepMs = 100L // ~10 fps
                val fr = fragmentRender
                val render: (Bitmap) -> Unit = { bmp ->
                    fr?.post { fr.render(bmp, 0f, seg.lastInferTime, seg.lastPreTime, seg.lastPostTime) }
                }

                var t = 0L
                while (t < durationMs) {
                    try {
                        mmr.getFrameAtTime(t * 1000, MediaMetadataRetriever.OPTION_CLOSEST)?.let { bmp ->
                            val cropWithRect = fr?.getCroppedBitmapWithRect(bmp)
                            if (cropWithRect != null) {
                                val (cropped, rect) = cropWithRect
                                val (outCrop, newHidden) = seg.predict(cropped, 0, hiddenState)
                                hiddenState = newHidden
                                try {
                                    val display = Bitmap.createBitmap(bmp.width, bmp.height, Bitmap.Config.ARGB_8888)
                                    val canvas = android.graphics.Canvas(display)
                                    canvas.drawBitmap(bmp, 0f, 0f, null)
                                    val out = if (outCrop.width == rect.width() && outCrop.height == rect.height())
                                        outCrop
                                    else
                                        Bitmap.createScaledBitmap(outCrop, rect.width(), rect.height(), true)
                                    canvas.drawBitmap(out, rect.left.toFloat(), rect.top.toFloat(), null)
                                    render(display)
                                } catch (_: Exception) {
                                    val (outFull, newHiddenFull) = seg.predict(bmp, 0, hiddenState)
                                    hiddenState = newHiddenFull
                                    render(outFull)
                                }
                            } else {
                                val (out, newHidden) = seg.predict(bmp, 0, hiddenState)
                                hiddenState = newHidden
                                render(out)
                            }
                        }
                    } catch (_: Exception) {
                        // ignore frame-level errors
                    }
                    t += stepMs
                }
            } catch (_: Exception) {
                // ignore setup errors
            } finally {
                try { mmr.release() } catch (_: Exception) {}
            }
        }
    }
}
