package com.quicinc.semanticsegmentation

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.usb.UsbDevice
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.*
import android.view.TextureView
import androidx.fragment.app.Fragment
import com.serenegiant.usb.IFrameCallback
import com.serenegiant.usb.USBMonitor
import com.serenegiant.usb.UVCCamera
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicReference

class UsbCameraFragment : Fragment() {
    private lateinit var textureView: TextureView
    private lateinit var fragmentRender: FragmentRender
    private var segmentor: TfLiteSegmentor? = null
    private var usbMonitor: USBMonitor? = null
    private var uvcCamera: UVCCamera? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    private var hiddenState: Array<FloatArray>? = null
    private val latestPre = AtomicReference<TfLiteSegmentor.Preprocessed?>()
    private var inferThread: HandlerThread? = null
    private var inferHandler: Handler? = null
    @Volatile private var pipelineRunning = false

    companion object { 
        private const val TAG = "UsbCameraFragment"
        fun create(seg: TfLiteSegmentor) = UsbCameraFragment().apply { segmentor = seg } 
    }

    private val surfaceListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(texture: SurfaceTexture, width: Int, height: Int) {
            if (pipelineRunning) openCamera()
        }
        override fun onSurfaceTextureSizeChanged(texture: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(texture: SurfaceTexture): Boolean = true
        override fun onSurfaceTextureUpdated(texture: SurfaceTexture) {}
    }

    private val frameCallback = IFrameCallback { frame ->
        val seg = segmentor ?: return@IFrameCallback
        if (latestPre.get() != null) return@IFrameCallback
        try {
            val bmp = convertFrameToBitmap(frame)
            if (bmp != null) {
                fragmentRender.updateReferenceSize(bmp.width, bmp.height)
                val pre = fragmentRender.getCroppedBitmapWithRect(bmp)?.let { (cropped, rect) ->
                    seg.preprocess(cropped, 0, originalForDisplay = bmp, cropRectInOriginal = rect)
                } ?: seg.preprocess(bmp, 0, originalForDisplay = bmp, cropRectInOriginal = null)
                latestPre.set(pre)
            }
        } catch (_: Exception) { }
    }

    private fun convertFrameToBitmap(frame: ByteBuffer): Bitmap? {
        return try {
            val size = frame.remaining()
            val bytes = ByteArray(size)
            frame.get(bytes)
            android.graphics.BitmapFactory.decodeByteArray(bytes, 0, size)
        } catch (_: Exception) { null }
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        return inflater.inflate(R.layout.fragment_camera, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        textureView = view.findViewById(R.id.surface)
        textureView.surfaceTextureListener = surfaceListener
        fragmentRender = view.findViewById(R.id.fragmentRender)
        initUsbMonitor()
    }

    private fun initUsbMonitor() {
        if (usbMonitor == null) {
            usbMonitor = USBMonitor(requireContext(), onDeviceConnectListener)
            usbMonitor?.register()
        }
    }

    override fun onResume() { 
        super.onResume()
        hiddenState = null
        textureView.surfaceTextureListener = surfaceListener
    }
    
    override fun onPause() { 
        closeCamera()
        stopPipeline()
        stopBackgroundThread()
        super.onPause()
    }
    
    override fun onDestroy() {
        usbMonitor?.unregister()
        usbMonitor?.destroy()
        usbMonitor = null
        super.onDestroy()
    }

    private val onDeviceConnectListener = object : USBMonitor.OnDeviceConnectListener {
        override fun onAttach(device: UsbDevice?) {
            Log.d(TAG, "USB device attached: ${device?.deviceName}")
            usbMonitor?.requestPermission(device)
        }
        override fun onConnect(device: UsbDevice?, ctrlBlock: USBMonitor.UsbControlBlock?, createNew: Boolean) {
            Log.d(TAG, "USB device connected: ${device?.deviceName}")
            openUvcCamera(ctrlBlock)
        }
        override fun onDisconnect(device: UsbDevice?, ctrlBlock: USBMonitor.UsbControlBlock?) {
            Log.d(TAG, "USB device disconnected: ${device?.deviceName}")
            closeCamera()
        }
        override fun onDettach(device: UsbDevice?) {
            Log.d(TAG, "USB device detached: ${device?.deviceName}")
            closeCamera()
        }
        override fun onCancel(device: UsbDevice?) {
            Log.d(TAG, "USB permission cancelled: ${device?.deviceName}")
        }
    }

    private fun openUvcCamera(ctrlBlock: USBMonitor.UsbControlBlock?) {
        if (ctrlBlock == null) {
            Log.e(TAG, "Control block is null")
            return
        }
        try {
            Log.d(TAG, "Opening UVC camera")
            uvcCamera = UVCCamera()
            uvcCamera?.open(ctrlBlock)
            uvcCamera?.setPreviewSize(640, 480, UVCCamera.FRAME_FORMAT_MJPEG)
            val surface = Surface(textureView.surfaceTexture)
            uvcCamera?.setPreviewDisplay(surface)
            uvcCamera?.setFrameCallback(frameCallback, UVCCamera.PIXEL_FORMAT_YUV420SP)
            uvcCamera?.startPreview()
            Log.d(TAG, "UVC camera preview started")
            android.widget.Toast.makeText(requireContext(), "USB camera connected", android.widget.Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open UVC camera", e)
            e.printStackTrace()
            android.widget.Toast.makeText(requireContext(), "Failed to open USB camera: ${e.message}", android.widget.Toast.LENGTH_LONG).show()
        }
    }

    private fun openCamera() {
        val devices = usbMonitor?.deviceList
        Log.d(TAG, "openCamera called, devices: ${devices?.size ?: 0}")
        if (devices.isNullOrEmpty()) {
            android.widget.Toast.makeText(requireContext(), "No USB devices found", android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        devices.forEach { device ->
            Log.d(TAG, "Found device: ${device.deviceName}, VID: ${device.vendorId}, PID: ${device.productId}")
        }
        devices.firstOrNull()?.let { device ->
            Log.d(TAG, "Requesting permission for: ${device.deviceName}")
            usbMonitor?.requestPermission(device)
        }
    }

    private fun closeCamera() {
        uvcCamera?.stopPreview()
        uvcCamera?.destroy()
        uvcCamera = null
    }
    
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("UsbCameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }
    
    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        backgroundThread = null
        backgroundHandler = null
    }

    private fun startPipeline() {
        if (pipelineRunning) return
        pipelineRunning = true
        inferThread = HandlerThread("UsbCamInfer").also { it.start() }
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
                            val orig = pre.originalForDisplay
                            fr.post { fr.render(outBmp, orig, 0f, 0L, 0L) }
                        }
                    } catch (_: Exception) { }
                }
                inferHandler?.post(this)
            }
        })
    }

    private fun stopPipeline() {
        pipelineRunning = false
        inferHandler?.removeCallbacksAndMessages(null)
        inferThread?.quitSafely()
        inferThread = null
        inferHandler = null
        latestPre.set(null)
        hiddenState = null
    }

    fun startManually() {
        if (pipelineRunning && uvcCamera != null) return
        hiddenState = null
        if (backgroundThread == null) startBackgroundThread()
        startPipeline()
        initUsbMonitor()
        mainHandler.postDelayed({
            if (textureView.isAvailable) openCamera() else textureView.surfaceTextureListener = surfaceListener
        }, 300)
    }

    private val mainHandler = Handler(android.os.Looper.getMainLooper())

    fun stopManually() {
        closeCamera()
        stopPipeline()
        stopBackgroundThread()
        hiddenState = null
    }

    fun setSegmentor(seg: TfLiteSegmentor) {
        val wasRunning = pipelineRunning
        segmentor = seg
        inferHandler?.removeCallbacksAndMessages(null)
        latestPre.set(null)
        hiddenState = null
        stopPipeline()
        if (wasRunning) startPipeline()
    }
}