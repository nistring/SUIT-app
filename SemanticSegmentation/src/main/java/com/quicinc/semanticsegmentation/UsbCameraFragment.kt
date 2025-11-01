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
import android.util.DisplayMetrics
import android.util.Log
import android.view.*
import android.view.TextureView
import androidx.fragment.app.Fragment
import com.serenegiant.usb.IFrameCallback
import com.serenegiant.usb.USBMonitor
import com.serenegiant.usb.UVCCamera
import org.opencv.core.Mat
import org.opencv.core.CvType
import org.opencv.android.Utils
import org.opencv.imgproc.Imgproc
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
    private var previewSurface: Surface? = null
    @Volatile private var isCameraOpening = false
    @Volatile private var isFragmentDestroyed = false
    private var previewWidth: Int = 1920
    private var previewHeight: Int = 1080
    private val mainHandler = Handler(android.os.Looper.getMainLooper())
    @Volatile private var wasRunningBeforePause = false
    private var consecutiveFailures = 0
    private val maxRetries = 3

    companion object { 
        private const val TAG = "UsbCameraFragment"
        fun create(seg: TfLiteSegmentor) = UsbCameraFragment().apply { segmentor = seg } 
    }

    private val surfaceListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(texture: SurfaceTexture, width: Int, height: Int) {
            Log.d(TAG, "SurfaceTexture available: ${width}x${height}")
        }
        override fun onSurfaceTextureSizeChanged(texture: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(texture: SurfaceTexture): Boolean {
            Log.d(TAG, "SurfaceTexture destroyed")
            return true
        }
        override fun onSurfaceTextureUpdated(texture: SurfaceTexture) {}
    }

    private val frameCallback = IFrameCallback { frame ->
        val seg = segmentor ?: return@IFrameCallback
        if (latestPre.get() != null) return@IFrameCallback
        try {
            val bmp = convertFrameToBitmap(frame) ?: return@IFrameCallback
            fragmentRender.updateReferenceSize(bmp.width, bmp.height)
            val pre = fragmentRender.getCroppedBitmapWithRect(bmp)?.let { (cropped, rect) ->
                seg.preprocess(cropped, 0, originalForDisplay = bmp, cropRectInOriginal = rect)
            } ?: seg.preprocess(bmp, 0, originalForDisplay = bmp, cropRectInOriginal = null)
            latestPre.set(pre)
        } catch (_: Exception) { }
    }

    private fun convertFrameToBitmap(frame: ByteBuffer): Bitmap? {
        return try {
            val bytes = ByteArray(frame.remaining()).also { frame.get(it) }
            android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                ?: run {
                    val yuv = Mat(previewHeight + previewHeight / 2, previewWidth, CvType.CV_8UC1)
                    yuv.put(0, 0, bytes)
                    val rgb = Mat()
                    Imgproc.cvtColor(yuv, rgb, Imgproc.COLOR_YUV2RGB_NV21)
                    Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888).also {
                        Utils.matToBitmap(rgb, it)
                        yuv.release()
                        rgb.release()
                    }
                }
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
            // Auto-detect all USB devices without device filter
            usbMonitor?.register()
        }
    }

    override fun onResume() { 
        super.onResume()
        hiddenState = null
        textureView.surfaceTextureListener = surfaceListener
        if (wasRunningBeforePause && !pipelineRunning) {
            Log.d(TAG, "Resuming, restarting USB pipeline")
            mainHandler.postDelayed({ if (!isFragmentDestroyed) startManually() }, 500)
        }
    }
    
    override fun onPause() { 
        wasRunningBeforePause = pipelineRunning
        closeCamera()
        stopPipeline()
        stopBackgroundThread()
        super.onPause()
    }
    
    override fun onDestroy() {
        isFragmentDestroyed = true
        closeCamera()
        stopPipeline()
        stopBackgroundThread()
        usbMonitor?.unregister()
        usbMonitor?.destroy()
        usbMonitor = null
        super.onDestroy()
    }

    private val onDeviceConnectListener = object : USBMonitor.OnDeviceConnectListener {
        override fun onAttach(device: UsbDevice?) {
            if (isFragmentDestroyed || isCameraOpening) {
                Log.w(TAG, "Fragment destroyed or camera already opening, ignoring attach")
                return
            }
            Log.d(TAG, "USB device attached: ${device?.deviceName}")
            device?.let {
                Log.d(TAG, "  Vendor ID: 0x${it.vendorId.toString(16)}, Product ID: 0x${it.productId.toString(16)}")
                Log.d(TAG, "  Manufacturer: ${it.manufacturerName}, Product: ${it.productName}")
                Log.d(TAG, "  Interface Count: ${it.interfaceCount}")
                repeat(it.interfaceCount) { i ->
                    it.getInterface(i)?.let { intf ->
                        Log.d(TAG, "  Interface $i - Class: 0x${intf.interfaceClass.toString(16)}, SubClass: 0x${intf.interfaceSubclass.toString(16)}")
                    }
                }
            }
            Log.d(TAG, "Requesting USB permission for ${device?.deviceName}")
            usbMonitor?.requestPermission(device)
        }
        
        override fun onConnect(device: UsbDevice?, ctrlBlock: USBMonitor.UsbControlBlock?, createNew: Boolean) {
            if (isFragmentDestroyed) {
                Log.w(TAG, "Fragment destroyed, ignoring connect")
                return
            }
            Log.d(TAG, "USB device connected: ${device?.deviceName}, permission granted")
            if (isCameraOpening) {
                Log.w(TAG, "Camera is already opening, ignoring duplicate connect event")
                return
            }
            ctrlBlock?.let { openUvcCamera(it) } ?: Log.e(TAG, "Control block is null on connect")
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

    private fun openUvcCamera(ctrlBlock: USBMonitor.UsbControlBlock) {
        if (isFragmentDestroyed) {
            isCameraOpening = false
            return
        }
        try {
            uvcCamera = UVCCamera().apply { open(ctrlBlock) }
            consecutiveFailures = 0
            val (w, h) = selectBestPreviewSize()
            previewWidth = w
            previewHeight = h
            uvcCamera?.setPreviewSize(w, h, UVCCamera.FRAME_FORMAT_MJPEG)
            
            textureView.surfaceTexture?.let { surfaceTexture ->
                previewSurface = Surface(surfaceTexture)
                uvcCamera?.apply {
                    setPreviewDisplay(previewSurface)
                    setFrameCallback(frameCallback, UVCCamera.PIXEL_FORMAT_YUV420SP)
                    startPreview()
                }
                (requireActivity() as? MainActivity)?.onUsbCameraConnected()
                if (!isFragmentDestroyed) {
                    android.widget.Toast.makeText(requireContext(), "USB camera connected at ${w}x${h}", android.widget.Toast.LENGTH_SHORT).show()
                }
            } ?: run {
                (requireActivity() as? MainActivity)?.onUsbCameraFailed("SurfaceTexture unavailable")
                if (!isFragmentDestroyed) {
                    android.widget.Toast.makeText(requireContext(), "SurfaceTexture unavailable", android.widget.Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            previewSurface?.release()
            previewSurface = null
            
            val isTransient = e.message?.let { it.contains("result=-99") || it.contains("LIBUSB") || it.contains("release interface") } ?: false
            
            if (isTransient && consecutiveFailures < maxRetries && !isFragmentDestroyed) {
                consecutiveFailures++
                val delayMs = (500L * Math.pow(1.5, consecutiveFailures.toDouble())).toLong().coerceAtMost(5000)
                Log.d(TAG, "Transient error, retry in ${delayMs}ms (${consecutiveFailures}/$maxRetries)")
                mainHandler.postDelayed({
                    usbMonitor?.deviceList?.firstOrNull()?.let { usbMonitor?.requestPermission(it) }
                }, delayMs)
            } else if (!isTransient) {
                (requireActivity() as? MainActivity)?.onUsbCameraFailed(e.message ?: "Unknown error")
                if (!isFragmentDestroyed) {
                    android.widget.Toast.makeText(requireContext(), "USB error: ${e.message}", android.widget.Toast.LENGTH_LONG).show()
                }
            }
        } finally {
            isCameraOpening = false
        }
    }

    private fun selectBestPreviewSize(): Pair<Int, Int> {
        val maxWidth = 1920
        val maxHeight = 1080
        
        return try {
            uvcCamera?.supportedSizeList?.takeIf { it.isNotEmpty() }?.let { supportedSizes ->
                Log.d(TAG, "Camera supports ${supportedSizes.size} sizes")
                supportedSizes.forEach { Log.d(TAG, "  Supported: ${it.width}x${it.height}") }
                
                val bestSize = supportedSizes.filter { it.width <= maxWidth && it.height <= maxHeight }
                    .maxByOrNull { it.width * it.height } ?: supportedSizes[0]
                
                Log.d(TAG, "Selected camera preview size: ${bestSize.width}x${bestSize.height}")
                Pair(bestSize.width, bestSize.height)
            } ?: run {
                Log.d(TAG, "Using max output resolution: ${maxWidth}x${maxHeight}")
                Pair(maxWidth, maxHeight)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not query supported sizes: ${e.message}")
            Pair(maxWidth, maxHeight)
        }
    }

    private fun closeCamera() {
        uvcCamera?.stopPreview()
        uvcCamera?.destroy()
        previewSurface?.release()
        previewSurface = null
        uvcCamera = null
        isCameraOpening = false
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
                val pre = latestPre.getAndSet(null) ?: run { inferHandler?.post(this); return }
                
                try {
                    val producerStart = System.nanoTime()
                    val inf = seg.infer(pre, hiddenState)
                    val producerEnd = System.nanoTime()
                    val producerTime = producerEnd - producerStart
                    
                    val consumerStart = System.nanoTime()
                    hiddenState = inf.newHidden
                    val outBmp = if (pre.cropRectInOriginal != null && pre.originalForDisplay != null) {
                        val rect = pre.cropRectInOriginal
                        val cropSeg = seg.postprocessToBitmap(inf, rect.width(), rect.height(), pre.sensorOrientation)
                        if (fragmentRender.isFullMode()) {
                            cropSeg
                        } else {
                            Bitmap.createBitmap(pre.originalForDisplay.width, pre.originalForDisplay.height, Bitmap.Config.ARGB_8888).apply {
                                Canvas(this).apply {
                                    drawBitmap(pre.originalForDisplay, 0f, 0f, null)
                                    drawBitmap(cropSeg, rect.left.toFloat(), rect.top.toFloat(), null)
                                }
                            }
                        }
                    } else {
                        seg.postprocessToBitmap(inf, pre.viewW, pre.viewH, pre.sensorOrientation)
                    }
                    val consumerEnd = System.nanoTime()
                    val consumerTime = consumerEnd - consumerStart
                    
                    fragmentRender.post { fragmentRender.render(outBmp, pre.originalForDisplay, 0f, consumerTime, producerTime) }
                } catch (_: Exception) { }
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
        mainHandler.postDelayed({
            usbMonitor?.deviceList?.firstOrNull()?.let {
                Log.d(TAG, "startManually: Found device, requesting permission")
                usbMonitor?.requestPermission(it)
            } ?: Log.w(TAG, "startManually: No USB devices found")
        }, 300)
    }

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