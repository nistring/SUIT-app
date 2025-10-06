package com.quicinc.semanticsegmentation

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.*
import android.view.TextureView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import org.opencv.android.OpenCVLoader
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

class UsbCameraFragment : Fragment() {
    private lateinit var textureView: TextureView
    private lateinit var fragmentRender: FragmentRender
    private var segmentor: TfLiteSegmentor? = null
    private var cameraId: String? = null
    private var captureSession: CameraCaptureSession? = null
    private var cameraDevice: CameraDevice? = null
    private var previewSize: Size? = null
    private var sensorOrientation: Int = 0
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    private val cameraLock = Semaphore(1)

    // array of hidden states (one per RNN output)
    private var hiddenState: Array<FloatArray>? = null

    companion object { fun create(seg: TfLiteSegmentor) = UsbCameraFragment().apply { segmentor = seg } }

    private val surfaceListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(texture: SurfaceTexture, width: Int, height: Int) { openCamera() }
        override fun onSurfaceTextureSizeChanged(texture: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(texture: SurfaceTexture): Boolean = true
        override fun onSurfaceTextureUpdated(texture: SurfaceTexture) {}
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        OpenCVLoader.initDebug()
        return inflater.inflate(R.layout.fragment_camera, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        textureView = view.findViewById(R.id.surface)
        textureView.surfaceTextureListener = surfaceListener
        fragmentRender = view.findViewById(R.id.fragmentRender)
    }

    override fun onResume() { 
        super.onResume(); 
        // Reset hidden state when starting/resuming camera session
        hiddenState = null
        startBackgroundThread(); 
        if (textureView.isAvailable) openCamera() else textureView.surfaceTextureListener = surfaceListener 
    }
    override fun onPause() { closeCamera(); stopBackgroundThread(); super.onPause() }

    private fun requestCameraPermission() { registerForActivityResult(ActivityResultContracts.RequestPermission()) { }.launch(Manifest.permission.CAMERA) }

    private fun setUpCameraOutputs() {
        val activity = activity ?: return
        val manager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        for (id in manager.cameraIdList) {
            val chars = manager.getCameraCharacteristics(id)
            val facing = chars.get(CameraCharacteristics.LENS_FACING)
            if (facing != CameraCharacteristics.LENS_FACING_EXTERNAL) continue
            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: continue
            sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
            val sizes = map.getOutputSizes(SurfaceTexture::class.java)
            val seg = segmentor ?: continue
            val targetW: Int; val targetH: Int
            if (sensorOrientation == 90 || sensorOrientation == 270) { targetW = seg.getInputHeight(); targetH = seg.getInputWidth() }
            else { targetW = seg.getInputWidth(); targetH = seg.getInputHeight() }
            previewSize = sizes.minByOrNull { (it.width - targetW).toLong() * (it.height - targetH) } ?: sizes.first()
            cameraId = id
            return
        }
        // Fallback to back camera if no external found
        for (id in manager.cameraIdList) {
            val chars = manager.getCameraCharacteristics(id)
            if (chars.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK) {
                val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: continue
                sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
                val sizes = map.getOutputSizes(SurfaceTexture::class.java)
                previewSize = sizes.first()
                cameraId = id
                return
            }
        }
    }

    private fun openCamera() {
        val act = activity ?: return
        if (ContextCompat.checkSelfPermission(act, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) { requestCameraPermission(); return }
        setUpCameraOutputs()
        val manager = act.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            if (!cameraLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) throw RuntimeException("Timeout acquiring camera lock")
            manager.openCamera(cameraId!!, stateCallback, backgroundHandler)
        } catch (e: Exception) { e.printStackTrace() }
    }

    private fun closeCamera() { try { cameraLock.acquire(); captureSession?.close(); captureSession=null; cameraDevice?.close(); cameraDevice=null } finally { cameraLock.release() } }
    private fun startBackgroundThread() { backgroundThread = HandlerThread("UsbCameraBackground").also { it.start() }; backgroundHandler = Handler(backgroundThread!!.looper) }
    private fun stopBackgroundThread() { backgroundThread?.quitSafely(); backgroundThread=null; backgroundHandler=null }

    private val stateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) { cameraLock.release(); cameraDevice = camera; createPreviewSession() }
        override fun onDisconnected(camera: CameraDevice) { cameraLock.release(); camera.close(); cameraDevice=null }
        override fun onError(camera: CameraDevice, error: Int) { cameraLock.release(); camera.close(); cameraDevice=null; activity?.finish() }
    }

    private fun createPreviewSession() {
        try {
            val texture = textureView.surfaceTexture ?: return
            val size = previewSize ?: return
            texture.setDefaultBufferSize(size.width, size.height)
            val surface = Surface(texture)
            val builder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply { addTarget(surface) }
            cameraDevice!!.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) { captureSession = session; builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE); session.setRepeatingRequest(builder.build(), captureCallback, backgroundHandler) }
                override fun onConfigureFailed(session: CameraCaptureSession) {}
            }, backgroundHandler)
        } catch (e: Exception) { e.printStackTrace() }
    }

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            val seg = segmentor ?: return
            val bmp: Bitmap = textureView.bitmap ?: return
            // pass and update hidden state array
            val (outBmp, newHidden) = seg.predict(bmp, sensorOrientation, hiddenState)
            hiddenState = newHidden
            fragmentRender.render(outBmp, 0f, seg.lastInferTime, seg.lastPreTime, seg.lastPostTime)
        }
    }
}