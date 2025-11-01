// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation

import android.os.Bundle
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.ProgressBar
import android.widget.Button
import android.hardware.camera2.CameraManager
import android.content.Context
import android.widget.Toast
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.hardware.usb.UsbConstants
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.commit
import com.quicinc.tflite.AIHubDefaults
import com.quicinc.tflite.TFLiteHelpers
import java.io.IOException
import java.security.NoSuchAlgorithmException
import java.util.concurrent.Executors
import org.opencv.android.OpenCVLoader
import androidx.appcompat.app.AlertDialog
import android.view.Gravity
import android.view.ViewGroup
import android.widget.FrameLayout

class MainActivity : AppCompatActivity() {
    private lateinit var progressBar: ProgressBar
    private var segmentor: TfLiteSegmentor? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val pickVideo = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null && segmentor != null) {
            supportFragmentManager.commit {
                replace(R.id.main_content, VideoFileFragment.create(segmentor!!, uri))
            }
        } else {
            Toast.makeText(this, "No video selected.", Toast.LENGTH_SHORT).show()
        }
    }
    private var selectedModelAsset: String? = null
    // Track USB inference state and availability
    private var isUsbRunning: Boolean = false
    private var hasExtCam: Boolean = false
    // New: Track if USB camera successfully connected
    private var isUsbCameraConnected: Boolean = false

    // Floating overlay buttons
    private var btnStartStop: Button? = null
    private var btnModel: Button? = null
    // New: Full mode toggle button + state
    private var btnFull: Button? = null
    private var isFullMode: Boolean = false
    // New: Show/Hide inference toggle
    private var btnShow: Button? = null
    private var isShowInference: Boolean = true

    companion object {
        private const val MENU_PICK_MODEL = 1
        private const val MENU_START_USB = 2
        private const val MENU_EXIT = 3
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main_activity)

        // Load OpenCV native libs early to prevent UnsatisfiedLinkError from background threads
        try {
            if (!OpenCVLoader.initDebug()) {
                System.loadLibrary("opencv_java4")
            }
        } catch (_: Throwable) { /* let TfLiteSegmentor fallback handle hard failure */ }

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        enterImmersiveMode()
        progressBar = findViewById(R.id.indeterminateBar)
        // Ensure overlay buttons exist and are on top even before segmentor initializes
        ensureStartStopButton()
        ensureModelButton()
        ensureOverlayButton() // keep existing Pick Video button available
        // New: ensure Full toggle button
        ensureFullButton()
        // New: ensure Show toggle button (below Full)
        ensureShowButton()
        createTFLiteClassifiersAsync()
    }

    // Create/ensure the floating Start/Stop button
    private fun ensureStartStopButton(): Button {
        val root = findViewById<ViewGroup>(android.R.id.content) as FrameLayout
        if (btnStartStop != null) {
            btnStartStop!!.visibility = View.VISIBLE
            btnStartStop!!.bringToFront()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                btnStartStop!!.elevation = 16f
                btnStartStop!!.translationZ = 16f
            }
            return btnStartStop!!
        }
        val btn = Button(this).apply {
            id = View.generateViewId()
            text = "Start"
            alpha = 0.95f
            isAllCaps = false
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        val lp = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.END or Gravity.BOTTOM
        ).apply { setMargins(m, m, m, m) }
        root.addView(btn, lp)
        btn.bringToFront()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            btn.elevation = 16f
            btn.translationZ = 16f
        }
        btnStartStop = btn
        return btn
    }

    // Create/ensure the floating Model button
    private fun ensureModelButton(): Button {
        val root = findViewById<ViewGroup>(android.R.id.content) as FrameLayout
        if (btnModel != null) {
            btnModel!!.visibility = View.VISIBLE
            btnModel!!.bringToFront()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                btnModel!!.elevation = 16f
                btnModel!!.translationZ = 16f
            }
            return btnModel!!
        }
        val btn = Button(this).apply {
            id = View.generateViewId()
            text = "Model"
            alpha = 0.95f
            isAllCaps = false
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        val lp = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.START or Gravity.BOTTOM
        ).apply { setMargins(m, m, m, m) }
        root.addView(btn, lp)
        btn.bringToFront()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            btn.elevation = 16f
            btn.translationZ = 16f
        }
        btnModel = btn
        return btn
    }

    // Ensure we have a visible overlay button with id btn_pick_video; create it if missing
    private fun ensureOverlayButton(): Button {
        val existing = findViewById<Button>(R.id.btn_pick_video)
        if (existing != null) {
            existing.visibility = View.VISIBLE
            // Make sure it's drawn above fragments
            existing.bringToFront()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                existing.elevation = 16f
                existing.translationZ = 16f
            }
            return existing
        }
        // Create an overlay button dynamically on the root content
        val root = findViewById<ViewGroup>(android.R.id.content)
        val btn = Button(this).apply {
            id = R.id.btn_pick_video
            text = "Start"
            alpha = 0.95f
            isAllCaps = false
        }
        val lp = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.END or Gravity.BOTTOM
        ).apply {
            val m = (16 * resources.displayMetrics.density).toInt()
            setMargins(m, m, m, m)
        }
        (root as FrameLayout).addView(btn, lp)
        btn.bringToFront()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            btn.elevation = 16f
            btn.translationZ = 16f
        }
        return btn
    }

    // Create/ensure the floating Full toggle button
    private fun ensureFullButton(): Button {
        val root = findViewById<ViewGroup>(android.R.id.content) as FrameLayout
        if (btnFull != null) {
            btnFull!!.visibility = View.VISIBLE
            btnFull!!.bringToFront()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                btnFull!!.elevation = 18f
                btnFull!!.translationZ = 18f
            }
            return btnFull!!
        }
        val btn = Button(this).apply {
            id = View.generateViewId()
            text = "Full"
            alpha = 0.95f
            isAllCaps = false
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        val lp = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.END
        ).apply { setMargins(m, m, m, m) }
        root.addView(btn, lp)
        btn.bringToFront()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            btn.elevation = 18f
            btn.translationZ = 18f
        }
        btnFull = btn
        return btn
    }

    // New: Create/ensure the floating Show/Hide button placed below Full button
    private fun ensureShowButton(): Button {
        val root = findViewById<ViewGroup>(android.R.id.content) as FrameLayout
        if (btnShow != null) {
            btnShow!!.visibility = View.VISIBLE
            btnShow!!.bringToFront()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                btnShow!!.elevation = 17f
                btnShow!!.translationZ = 17f
            }
            return btnShow!!
        }
        val btn = Button(this).apply {
            id = View.generateViewId()
            text = "Hide"
            alpha = 0.95f
            isAllCaps = false
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        // Position below Full button by adding extra top margin (~56dp)
        val below = (56 * resources.displayMetrics.density).toInt()
        val lp = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.END
        ).apply { setMargins(m, m + below, m, m) }
        root.addView(btn, lp)
        btn.bringToFront()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            btn.elevation = 17f
            btn.translationZ = 17f
        }
        btnShow = btn
        return btn
    }

    private fun applyFullModeToRender() {
        findViewById<FragmentRender>(R.id.fragmentRender)?.setFullMode(isFullMode)
    }
    // New: forward show/hide to renderer
    private fun applyShowModeToRender() {
        findViewById<FragmentRender>(R.id.fragmentRender)?.setShowInference(isShowInference)
    }

    private fun enterImmersiveMode() {
        // Hide system bars for full-screen experience
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
            val controller = window.insetsController
            controller?.let {
                it.hide(WindowInsets.Type.systemBars())
                it.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_FULLSCREEN
            )
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            enterImmersiveMode()
        }
    }

    private fun setLoadingUI(loading: Boolean) {
        runOnUiThread { progressBar.visibility = if (loading) View.VISIBLE else View.INVISIBLE }
    }

    private fun createTFLiteClassifiersAsync() {
        setLoadingUI(true)
        backgroundExecutor.execute {
            try {
                // Close previous segmentor if present
                try { segmentor?.close() } catch (_: Exception) {}
                segmentor = null

                // Resolve model asset: user-selected or default
                val tfLiteModelAsset = selectedModelAsset ?: resources.getString(R.string.tfLiteModelAsset)

                // Choose delegates based on environment: QNN+GPU on real device, GPU/CPU on emulator
                val delegates = if (isProbablyEmulator()) {
                    AIHubDefaults.delegatePriorityOrderForDelegates(setOf(TFLiteHelpers.DelegateType.GPUv2))
                } else {
                    AIHubDefaults.delegatePriorityOrderForDelegates(AIHubDefaults.enabledDelegates)
                }
                segmentor = TfLiteSegmentor(
                    context = this,
                    modelPath = tfLiteModelAsset,
                    delegatePriorityOrder = delegates
                )
            } catch (e: IOException) {
                throw RuntimeException(e.message)
            } catch (e: NoSuchAlgorithmException) {
                throw RuntimeException(e.message)
            }
            setLoadingUI(false)
            mainHandler.post {
                val seg = segmentor ?: return@post
                updateActiveFragmentWithSegmentor(seg)
                // Ensure Full button exists and current mode applied
                ensureFullButton()
                ensureShowButton()
                applyFullModeToRender()
                applyShowModeToRender()

                // Cache camera availability for UI logic (deferred check)
                hasExtCam = hasExternalCamera()

                if (supportFragmentManager.findFragmentById(R.id.main_content) == null) {
                    if (isProbablyEmulator()) {
                        pickVideo.launch("video/*")
                    } else if (hasExtCam || detectUvcVideoDevices().isNotEmpty()) {
                        configureMainButton()
                        // Don't show toast here; wait until user actually presses Start
                    } else {
                        configureMainButton()
                        Toast.makeText(this, "Pick a video to run segmentation.", Toast.LENGTH_LONG).show()
                    }
                } else {
                    configureMainButton()
                }
            }
        }
    }

    // Configure overlay buttons every time state changes
    private fun configureMainButton() {
        // Start/Stop button: always visible (unless in full mode)
        val startStop = ensureStartStopButton()
        startStop.visibility = View.VISIBLE
        startStop.isClickable = true
        startStop.isFocusable = true
        startStop.bringToFront()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            startStop.elevation = 16f
            startStop.translationZ = 16f
        }
        // Long press also opens model picker
        startStop.setOnLongClickListener { showModelPicker(); true }

        if (isUsbRunning) {
            startStop.text = "Stop"
            startStop.setOnClickListener { stopUsbAndExit() }
        } else {
            startStop.text = "Start"
            startStop.setOnClickListener { startUsb() }
        }

        // Model button: opens model picker
        val modelBtn = ensureModelButton()
        modelBtn.visibility = View.VISIBLE
        modelBtn.setOnClickListener { showModelPicker() }
        modelBtn.bringToFront()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            modelBtn.elevation = 16f
            modelBtn.translationZ = 16f
        }

        // Keep Pick Video button (if in layout) working as before
        findViewById<Button>(R.id.btn_pick_video)?.apply {
            visibility = View.VISIBLE
            setOnClickListener { pickVideo.launch("video/*") }
            bringToFront()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                elevation = 8f
                translationZ = 8f
            }
        }

        // Full button behavior
        val fullBtn = ensureFullButton()
        fullBtn.visibility = View.VISIBLE
        fullBtn.text = if (isFullMode) "Exit Full" else "Full"
        fullBtn.setOnClickListener {
            isFullMode = !isFullMode
            fullBtn.text = if (isFullMode) "Exit Full" else "Full"
            // Hide other overlay buttons in full mode; show them otherwise
            val vStart = btnStartStop
            val vModel = btnModel
            val vPick = findViewById<Button>(R.id.btn_pick_video)
            if (isFullMode) {
                vStart?.visibility = View.GONE
                vModel?.visibility = View.GONE
                vPick?.visibility = View.GONE
            } else {
                vStart?.visibility = View.VISIBLE
                vModel?.visibility = View.VISIBLE
                vPick?.visibility = View.VISIBLE
                configureMainButton() // refresh callbacks/labels
            }
            applyFullModeToRender()
        }
        fullBtn.bringToFront()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            fullBtn.elevation = 18f
            fullBtn.translationZ = 18f
        }

        // New: Show/Hide inference button behavior (kept visible in full mode)
        val showBtn = ensureShowButton()
        showBtn.visibility = View.VISIBLE
        showBtn.text = if (isShowInference) "Hide" else "Show"
        showBtn.setOnClickListener {
            isShowInference = !isShowInference
            showBtn.text = if (isShowInference) "Hide" else "Show"
            applyShowModeToRender()
        }
        showBtn.bringToFront()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            showBtn.elevation = 17f
            showBtn.translationZ = 17f
        }

        // Apply current states to renderer
        applyShowModeToRender()
        applyFullModeToRender()

        // When in full mode, hide other buttons but KEEP show/hide visible
        if (isFullMode) {
            btnStartStop?.visibility = View.GONE
            btnModel?.visibility = View.GONE
            findViewById<Button>(R.id.btn_pick_video)?.visibility = View.GONE
            // btnShow remains visible to allow toggling while maintaining full-mode size
        } else {
            btnShow?.visibility = View.VISIBLE
        }
    }

    private fun startUsb() {
        val seg = segmentor ?: return
        isUsbRunning = true
        isUsbCameraConnected = false
        supportFragmentManager.commit { replace(R.id.main_content, UsbCameraFragment.create(seg)) }
        mainHandler.post {
            (supportFragmentManager.findFragmentById(R.id.main_content) as? UsbCameraFragment)?.startManually()
            configureMainButton() // refresh label to Stop
        }
        // Ensure renderer picks up current full mode
        mainHandler.post { applyFullModeToRender() }
        Toast.makeText(this, "Searching for USB camera...", Toast.LENGTH_SHORT).show()
    }

    // Public method called by UsbCameraFragment when camera successfully connects
    fun onUsbCameraConnected() {
        isUsbCameraConnected = true
    }

    // Public method called by UsbCameraFragment when camera fails to connect
    fun onUsbCameraFailed(errorMsg: String) {
        if (isUsbRunning && !isUsbCameraConnected) {
            Toast.makeText(this, "Failed to connect USB camera: $errorMsg", Toast.LENGTH_LONG).show()
        }
    }

    private fun stopUsbAndExit() {
        val frag = supportFragmentManager.findFragmentById(R.id.main_content)
        if (frag is UsbCameraFragment) {
            frag.stopManually()
            supportFragmentManager.commit { remove(frag) }
        }
        isUsbRunning = false
        configureMainButton() // refresh label to Start
        // Ensure renderer picks up current full mode (likely false)
        applyFullModeToRender()
        // Ensure renderer picks up current show/hide state
        applyShowModeToRender()
        Toast.makeText(this, "Stopped USB inference.", Toast.LENGTH_SHORT).show()
    }

    private fun updateActiveFragmentWithSegmentor(seg: TfLiteSegmentor) {
        when (val frag = supportFragmentManager.findFragmentById(R.id.main_content)) {
            is UsbCameraFragment -> frag.setSegmentor(seg)
            is VideoFileFragment -> frag.setSegmentor(seg)
            is ScreenCaptureFragment -> frag.setSegmentor(seg)
        }
    }

    private fun showModelPicker() {
        val allFiles: Array<String> = try {
            assets.list("") ?: emptyArray<String>()
        } catch (_: Exception) { 
            emptyArray<String>()
        }
        
        val models = allFiles.filter { it.endsWith(".tflite") || it.endsWith(".bin") }.sorted().toTypedArray()

        if (models.isEmpty()) {
            Toast.makeText(this, "No models found in assets.", Toast.LENGTH_SHORT).show()
            return
        }

        val current = selectedModelAsset ?: runCatching { resources.getString(R.string.tfLiteModelAsset) }.getOrNull()
        val preselect = models.indexOfFirst { it == current }.coerceAtLeast(0)

        val dialog = AlertDialog.Builder(this)
            .setTitle("Select model")
            .setSingleChoiceItems(models, preselect) { dialogInterface, which ->
                selectedModelAsset = models[which]
                Toast.makeText(this, "Loading: ${models[which]}", Toast.LENGTH_SHORT).show()
                dialogInterface.dismiss()
                createTFLiteClassifiersAsync()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun isProbablyEmulator(): Boolean {
        val fingerprint = Build.FINGERPRINT?.lowercase() ?: ""
        val model = Build.MODEL?.lowercase() ?: ""
        val hardware = Build.HARDWARE?.lowercase() ?: ""
        val brand = Build.BRAND?.lowercase() ?: ""
        val device = Build.DEVICE?.lowercase() ?: ""
        return fingerprint.contains("generic") || fingerprint.contains("emulator") ||
                model.contains("emulator") || model.contains("android sdk built for x86") ||
                hardware.contains("goldfish") || hardware.contains("ranchu") ||
                brand.startsWith("generic") && device.startsWith("generic")
    }

    override fun onResume() {
        super.onResume()
        enterImmersiveMode()
        // Ensure renderer and fragment are properly restored
        val frag = supportFragmentManager.findFragmentById(R.id.main_content)
        if (frag is UsbCameraFragment && isUsbRunning) {
            // Fragment handles its own resume and reconnection
            Log.d("MainActivity", "Resumed with active USB fragment")
        }
    }

    override fun onPause() {
        super.onPause()
        // Don't stop USB here; let fragment handle it in its onPause
    }


    private fun hasExternalCamera(): Boolean {
        val cm = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        return try {
            cm.cameraIdList.any { id ->
                cm.getCameraCharacteristics(id).get(android.hardware.camera2.CameraCharacteristics.LENS_FACING) == android.hardware.camera2.CameraCharacteristics.LENS_FACING_EXTERNAL
            }
        } catch (e: Exception) { false }
    }

    private fun listCamera2DevicesForDebug() {
        val cm = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            for (id in cm.cameraIdList) {
                val chars = cm.getCameraCharacteristics(id)
                val facing = chars.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING)
                Log.i("SemanticSeg", "Camera2 device id=$id facing=${facing}")
            }
        } catch (e: Exception) {
            Log.e("SemanticSeg", "Error listing Camera2 devices: ${e.message}")
        }
    }

    private fun detectUvcVideoDevices(): List<UsbDevice> {
        val usb = getSystemService(Context.USB_SERVICE) as UsbManager
        val found = mutableListOf<UsbDevice>()
        for ((_, dev) in usb.deviceList) {
            // Check interfaces for VIDEO class (0x0E)
            for (i in 0 until dev.interfaceCount) {
                val intf = dev.getInterface(i)
                if (intf.interfaceClass == UsbConstants.USB_CLASS_VIDEO) {
                    found.add(dev); break
                }
            }
        }
        return found
    }
}
