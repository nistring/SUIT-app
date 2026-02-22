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
    private var isUsbRunning: Boolean = false
    private var hasExtCam: Boolean = false
    private var isFullMode: Boolean = false
    private var isShowInference: Boolean = true

    private var btnStartStop: Button? = null
    private var btnModel: Button? = null
    private var btnFull: Button? = null
    private var btnShow: Button? = null
    private var btnInit: Button? = null
    private var btnLicense: Button? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main_activity)

        try {
            if (!OpenCVLoader.initDebug()) System.loadLibrary("opencv_java4")
        } catch (_: Throwable) {}

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        enterImmersiveMode()
        progressBar = findViewById(R.id.indeterminateBar)
        
        ensureButton { btnStartStop = it }
        ensureButton { btnModel = it }
        ensureButton { btnFull = it }
        ensureButton { btnShow = it }
        
        downloadWeightsIfNeeded { createTFLiteClassifiersAsync() }
    }

    private fun applyElevation(btn: Button, elevation: Float) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            btn.elevation = elevation
            btn.translationZ = elevation
        }
    }

    private fun ensureButton(assign: (Button) -> Unit): Button? {
        return null // placeholder; individual button creation follows below
    }

    private fun ensureStartStopButton(): Button {
        if (btnStartStop != null) {
            btnStartStop!!.apply { visibility = View.VISIBLE; bringToFront(); applyElevation(this, 16f) }
            return btnStartStop!!
        }
        val btn = Button(this).apply {
            id = View.generateViewId()
            text = "Start"
            alpha = 0.95f
            isAllCaps = false
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.END or Gravity.BOTTOM).apply { setMargins(m, m, m, m) }
        (findViewById<ViewGroup>(android.R.id.content) as FrameLayout).addView(btn, lp)
        btn.bringToFront()
        applyElevation(btn, 16f)
        btnStartStop = btn
        return btn
    }

    private fun ensureModelButton(): Button {
        if (btnModel != null) {
            btnModel!!.apply { visibility = View.VISIBLE; bringToFront(); applyElevation(this, 16f) }
            return btnModel!!
        }
        val btn = Button(this).apply {
            id = View.generateViewId()
            text = "Model"
            alpha = 0.95f
            isAllCaps = false
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.START or Gravity.BOTTOM).apply { setMargins(m, m, m, m) }
        (findViewById<ViewGroup>(android.R.id.content) as FrameLayout).addView(btn, lp)
        btn.bringToFront()
        applyElevation(btn, 16f)
        btnModel = btn
        return btn
    }

    private fun ensureFullButton(): Button {
        if (btnFull != null) {
            btnFull!!.apply { visibility = View.VISIBLE; bringToFront(); applyElevation(this, 18f) }
            return btnFull!!
        }
        val btn = Button(this).apply {
            id = View.generateViewId()
            text = "Full"
            alpha = 0.95f
            isAllCaps = false
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP or Gravity.END).apply { setMargins(m, m, m, m) }
        (findViewById<ViewGroup>(android.R.id.content) as FrameLayout).addView(btn, lp)
        btn.bringToFront()
        applyElevation(btn, 18f)
        btnFull = btn
        return btn
    }

    private fun ensureShowButton(): Button {
        if (btnShow != null) {
            btnShow!!.apply { visibility = View.VISIBLE; bringToFront(); applyElevation(this, 17f) }
            return btnShow!!
        }
        val btn = Button(this).apply {
            id = View.generateViewId()
            text = "Hide"
            alpha = 0.95f
            isAllCaps = false
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        val below = (56 * resources.displayMetrics.density).toInt()
        val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP or Gravity.END).apply { setMargins(m, m + below, m, m) }
        (findViewById<ViewGroup>(android.R.id.content) as FrameLayout).addView(btn, lp)
        btn.bringToFront()
        applyElevation(btn, 17f)
        btnShow = btn
        return btn
    }

    private fun ensureOverlayButton(): Button {
        val existing = findViewById<Button>(R.id.btn_pick_video)
        if (existing != null) {
            existing.apply { visibility = View.VISIBLE; bringToFront(); applyElevation(this, 8f) }
            return existing
        }
        val btn = Button(this).apply {
            id = R.id.btn_pick_video
            text = "Start"
            alpha = 0.95f
            isAllCaps = false
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.END or Gravity.BOTTOM).apply { setMargins(m, m, m, m) }
        (findViewById<ViewGroup>(android.R.id.content) as FrameLayout).addView(btn, lp)
        btn.bringToFront()
        applyElevation(btn, 8f)
        return btn
    }

    private fun applyFullModeToRender() {
        findViewById<FragmentRender>(R.id.fragmentRender)?.setFullMode(isFullMode)
    }

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

    private fun downloadWeightsIfNeeded(onComplete: () -> Unit) {
        val device = (Build.DEVICE ?: "").lowercase()
        val url = when {
            "gts8" in device -> "https://drive.google.com/uc?export=download&id=1AAqiqXwA-a_oKZrhRsAQWBSsqwe58QJU"
            "gts9" in device -> "https://drive.google.com/uc?export=download&id=1EjWHqzsinD8yNrnNN0l98nD8UFJyfRZW"
            else -> { onComplete(); return }
        }
        val dest = java.io.File(filesDir, "SegFormerB0_ReLU.tflite")
        if (dest.exists()) { onComplete(); return }
        AlertDialog.Builder(this)
            .setTitle("Download Model")
            .setMessage("Device-specific model weights need to be downloaded (~19 MB). Proceed?")
            .setPositiveButton("Download") { _, _ ->
                setLoadingUI(true)
                backgroundExecutor.execute {
                    try {
                        java.net.URL(url).openStream().use { inp -> dest.outputStream().use { inp.copyTo(it) } }
                    } catch (e: Exception) {
                        Log.e("MainActivity", "Weight download failed", e)
                    }
                    mainHandler.post { onComplete() }
                }
            }
            .setNegativeButton("Cancel") { _, _ -> onComplete() }
            .setCancelable(false)
            .show()
    }

    private fun setLoadingUI(loading: Boolean) {
        runOnUiThread { progressBar.visibility = if (loading) View.VISIBLE else View.INVISIBLE }
    }

    private fun createTFLiteClassifiersAsync() {
        setLoadingUI(true)
        backgroundExecutor.execute {
            try {
                segmentor?.close()
            } catch (_: Exception) {}
            segmentor = null

            val tfLiteModelAsset = selectedModelAsset ?: resources.getString(R.string.tfLiteModelAsset)
            val delegates = if (isProbablyEmulator()) {
                AIHubDefaults.delegatePriorityOrderForDelegates(setOf(TFLiteHelpers.DelegateType.GPUv2))
            } else {
                AIHubDefaults.delegatePriorityOrderForDelegates(AIHubDefaults.enabledDelegates)
            }
            
            segmentor = TfLiteSegmentor(this, tfLiteModelAsset, delegates)
            setLoadingUI(false)
            
            mainHandler.post {
                val seg = segmentor ?: return@post
                updateActiveFragmentWithSegmentor(seg)
                hasExtCam = hasExternalCamera()
                
                if (supportFragmentManager.findFragmentById(R.id.main_content) == null) {
                    if (isProbablyEmulator()) {
                        pickVideo.launch("video/*")
                    } else if (hasExtCam || detectUvcVideoDevices().isNotEmpty()) {
                        configureMainButton()
                    } else {
                        configureMainButton()
                        Toast.makeText(this@MainActivity, "Pick a video to run segmentation.", Toast.LENGTH_LONG).show()
                    }
                } else {
                    configureMainButton()
                }
            }
        }
    }

    private fun ensureInitButton(): Button {
        if (btnInit != null) {
            btnInit!!.apply { visibility = View.VISIBLE; bringToFront(); applyElevation(this, 17f) }
            return btnInit!!
        }
        val btn = Button(this).apply {
            id = View.generateViewId()
            text = "Init"
            alpha = 0.95f
            isAllCaps = false
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        val initOffset = (56 * 2 * resources.displayMetrics.density).toInt()  // Below Hide button
        val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP or Gravity.END).apply { setMargins(m, m + initOffset, m, m) }
        (findViewById<ViewGroup>(android.R.id.content) as FrameLayout).addView(btn, lp)
        btn.bringToFront()
        applyElevation(btn, 17f)
        btnInit = btn
        return btn
    }

    private fun ensureLicenseButton(): Button {
        if (btnLicense != null) {
            btnLicense!!.apply { visibility = View.VISIBLE; bringToFront(); applyElevation(this, 16f) }
            return btnLicense!!
        }
        val btn = Button(this).apply {
            id = View.generateViewId()
            text = "License"
            alpha = 0.95f
            isAllCaps = false
            textSize = 12f
        }
        val m = (16 * resources.displayMetrics.density).toInt()
        val licenseOffset = (56 * 3 * resources.displayMetrics.density).toInt()  // Below Init button
        val lp = FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP or Gravity.END).apply { setMargins(m, m + licenseOffset, m, m) }
        (findViewById<ViewGroup>(android.R.id.content) as FrameLayout).addView(btn, lp)
        btn.bringToFront()
        applyElevation(btn, 16f)
        btnLicense = btn
        return btn
    }

    private fun configureMainButton() {
        val startStop = ensureStartStopButton().apply {
            visibility = View.VISIBLE
            isClickable = true
            setOnLongClickListener { showModelPicker(); true }
        }
        
        if (isUsbRunning) {
            startStop.text = "Stop"
            startStop.setOnClickListener { stopUsbAndExit() }
        } else {
            startStop.text = "Start"
            startStop.setOnClickListener { startUsb() }
        }

        ensureModelButton().apply {
            visibility = View.VISIBLE
            setOnClickListener { showModelPicker() }
        }

        ensureOverlayButton().apply {
            visibility = View.VISIBLE
            setOnClickListener { pickVideo.launch("video/*") }
        }

        ensureFullButton().apply {
            visibility = View.VISIBLE
            text = if (isFullMode) "Exit Full" else "Full"
            setOnClickListener {
                isFullMode = !isFullMode
                text = if (isFullMode) "Exit Full" else "Full"
                toggleOverlayButtonsVisibility()
                applyFullModeToRender()
            }
        }

        ensureShowButton().apply {
            visibility = View.VISIBLE
            text = if (isShowInference) "Hide" else "Show"
            setOnClickListener {
                isShowInference = !isShowInference
                text = if (isShowInference) "Hide" else "Show"
                applyShowModeToRender()
            }
        }

        // NEW: Init button
        ensureInitButton().apply {
            visibility = View.VISIBLE
            setOnClickListener {
                initializeBBox()
            }
        }

        // NEW: License button
        ensureLicenseButton().apply {
            visibility = View.VISIBLE
            setOnClickListener {
                showLicenseDialog()
            }
        }

        toggleOverlayButtonsVisibility()
    }

    private fun showLicenseDialog() {
        try {
            val licenseText = assets.open("LICENSE.txt").bufferedReader().use { it.readText() }
            val modificationHeader = "This work has been modified for real-time brachial plexus segmentation task.\n\n"
            
            val scrollView = android.widget.ScrollView(this).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    (400 * resources.displayMetrics.density).toInt()
                )
            }
            
            val textView = android.widget.TextView(this).apply {
                text = modificationHeader + licenseText
                textSize = 10f
                setPadding(16, 16, 16, 16)
                setTextIsSelectable(true)
            }
            
            scrollView.addView(textView)
            
            AlertDialog.Builder(this)
                .setTitle("License")
                .setView(scrollView)
                .setPositiveButton("OK", null)
                .show()
        } catch (e: Exception) {
            Toast.makeText(this, "Could not load license file: ${e.message}", Toast.LENGTH_SHORT).show()
            Log.e("MainActivity", "Error loading license", e)
        }
    }

    private fun initializeBBox() {
        val frag = supportFragmentManager.findFragmentById(R.id.main_content)
        when (frag) {
            is UsbCameraFragment -> frag.requestBBoxInit()
            is VideoFileFragment -> frag.requestBBoxInit()
        }
    }

    private fun toggleOverlayButtonsVisibility() {
        val visibility = if (isFullMode) View.GONE else View.VISIBLE
        btnStartStop?.visibility = visibility
        btnModel?.visibility = visibility
        btnInit?.visibility = visibility
        btnLicense?.visibility = visibility  // NEW
        findViewById<Button>(R.id.btn_pick_video)?.visibility = visibility
    }

    private fun startUsb() {
        val seg = segmentor ?: return
        isUsbRunning = true
        supportFragmentManager.commit { replace(R.id.main_content, UsbCameraFragment.create(seg)) }
        mainHandler.post {
            (supportFragmentManager.findFragmentById(R.id.main_content) as? UsbCameraFragment)?.startManually()
            configureMainButton()
            applyFullModeToRender()
        }
        Toast.makeText(this, "Searching for USB camera...", Toast.LENGTH_SHORT).show()
    }

    private fun stopUsbAndExit() {
        val frag = supportFragmentManager.findFragmentById(R.id.main_content)
        if (frag is UsbCameraFragment) {
            frag.stopManually()
            supportFragmentManager.commit { remove(frag) }
        }
        isUsbRunning = false
        configureMainButton()
        applyFullModeToRender()
        applyShowModeToRender()
        Toast.makeText(this, "Stopped USB inference.", Toast.LENGTH_SHORT).show()
    }

    // Callbacks invoked by UsbCameraFragment
    fun onUsbCameraConnected() {}
    fun onUsbCameraFailed(errorMsg: String) {
        if (isUsbRunning) {
            Toast.makeText(this, "Failed to connect USB camera: $errorMsg", Toast.LENGTH_LONG).show()
        }
    }

    private fun updateActiveFragmentWithSegmentor(seg: TfLiteSegmentor) {
        when (val frag = supportFragmentManager.findFragmentById(R.id.main_content)) {
            is UsbCameraFragment -> frag.setSegmentor(seg)
            is VideoFileFragment -> frag.setSegmentor(seg)
        }
    }

    private fun showModelPicker() {
        val assetModels = try {
            assets.list("")?.filter { it.endsWith(".tflite") || it.endsWith(".bin") } ?: emptyList()
        } catch (_: Exception) { emptyList() }
        val localModels = filesDir.listFiles()?.filter { it.extension in listOf("tflite", "bin") }?.map { it.name } ?: emptyList()
        val models = (assetModels + localModels).distinct().sorted().toTypedArray()

        if (models.isEmpty()) {
            Toast.makeText(this, "No models found.", Toast.LENGTH_SHORT).show()
            return
        }

        val current = selectedModelAsset ?: runCatching { resources.getString(R.string.tfLiteModelAsset) }.getOrNull()
        val preselect = models.indexOfFirst { it == current }.coerceAtLeast(0)

        AlertDialog.Builder(this)
            .setTitle("Select model")
            .setSingleChoiceItems(models, preselect) { dialog, which ->
                selectedModelAsset = models[which]
                Toast.makeText(this, "Loading: ${models[which]}", Toast.LENGTH_SHORT).show()
                dialog.dismiss()
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
