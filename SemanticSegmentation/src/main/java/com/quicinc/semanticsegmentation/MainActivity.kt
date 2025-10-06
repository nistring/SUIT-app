// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation

import android.app.Activity
import android.os.Bundle
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.WindowManager
import android.widget.ProgressBar
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

class MainActivity : AppCompatActivity() {
    private lateinit var progressBar: ProgressBar
    private var segmentor: TfLiteSegmentor? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    @Volatile private var isProcessing = false
    private val pickVideo = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null && segmentor != null) {
            supportFragmentManager.commit {
                replace(R.id.main_content, VideoFileFragment.create(segmentor!!, uri))
            }
        } else {
            Toast.makeText(this, "No video selected.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main_activity)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        progressBar = findViewById(R.id.indeterminateBar)
        createTFLiteClassifiersAsync()
    }

    private fun setLoadingUI(loading: Boolean) {
        runOnUiThread { progressBar.visibility = if (loading) View.VISIBLE else View.INVISIBLE }
    }

    private fun createTFLiteClassifiersAsync() {
        check(segmentor == null) { "Segmentor was already created" }
        setLoadingUI(true)
        backgroundExecutor.execute {
            val tfLiteModelAsset = resources.getString(R.string.tfLiteModelAsset)
            try {
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
                if (isProbablyEmulator()) {
                    // In emulator: open file picker for MP4 and analyze frames
                    pickVideo.launch("video/*")
                } else if (hasExternalCamera()) {
                    supportFragmentManager.commit {
                        replace(R.id.main_content, UsbCameraFragment.create(segmentor!!))
                    }
                } else {
                    val uvc = detectUvcVideoDevices()
                    if (uvc.isNotEmpty()) {
                        Log.i("SemanticSeg", "UVC video devices detected: ${uvc.map { it.deviceName }}")
                        Toast.makeText(this, "UVC device detected but not available via Camera2. A UVC library is required for live capture.", Toast.LENGTH_LONG).show()
                    } else {
                        listCamera2DevicesForDebug()
                        Toast.makeText(this, "No external HDMI capture camera detected. Ensure your dongle is UVC and powered (OTG/Hub).", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
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
        // Nothing to do here; live capture fragment will handle resume
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
