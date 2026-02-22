// ---------------------------------------------------------------------
// Kotlin TFLite segmentor tailored to depth352.tflite and utils.py logic
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import com.quicinc.tflite.AIHubDefaults
import com.quicinc.tflite.TFLiteHelpers
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.android.OpenCVLoader
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import android.util.Pair
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.ArrayList
import kotlin.math.min

class TfLiteSegmentor(
    context: Context,
    modelPath: String,
    delegatePriorityOrder: Array<Array<TFLiteHelpers.DelegateType>>
) : AutoCloseable {

    private val interpreter: Interpreter
    private val delegateStore: Map<TFLiteHelpers.DelegateType, Delegate>
    private val inputShape: IntArray

    private val hiddenInBuffers = ArrayList<ByteBuffer>()
    private var out0Buffer: ByteBuffer? = null
    private val hiddenOutBuffers = ArrayList<ByteBuffer>()

    private val inH: Int
    private val inW: Int
    private val inTensor0Bytes: Int
    private val frameElems: Int
    private val frameRepeat: Int
    private val inputCount: Int
    private val outputCount: Int
    private val hiddenInSizes: IntArray
    private val hiddenOutSizes: IntArray
    private val out0H: Int
    private val out0W: Int
    private val out0Elems: Int
    private val outputsMap = HashMap<Int, Any>()

    init {
        // Ensure OpenCV native libs are loaded
        val cvReady = try { OpenCVLoader.initDebug() } catch (_: Throwable) { false }
        if (!cvReady) System.loadLibrary("opencv_java4")

        val localFile = java.io.File(context.filesDir, modelPath)
        val modelAndHash = if (localFile.exists()) TFLiteHelpers.loadModelFromFile(localFile)
            else TFLiteHelpers.loadModelFile(context.assets, modelPath)
        val iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
            modelAndHash.first,
            delegatePriorityOrder,
            AIHubDefaults.numCPUThreads,
            context.applicationInfo.nativeLibraryDir,
            context.cacheDir.absolutePath,
            modelAndHash.second
        )
        interpreter = iResult.first
        @Suppress("UNCHECKED_CAST")
        delegateStore = iResult.second as Map<TFLiteHelpers.DelegateType, Delegate>

        inputCount = interpreter.inputTensorCount
        require(inputCount >= 1) { "Model has no inputs." }
        inputShape = interpreter.getInputTensor(0).shape()
        require(inputShape.size >= 3 && inputShape[0] == 1) { "Expected input with batch=1 and >=3 dims, got ${inputShape.contentToString()}" }
        
        inH = inputShape[inputShape.size - 2]
        inW = inputShape[inputShape.size - 1]
        inTensor0Bytes = interpreter.getInputTensor(0).numBytes()
        frameElems = inH * inW
        frameRepeat = if (frameElems == 0) 0 else interpreter.getInputTensor(0).numElements() / frameElems

        hiddenInSizes = if (inputCount > 1) {
            IntArray(inputCount - 1) { idx -> interpreter.getInputTensor(idx + 1).numElements() }
        } else IntArray(0)

        outputCount = interpreter.outputTensorCount
        val outTensor0 = interpreter.getOutputTensor(0)
        val outShape0 = outTensor0.shape()
        out0H = outShape0[0]
        out0W = outShape0[1]
        out0Elems = outTensor0.numElements()

        hiddenOutSizes = if (outputCount > 1) {
            IntArray(outputCount - 1) { idx -> interpreter.getOutputTensor(idx + 1).numElements() }
        } else IntArray(0)
    }

    fun getInputWidth(): Int = inW
    fun getInputHeight(): Int = inH

    override fun close() {
        interpreter.close()
        delegateStore.values.forEach { it.close() }
    }

    // Preprocessed input payload produced on producer thread.
    data class Preprocessed(
        val inputBuffer: ByteBuffer,
        val viewW: Int,
        val viewH: Int,
        val sensorOrientation: Int,
        val originalForDisplay: Bitmap? = null,   // optional: original image for overlay
        val cropRectInOriginal: Rect? = null      // optional: rect for overlay
    )

    // Inference result produced on inference thread.
    data class InferResult(
        val outFloats: FloatArray,
        val outH: Int,
        val outW: Int,
        val newHidden: Array<FloatArray>?

    )

    // Build model input [1,T,C,H,W] float32 from a Bitmap. Thread-safe (no shared state).
    fun preprocess(input: Bitmap, sensorOrientation: Int, originalForDisplay: Bitmap? = null, cropRectInOriginal: Rect? = null): Preprocessed {
        // Convert to GRAY, resize to model input, rotate on small image, then convert to float
        val matGray = Mat().apply {
            Utils.bitmapToMat(input, this)
            Imgproc.cvtColor(this, this, Imgproc.COLOR_BGRA2GRAY)
        }

        val resizedU8 = Mat()
        if (matGray.rows() == inH && matGray.cols() == inW) matGray.copyTo(resizedU8)
        else Imgproc.resize(matGray, resizedU8, Size(inW.toDouble(), inH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)

        val rotU8 = Mat()
        when (sensorOrientation) {
            90 -> Core.rotate(resizedU8, rotU8, Core.ROTATE_90_COUNTERCLOCKWISE)
            180 -> Core.rotate(resizedU8, rotU8, Core.ROTATE_180)
            270 -> Core.rotate(resizedU8, rotU8, Core.ROTATE_90_CLOCKWISE)
            else -> resizedU8.copyTo(rotU8)
        }

        val frameArr = FloatArray(frameElems)
        Mat().apply { rotU8.convertTo(this, CvType.CV_32FC1); get(0, 0, frameArr) }

        val bbImg = ByteBuffer.allocateDirect(inTensor0Bytes).order(ByteOrder.nativeOrder())
        val fb = bbImg.asFloatBuffer()
        val bufferFloats = inTensor0Bytes / 4
        var filledFloats = 0
        while (filledFloats < bufferFloats) {
            val toWrite = minOf(frameArr.size, bufferFloats - filledFloats)
            fb.put(frameArr, 0, toWrite)
            filledFloats += toWrite
        }
        bbImg.rewind()

        return Preprocessed(bbImg, input.width, input.height, sensorOrientation, originalForDisplay, cropRectInOriginal)
    }

    // Run interpreter using a preprocessed input. Not thread-safe with other infer() calls.
    fun infer(pre: Preprocessed, hiddenState: Array<FloatArray>? = null): InferResult {
        val inputsArray = arrayOfNulls<Any>(inputCount)
        inputsArray[0] = pre.inputBuffer

        for (h in 0 until hiddenInSizes.size) {
            val bb = allocateOrGetBuffer(hiddenInBuffers, h, hiddenInSizes[h] * 4)
            bb.clear()
            val fbHidden = bb.asFloatBuffer()
            if (hiddenState != null && h < hiddenState.size) {
                val src = hiddenState[h]
                fbHidden.put(src, 0, min(src.size, hiddenInSizes[h]))
                if (src.size < hiddenInSizes[h]) fbHidden.put(FloatArray(hiddenInSizes[h] - src.size))
            } else {
                fbHidden.put(FloatArray(hiddenInSizes[h]))
            }
            bb.rewind()
            inputsArray[h + 1] = bb
        }

        val bb0 = out0Buffer?.takeIf { it.capacity() == out0Elems * 4 }
            ?: ByteBuffer.allocateDirect(out0Elems * 4).order(ByteOrder.nativeOrder()).also { out0Buffer = it }
        bb0.clear()
        bb0.rewind()
        outputsMap[0] = bb0

        for (h in 0 until hiddenOutSizes.size) {
            val bb = allocateOrGetBuffer(hiddenOutBuffers, h, hiddenOutSizes[h] * 4)
            bb.clear()
            bb.rewind()
            outputsMap[h + 1] = bb
        }

        try {
            interpreter.runForMultipleInputsOutputs(inputsArray, outputsMap)
        } catch (e: Exception) {
            Log.e("TfLiteSegmentor", "Inference failed!", e)
            throw e
        }

        val outFloats = FloatArray(out0Elems)
        (outputsMap[0] as ByteBuffer).apply { 
            rewind()
            asFloatBuffer().get(outFloats, 0, outFloats.size)
        }

        val returnedHidden = if (hiddenOutSizes.isNotEmpty()) {
            Array(hiddenOutSizes.size) { idx ->
                val arr = FloatArray(hiddenOutSizes[idx])
                (outputsMap[idx + 1] as ByteBuffer).apply { 
                    rewind()
                    asFloatBuffer().get(arr, 0, arr.size)
                }
                arr
            }
        } else null

        return InferResult(outFloats, out0H, out0W, returnedHidden)
    }

    // Convert float output to displayable RGBA Bitmap with desired view size and orientation. Thread-safe (local Mats).
    fun postprocessToBitmap(res: InferResult, viewW: Int, viewH: Int, sensorOrientation: Int): Bitmap {
        val outFloatMat = Mat(res.outH, res.outW, CvType.CV_32FC3).apply { put(0, 0, res.outFloats) }

        val outRotFloat = Mat()
        when (sensorOrientation) {
            90 -> Core.rotate(outFloatMat, outRotFloat, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(outFloatMat, outRotFloat, Core.ROTATE_180)
            270 -> Core.rotate(outFloatMat, outRotFloat, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> outFloatMat.copyTo(outRotFloat)
        }

        val outResized = Mat()
        Imgproc.resize(outRotFloat, outResized, Size(viewW.toDouble(), viewH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)

        val outU8 = Mat()
        outResized.convertTo(outU8, CvType.CV_8UC3)

        val outRgba = Mat()
        Imgproc.cvtColor(outU8, outRgba, Imgproc.COLOR_BGR2RGBA)

        val outBmp = Bitmap.createBitmap(viewW, viewH, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(outRgba, outBmp)

        return outBmp
    }

    private fun allocateOrGetBuffer(bufferList: ArrayList<ByteBuffer>, index: Int, sizeBytes: Int): ByteBuffer {
        return if (index < bufferList.size && bufferList[index].capacity() == sizeBytes) {
            bufferList[index]
        } else {
            ByteBuffer.allocateDirect(sizeBytes).order(ByteOrder.nativeOrder()).also {
                if (index < bufferList.size) bufferList[index] = it else bufferList.add(it)
            }
        }
    }
}