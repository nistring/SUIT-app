// ---------------------------------------------------------------------
// Kotlin TFLite segmentor tailored to depth352.tflite and utils.py logic
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation

import android.content.Context
import android.graphics.Bitmap
import com.quicinc.tflite.AIHubDefaults
import com.quicinc.tflite.TFLiteHelpers
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.osgi.OpenCVNativeLoader
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import android.util.Pair
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

class TfLiteSegmentor(
    context: Context,
    modelPath: String,
    delegatePriorityOrder: Array<Array<TFLiteHelpers.DelegateType>>
) : AutoCloseable {

    private val interpreter: Interpreter
    private val delegateStore: Map<TFLiteHelpers.DelegateType, Delegate>
    private val inputShape: IntArray

    private val inputMatAbgr: Mat
    private val inputMatGray: Mat

    var lastPreTime: Long = 0; private set
    var lastInferTime: Long = 0; private set
    var lastPostTime: Long = 0; private set

    init {
        OpenCVNativeLoader().init()
        val modelAndHash: Pair<MappedByteBuffer, String> = TFLiteHelpers.loadModelFile(context.assets, modelPath)
        val iResult: Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> =
            TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
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

        val inCount = interpreter.inputTensorCount
        if (inCount < 1) {
            throw IllegalArgumentException("Model has no inputs. inputTensorCount=$inCount")
        }
        val inTensor: Tensor = interpreter.getInputTensor(0)
        inputShape = inTensor.shape()
        val inType = inTensor.dataType()
        require(inputShape.size == 5 && inputShape[0] == 1) {
            "Expected input [1,T,C,H,W], got ${inputShape.contentToString()}"
        }

        val outTensor: Tensor = interpreter.getOutputTensor(0)
        val outShape = outTensor.shape()
        val outType = outTensor.dataType()

        val inT = inputShape[1]
        val inC = inputShape[2]
        val inH = inputShape[3]
        val inW = inputShape[4]

        // Mat(rows=height, cols=width)
        inputMatAbgr = Mat(inH, inW, CvType.CV_8UC4)
        inputMatGray = Mat(inH, inW, CvType.CV_8UC1)
    }

    fun getInputWidth(): Int = inputShape[4]
    fun getInputHeight(): Int = inputShape[3]

    override fun close() {
        interpreter.close()
        delegateStore.values.forEach { it.close() }
    }


    fun predict(input: Bitmap, sensorOrientation: Int, hiddenState: Array<FloatArray>? = null): kotlin.Pair<Bitmap, Array<FloatArray>?> {
        val t0 = System.nanoTime()
        Utils.bitmapToMat(input, inputMatAbgr)

        Imgproc.cvtColor(inputMatAbgr, inputMatGray, Imgproc.COLOR_BGRA2GRAY)

        val matGrayRot = when (sensorOrientation) {
            90 -> Mat().also { Core.rotate(inputMatGray, it, Core.ROTATE_90_COUNTERCLOCKWISE) }
            180 -> Mat().also { Core.rotate(inputMatGray, it, Core.ROTATE_180) }
            270 -> Mat().also { Core.rotate(inputMatGray, it, Core.ROTATE_90_CLOCKWISE) }
            else -> inputMatGray
        }

        // 5D input only: [1,T,C,H,W]
        val inT = inputShape[1]
        val inC = inputShape[2]
        val inH = inputShape[3]
        val inW = inputShape[4]

        // Build input objects for interpreter
        val inputCount = interpreter.inputTensorCount
        val inputsArray = arrayOfNulls<Any>(inputCount)

        // First input: image.
        val imageTotalElems = inT * inC * inH * inW

        // Convert grayscale to float32 first, then resize the float matrix to the model input size.
        val matGrayFloat = Mat()
        matGrayRot.convertTo(matGrayFloat, CvType.CV_32FC1) // values remain 0..255 as floats

        val resizedFloatMat = if (matGrayFloat.rows() == inH && matGrayFloat.cols() == inW) {
            matGrayFloat
        } else {
            Mat().also {
            Imgproc.resize(matGrayFloat, it, Size(inW.toDouble(), inH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            }
        }

        // Extract float pixels directly
        val frameFloats = FloatArray(inH * inW)
        resizedFloatMat.get(0, 0, frameFloats)

        // Tile into [1,T,C,H,W] float32 buffer
        val repeats = inT * inC
        val bbImg = ByteBuffer.allocateDirect(imageTotalElems * 4).order(ByteOrder.nativeOrder())
        val fb = bbImg.asFloatBuffer()
        repeat(repeats) { fb.put(frameFloats) }
        bbImg.rewind()
        inputsArray[0] = bbImg

        val t1 = System.nanoTime()
        lastPreTime = t1 - t0

        // Hidden-state inputs (indices 1..)
        if (inputCount > 1) {
            for (i in 1 until inputCount) {
                val inTens = interpreter.getInputTensor(i)
                val shape = inTens.shape()
                var size = 1
                for (s in shape) size *= s
                val bb = ByteBuffer.allocateDirect(size * 4).order(ByteOrder.nativeOrder())
                val fbHidden = bb.asFloatBuffer()
                if (hiddenState != null && i - 1 < hiddenState.size) {
                    val src = hiddenState[i - 1]
                    val toCopy = minOf(src.size, size)
                    fbHidden.put(src, 0, toCopy)
                    if (toCopy < size) fbHidden.put(FloatArray(size - toCopy))
                } else {
                    fbHidden.put(FloatArray(size))
                }
                bb.rewind()
                inputsArray[i] = bb
            }
        }

        val t2 = System.nanoTime()

        // Prepare outputs map. First output is float32 3D, rest are hidden states.
        val outputCount = interpreter.outputTensorCount
        val outputs = mutableMapOf<Int, Any>()
        val outTensor0 = interpreter.getOutputTensor(0)
        val outShape0 = outTensor0.shape()
        var outTotalSize = 1
        for (s in outShape0) outTotalSize *= s

        val bb0 = ByteBuffer.allocateDirect(outTotalSize * 4).order(ByteOrder.nativeOrder())
        bb0.rewind()
        outputs[0] = bb0

        if (outputCount > 1) {
            for (i in 1 until outputCount) {
                val outT = interpreter.getOutputTensor(i)
                val outShape = outT.shape()
                var size = 1
                for (s in outShape) size *= s
                val bb = when (outT.dataType()) {
                    DataType.FLOAT32, DataType.INT32 -> ByteBuffer.allocateDirect(size * 4).order(ByteOrder.nativeOrder())
                    else -> ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
                }
                bb.rewind()
                outputs[i] = bb
            }
        }

        try {
            interpreter.runForMultipleInputsOutputs(inputsArray, outputs)
        } catch (e: Exception) {
            Log.e("TfLiteSegmentor", "Inference failed!", e)
            throw e
        }
        val t3 = System.nanoTime()
        lastInferTime = t3 - t2

        // Interpret segmentation output (first output)
        val dims = outShape0
        val outH = dims[0]
        val outW = dims[1]

        val outBb0 = outputs[0] as ByteBuffer
        outBb0.rewind()
        val outFloats = FloatArray(outTotalSize)
        outBb0.asFloatBuffer().get(outFloats)

        val outFloatMat = Mat(outH, outW, CvType.CV_32FC3)
        outFloatMat.put(0, 0, outFloats)

        val outRotFloat = Mat()
        when (sensorOrientation) {
            90 -> Core.rotate(outFloatMat, outRotFloat, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(outFloatMat, outRotFloat, Core.ROTATE_180)
            270 -> Core.rotate(outFloatMat, outRotFloat, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> outFloatMat.copyTo(outRotFloat)
        }

        val viewW = input.width
        val viewH = input.height
        val outResizedFloat = Mat()
        Imgproc.resize(outRotFloat, outResizedFloat, Size(viewW.toDouble(), viewH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)

        val outU8 = Mat()
        outResizedFloat.convertTo(outU8, CvType.CV_8UC3)

        val outRgba = Mat()
        Imgproc.cvtColor(outU8, outRgba, Imgproc.COLOR_BGR2RGBA)
        val outBmp = Bitmap.createBitmap(viewW, viewH, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(outRgba, outBmp)

        // Collect hidden outputs (indices 1..)
        var returnedHidden: Array<FloatArray>? = null
        if (outputCount > 1) {
            val list = ArrayList<FloatArray>(outputCount - 1)
            for (i in 1 until outputCount) {
                val outBb = outputs[i] as ByteBuffer
                outBb.rewind()
                val outT = interpreter.getOutputTensor(i)
                val size = outT.numElements()
                val fa = FloatArray(size)
                outBb.asFloatBuffer().get(fa)
                list.add(fa)
            }
            returnedHidden = list.toTypedArray()
        }

        lastPostTime = System.nanoTime() - t3
        return kotlin.Pair(outBmp, returnedHidden)
    }
}