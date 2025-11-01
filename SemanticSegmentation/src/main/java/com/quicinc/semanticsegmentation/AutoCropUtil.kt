// ---------------------------------------------------------------------
// AutoCropUtil: Automatic crop detection using largest contour
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

object AutoCropUtil {
    private const val KERNEL_SIZE = 5
    private const val THRESH_BINARY = 30

    /**
     * Detect the bounding box of the largest contour in the image.
     * This is used to automatically initialize the ROI bounding box.
     *
     * @param bitmap The input bitmap
     * @param roi Optional region of interest to restrict the search area
     * @return Rect representing the bounding box (x, y, width, height) of the largest contour,
     *         or null if no contours found
     */
    fun detectLargestContourBBox(bitmap: Bitmap, roi: Rect? = null): Rect? {
        try {
            // Convert bitmap to OpenCV Mat
            val mat = Mat()
            Utils.bitmapToMat(bitmap, mat)

            // If ROI is specified, crop to that region first
            val workingMat = if (roi != null) {
                val roiMat = Mat(mat, org.opencv.core.Rect(roi.left, roi.top, roi.width(), roi.height()))
                roiMat.clone()
            } else {
                mat
            }

            // Convert to grayscale
            val gray = Mat()
            Imgproc.cvtColor(workingMat, gray, Imgproc.COLOR_BGR2GRAY)

            // Apply threshold
            val thresholded = Mat()
            Imgproc.threshold(gray, thresholded, THRESH_BINARY.toDouble(), 255.0, Imgproc.THRESH_TOZERO)

            // Create kernel for morphological operations
            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(KERNEL_SIZE.toDouble(), KERNEL_SIZE.toDouble())
            )

            // Apply morphological opening (erosion followed by dilation)
            val opened = Mat()
            Imgproc.morphologyEx(thresholded, opened, Imgproc.MORPH_OPEN, kernel)

            // Find contours
            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(
                opened,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
            )

            // Find the largest contour by area
            if (contours.isEmpty()) {
                return null
            }

            var maxContour: MatOfPoint? = null
            var maxArea = 0.0
            for (contour in contours) {
                val area = Imgproc.contourArea(contour)
                if (area > maxArea) {
                    maxArea = area
                    maxContour = contour
                }
            }

            if (maxContour == null) {
                return null
            }

            // Get bounding rectangle of the largest contour
            val boundingRect = Imgproc.boundingRect(maxContour)

            // Adjust coordinates if ROI was used
            val finalX = boundingRect.x + (roi?.left ?: 0)
            val finalY = boundingRect.y + (roi?.top ?: 0)

            // Clean up
            mat.release()
            workingMat.release()
            gray.release()
            thresholded.release()
            kernel.release()
            opened.release()
            hierarchy.release()
            for (contour in contours) {
                contour.release()
            }

            return Rect(finalX, finalY, finalX + boundingRect.width, finalY + boundingRect.height)
        } catch (e: Exception) {
            Log.e("AutoCropUtil", "Exception in detectLargestContourBBox: ${e.message}", e)
            return null
        }
    }
}
