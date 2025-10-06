// ---------------------------------------------------------------------
// FragmentRender draws the final prediction image and overlays debugging text.
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.view.MotionEvent
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import java.util.concurrent.locks.ReentrantLock

class FragmentRender(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private val mLock = ReentrantLock()
    private var mBitmap: Bitmap? = null
    private val mTargetRect = Rect()
    private var fps: Float = 0f
    private var inferTime: Long = 0
    private var preprocessTime: Long = 0
    private var postprocessTime: Long = 0
    private val mTextColor = Paint().apply {
        color = Color.WHITE
        typeface = Typeface.DEFAULT_BOLD
        style = Paint.Style.FILL
        textSize = 50f
    }

    // Bounding box state (view coordinates)
    private val bbox = RectF()
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val handleRadius = 30f
    private enum class TouchMode { NONE, DRAG, RESIZE_TL, RESIZE_BR }
    private var touchMode = TouchMode.NONE
    private var lastTouchX = 0f
    private var lastTouchY = 0f

    fun render(image: Bitmap, fps: Float, inferTime: Long, preprocessTime: Long, postprocessTime: Long) {
        this.mBitmap = image
        this.fps = fps
        this.inferTime = inferTime
        this.preprocessTime = preprocessTime
        this.postprocessTime = postprocessTime
        postInvalidate()
    }

    /**
     * Return a cropped bitmap according to the current bounding box.
     * If the bbox is not initialized or invalid, returns null.
     * This method is thread-safe and can be called off the UI thread.
     */
    /**
     * Return a Pair of (cropped bitmap, rect in original bitmap coordinates)
     * according to the current bounding box. If the bbox is not initialized or invalid, returns null.
     * This method is thread-safe and can be called off the UI thread.
     */
    fun getCroppedBitmapWithRect(original: Bitmap): Pair<Bitmap, Rect>? {
        mLock.lock()
        try {
            if (bbox.width() <= 0f || bbox.height() <= 0f) return null
            // Ensure target rect is valid
            val target = mTargetRect
            if (target.width() <= 0 || target.height() <= 0) return null

            // Map bbox (view coords) into original bitmap coords
            val scaleX = original.width.toFloat() / target.width().toFloat()
            val scaleY = original.height.toFloat() / target.height().toFloat()
            val leftF = (bbox.left - target.left) * scaleX
            val topF = (bbox.top - target.top) * scaleY
            val rightF = (bbox.right - target.left) * scaleX
            val bottomF = (bbox.bottom - target.top) * scaleY

            val left = leftF.toInt().coerceIn(0, original.width - 1)
            val top = topF.toInt().coerceIn(0, original.height - 1)
            val right = rightF.toInt().coerceIn(0, original.width)
            val bottom = bottomF.toInt().coerceIn(0, original.height)
            val w = (right - left).coerceAtLeast(1)
            val h = (bottom - top).coerceAtLeast(1)

            // If the bbox covers the whole image, just return the original and full rect
            if (left == 0 && top == 0 && w == original.width && h == original.height) return Pair(original, Rect(0, 0, original.width, original.height))

            return try {
                val cropped = Bitmap.createBitmap(original, left, top, w, h)
                val rect = Rect(left, top, right, bottom)
                Pair(cropped, rect)
            } catch (e: Exception) {
                null
            }
        } finally {
            mLock.unlock()
        }
    }

    @SuppressLint("DefaultLocale")
    override fun onDraw(canvas: Canvas) {
        mLock.lock()
        val bmp = mBitmap
        if (bmp != null) {
            val insetWidth: Int
            val insetHeight: Int
            val canvasRatio = width.toFloat() / height.toFloat()
            val bitmapRatio = bmp.width.toFloat() / bmp.height.toFloat()
            if (canvasRatio > bitmapRatio) {
                insetHeight = height
                insetWidth = (height.toFloat() * bitmapRatio).toInt()
            } else {
                insetWidth = width
                insetHeight = (width.toFloat() / bitmapRatio).toInt()
            }
            val offsetWidth = (width - insetWidth) / 2
            val offsetHeight = (height - insetHeight) / 2
            mTargetRect.left = offsetWidth
            mTargetRect.top = offsetHeight
            mTargetRect.right = offsetWidth + insetWidth
            mTargetRect.bottom = offsetHeight + insetHeight
            canvas.drawBitmap(bmp, null, mTargetRect, null)
            // Initialize bbox if empty: center 60% of the rendered image
            if (bbox.width() == 0f && bbox.height() == 0f && mTargetRect.width() > 0 && mTargetRect.height() > 0) {
                val padW = mTargetRect.width() * 0.2f
                val padH = mTargetRect.height() * 0.2f
                bbox.left = mTargetRect.left + padW
                bbox.top = mTargetRect.top + padH
                bbox.right = mTargetRect.right - padW
                bbox.bottom = mTargetRect.bottom - padH
            }

            // Draw bounding box + corner handles
            if (bbox.width() > 0f && bbox.height() > 0f) {
                canvas.drawRect(bbox, boxPaint)
                // handles: top-left and bottom-right
                canvas.drawCircle(bbox.left, bbox.top, handleRadius, boxPaint)
                canvas.drawCircle(bbox.right, bbox.bottom, handleRadius, boxPaint)
            }
            // Draw debug text in top-left of the view (horizontal)
            canvas.drawText("FPS: ${String.format("%.0f", fps)}", 15f, 50f, mTextColor)
            canvas.drawText("Preprocess: ${String.format("%.0f", preprocessTime / 1_000_000f)}ms", 15f, 55f + 60f * 2, mTextColor)
            canvas.drawText("Infer: ${String.format("%.0f", inferTime / 1_000_000f)}ms", 15f, 55f + 60f * 3, mTextColor)
            canvas.drawText("Postprocess: ${String.format("%.0f", postprocessTime / 1_000_000f)}ms", 15f, 55f + 60f * 4, mTextColor)
        }
        mLock.unlock()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = event.y
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                mLock.lock()
                try {
                    // determine interaction area
                    if (isNear(x, y, bbox.left, bbox.top, handleRadius)) {
                        touchMode = TouchMode.RESIZE_TL
                    } else if (isNear(x, y, bbox.right, bbox.bottom, handleRadius)) {
                        touchMode = TouchMode.RESIZE_BR
                    } else if (bbox.contains(x, y)) {
                        touchMode = TouchMode.DRAG
                    } else {
                        touchMode = TouchMode.NONE
                    }
                } finally {
                    mLock.unlock()
                }
                lastTouchX = x
                lastTouchY = y
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = x - lastTouchX
                val dy = y - lastTouchY
                mLock.lock()
                try {
                    when (touchMode) {
                        TouchMode.DRAG -> {
                            bbox.offset(dx, dy)
                            // clamp inside target rect
                            if (bbox.left < mTargetRect.left) bbox.offset(mTargetRect.left - bbox.left, 0f)
                            if (bbox.top < mTargetRect.top) bbox.offset(0f, mTargetRect.top - bbox.top)
                            if (bbox.right > mTargetRect.right) bbox.offset(mTargetRect.right - bbox.right, 0f)
                            if (bbox.bottom > mTargetRect.bottom) bbox.offset(0f, mTargetRect.bottom - bbox.bottom)
                        }
                        TouchMode.RESIZE_TL -> {
                            bbox.left += dx
                            bbox.top += dy
                            // clamp
                            if (bbox.left < mTargetRect.left) bbox.left = mTargetRect.left.toFloat()
                            if (bbox.top < mTargetRect.top) bbox.top = mTargetRect.top.toFloat()
                            if (bbox.left > bbox.right - 50f) bbox.left = bbox.right - 50f
                            if (bbox.top > bbox.bottom - 50f) bbox.top = bbox.bottom - 50f
                        }
                        TouchMode.RESIZE_BR -> {
                            bbox.right += dx
                            bbox.bottom += dy
                            if (bbox.right > mTargetRect.right) bbox.right = mTargetRect.right.toFloat()
                            if (bbox.bottom > mTargetRect.bottom) bbox.bottom = mTargetRect.bottom.toFloat()
                            if (bbox.right < bbox.left + 50f) bbox.right = bbox.left + 50f
                            if (bbox.bottom < bbox.top + 50f) bbox.bottom = bbox.top + 50f
                        }
                        else -> {}
                    }
                } finally {
                    mLock.unlock()
                }
                lastTouchX = x
                lastTouchY = y
                postInvalidate()
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                touchMode = TouchMode.NONE
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    private fun isNear(px: Float, py: Float, cx: Float, cy: Float, r: Float): Boolean {
        val dx = px - cx
        val dy = py - cy
        return dx * dx + dy * dy <= r * r
    }

    // Make the view horizontally wide (match parent width) while preserving the
    // displayed image's aspect ratio. When a bitmap is available, compute the
    // desired height so that the bitmap fills the width and keeps ratio.
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = View.MeasureSpec.getSize(widthMeasureSpec)
        if (width <= 0) {
            super.onMeasure(widthMeasureSpec, heightMeasureSpec)
            return
        }

        // Try to calculate desired height from current bitmap aspect ratio
        var desiredHeight: Int? = null
        mLock.lock()
        try {
            val bmp = mBitmap
            if (bmp != null && bmp.width > 0) {
                val bitmapRatio = bmp.width.toFloat() / bmp.height.toFloat()
                desiredHeight = (width / bitmapRatio).toInt()
            }
        } finally {
            mLock.unlock()
        }

        if (desiredHeight != null) {
            val heightMode = View.MeasureSpec.getMode(heightMeasureSpec)
            val heightSize = View.MeasureSpec.getSize(heightMeasureSpec)
            val finalHeight = when (heightMode) {
                View.MeasureSpec.EXACTLY -> heightSize
                View.MeasureSpec.AT_MOST -> desiredHeight.coerceAtMost(heightSize)
                else -> desiredHeight
            }
            setMeasuredDimension(width, finalHeight)
        } else {
            super.onMeasure(widthMeasureSpec, heightMeasureSpec)
        }
    }
}
