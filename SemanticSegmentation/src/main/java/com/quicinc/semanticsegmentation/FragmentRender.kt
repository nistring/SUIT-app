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
import android.util.Log
import java.util.concurrent.locks.ReentrantLock

class FragmentRender(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private val mLock = ReentrantLock()
    private var mBitmap: Bitmap? = null
    private var mOriginalBitmap: Bitmap? = null
    private val mTargetRect = Rect()
    private val mDisplayRect = Rect()
    @Volatile private var fullMode: Boolean = false
    @Volatile private var showInference: Boolean = true
    private var refW: Int = 0
    private var refH: Int = 0
    private var fps: Float = 0f
    // replace timing fields
    private var consumerTime: Long = 0
    private var producerTime: Long = 0
    private val mTextColor = Paint().apply {
        color = Color.WHITE
        typeface = Typeface.DEFAULT_BOLD
        style = Paint.Style.FILL
        textSize = 40f
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
    private var bboxDefaultInitialized = false  // NEW: Track if default bbox was set

    // Public API to toggle/query full mode
    fun setFullMode(enabled: Boolean) {
        fullMode = enabled
        postInvalidate()
    }
    fun isFullMode(): Boolean = fullMode

    // New: Public API to toggle/query inference visibility
    fun setShowInference(enabled: Boolean) {
        showInference = enabled
        postInvalidate()
    }
    fun isShowInference(): Boolean = showInference

    // Reset bbox initialization state (e.g., when switching video sources)
    fun resetBBoxInitialization() {
        mLock.lock()
        try {
            bboxDefaultInitialized = false  // NEW
            bbox.setEmpty()
        } finally {
            mLock.unlock()
        }
    }

    // Update the original source frame size so mapping rect stays correct when showing only crop
    fun updateReferenceSize(w: Int, h: Int) {
        mLock.lock()
        try {
            if (w > 0 && h > 0) {
                refW = w
                refH = h
            }
        } finally {
            mLock.unlock()
        }
    }

    // Update signature: accept both result and original
    fun render(image: Bitmap, original: Bitmap?, fps: Float, consumerTime: Long, producerTime: Long) {
        this.mBitmap = image
        this.mOriginalBitmap = original
        this.fps = fps
        this.consumerTime = consumerTime
        this.producerTime = producerTime
        postInvalidate()
    }

    /**
     * Initialize the bounding box manually based on the largest contour in the current frame.
     * Call this when user presses the init button.
     */
    fun initializeBBoxFromOriginal(original: Bitmap): Boolean {
        mLock.lock()
        try {
            if (mTargetRect.width() <= 0) {
                return false
            }

            // Detect largest contour in the original image
            val detectedRect = AutoCropUtil.detectLargestContourBBox(original, null) ?: return false

            // Map the detected bbox from original image coordinates to view coordinates
            val scaleX = mTargetRect.width().toFloat() / original.width
            val scaleY = mTargetRect.height().toFloat() / original.height

            val viewLeft = mTargetRect.left + (detectedRect.left * scaleX)
            val viewTop = mTargetRect.top + (detectedRect.top * scaleY)
            val viewRight = mTargetRect.left + (detectedRect.right * scaleX)
            val viewBottom = mTargetRect.top + (detectedRect.bottom * scaleY)

            bbox.set(viewLeft, viewTop, viewRight, viewBottom)
            postInvalidate()
            return true
        } finally {
            mLock.unlock()
        }
    }

    /**
     * Return a Pair of (cropped bitmap, rect in original bitmap coordinates)
     * according to the current bounding box. If the bbox is not initialized or invalid, returns null.
     * This method is thread-safe and can be called off the UI thread.
     */
    fun getCroppedBitmapWithRect(original: Bitmap): Pair<Bitmap, Rect>? {
        mLock.lock()
        try {
            if (bbox.width() <= 0f || bbox.height() <= 0f || mTargetRect.width() <= 0) return null

            val scaleX = original.width.toFloat() / mTargetRect.width()
            val scaleY = original.height.toFloat() / mTargetRect.height()
            val left = ((bbox.left - mTargetRect.left) * scaleX).toInt().coerceIn(0, original.width - 1)
            val top = ((bbox.top - mTargetRect.top) * scaleY).toInt().coerceIn(0, original.height - 1)
            val right = ((bbox.right - mTargetRect.left) * scaleX).toInt().coerceIn(0, original.width)
            val bottom = ((bbox.bottom - mTargetRect.top) * scaleY).toInt().coerceIn(0, original.height)
            val w = (right - left).coerceAtLeast(1)
            val h = (bottom - top).coerceAtLeast(1)

            if (left == 0 && top == 0 && w == original.width && h == original.height) 
                return Pair(original, Rect(0, 0, original.width, original.height))

            val cropped = Bitmap.createBitmap(original, left, top, w, h)
            return Pair(cropped, Rect(left, top, right, bottom))
        } finally {
            mLock.unlock()
        }
    }

    private val prefs by lazy { context.getSharedPreferences("bbox_prefs", Context.MODE_PRIVATE) }

    // NEW: Save bbox to SharedPreferences
    private fun saveBBox() {
        prefs.edit().apply {
            putFloat("bbox_left", bbox.left)
            putFloat("bbox_top", bbox.top)
            putFloat("bbox_right", bbox.right)
            putFloat("bbox_bottom", bbox.bottom)
            apply()
        }
    }

    // NEW: Restore bbox from SharedPreferences
    private fun restoreBBox() {
        val left = prefs.getFloat("bbox_left", -1f)
        if (left >= 0f) {
            bbox.set(
                left,
                prefs.getFloat("bbox_top", 0f),
                prefs.getFloat("bbox_right", 0f),
                prefs.getFloat("bbox_bottom", 0f)
            )
            bboxDefaultInitialized = true
        }
    }

    @SuppressLint("DefaultLocale")
    override fun onDraw(canvas: Canvas) {
        mLock.lock()
        canvas.drawColor(Color.BLACK)

        val normalToDraw: Bitmap? = if (showInference) mBitmap else (mOriginalBitmap ?: mBitmap)
        val bmp = if (fullMode) (mBitmap ?: normalToDraw) else normalToDraw
        
        if (bmp != null) {
            val canvasRatio = width.toFloat() / height.toFloat()
            val bitmapRatio = bmp.width.toFloat() / bmp.height.toFloat()

            if (!fullMode) {
                val insetWidth: Int
                val insetHeight: Int
                if (canvasRatio > bitmapRatio) {
                    insetHeight = height
                    insetWidth = (height.toFloat() * bitmapRatio).toInt()
                } else {
                    insetWidth = width
                    insetHeight = (width.toFloat() / bitmapRatio).toInt()
                }
                val offsetWidth = (width - insetWidth) / 2
                val offsetHeight = (height - insetHeight) / 2
                mDisplayRect.set(offsetWidth, offsetHeight, offsetWidth + insetWidth, offsetHeight + insetHeight)
                mTargetRect.set(mDisplayRect)
            } else {
                val bmpRatio = bmp.width.toFloat() / bmp.height.toFloat()
                if (canvasRatio > bmpRatio) {
                    val insetWidth = (height.toFloat() * bmpRatio).toInt()
                    val offsetWidth = (width - insetWidth) / 2
                    mDisplayRect.set(offsetWidth, 0, offsetWidth + insetWidth, height)
                } else {
                    val insetHeight = (width.toFloat() / bmpRatio).toInt()
                    val offsetHeight = (height - insetHeight) / 2
                    mDisplayRect.set(0, offsetHeight, width, offsetHeight + insetHeight)
                }

                val refRatio = if (refW > 0 && refH > 0) refW.toFloat() / refH.toFloat() else bitmapRatio
                if (canvasRatio > refRatio) {
                    val insetWidth = (height.toFloat() * refRatio).toInt()
                    val offsetWidth = (width - insetWidth) / 2
                    mTargetRect.set(offsetWidth, 0, offsetWidth + insetWidth, height)
                } else {
                    val insetHeight = (width.toFloat() / refRatio).toInt()
                    val offsetHeight = (height - insetHeight) / 2
                    mTargetRect.set(0, offsetHeight, width, offsetHeight + insetHeight)
                }
            }

            if (!fullMode) {
                canvas.drawBitmap(bmp, null, mDisplayRect, null)
            } else if (showInference || mOriginalBitmap == null) {
                canvas.drawBitmap(bmp, null, mDisplayRect, null)
            } else {
                val orig = mOriginalBitmap
                if (orig != null && bbox.width() > 0 && bbox.height() > 0 && mTargetRect.width() > 0) {
                    val scaleX = (refW.takeIf { it > 0 } ?: orig.width).toFloat() / mTargetRect.width()
                    val scaleY = (refH.takeIf { it > 0 } ?: orig.height).toFloat() / mTargetRect.height()
                    val left = ((bbox.left - mTargetRect.left) * scaleX).toInt().coerceIn(0, orig.width - 1)
                    val top = ((bbox.top - mTargetRect.top) * scaleY).toInt().coerceIn(0, orig.height - 1)
                    val right = ((bbox.right - mTargetRect.left) * scaleX).toInt().coerceIn(left + 1, orig.width)
                    val bottom = ((bbox.bottom - mTargetRect.top) * scaleY).toInt().coerceIn(top + 1, orig.height)
                    canvas.drawBitmap(orig, Rect(left, top, right, bottom), mDisplayRect, null)
                } else {
                    canvas.drawBitmap(orig ?: bmp, null, mDisplayRect, null)
                }
            }

            // NEW: Set default bbox once if not initialized
            if (bbox.width() == 0f && bbox.height() == 0f && mTargetRect.width() > 0 && !bboxDefaultInitialized) {
                restoreBBox()
                if (bbox.width() == 0f) {  // Still empty after restore
                    val padW = mTargetRect.width() * 0.2f
                    val padH = mTargetRect.height() * 0.2f
                    bbox.set(mTargetRect.left + padW, mTargetRect.top + padH, mTargetRect.right - padW, mTargetRect.bottom - padH)
                }
                bboxDefaultInitialized = true
            }

            if (!fullMode) {
                if (bbox.width() > 0f && bbox.height() > 0f) {
                    canvas.drawRect(bbox, boxPaint)
                    canvas.drawCircle(bbox.left, bbox.top, handleRadius, boxPaint)
                    canvas.drawCircle(bbox.right, bbox.bottom, handleRadius, boxPaint)
                }
                val topY = mTargetRect.bottom - 150f
                canvas.drawText("Producer: ${String.format("%.0f", producerTime / 1_000_000f)}ms", 10f, topY, mTextColor)
                canvas.drawText("Consumer: ${String.format("%.0f", consumerTime / 1_000_000f)}ms", 10f, topY + 50f, mTextColor)
            }
        }
        mLock.unlock()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (fullMode) return false // disable bbox interactions in full mode
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
                saveBBox()  // NEW: Persist on every move
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
        if (width <= 0) { super.onMeasure(widthMeasureSpec, heightMeasureSpec); return }

        var desiredHeight: Int? = null
        mLock.lock()
        try {
            val bmp = if (fullMode) mBitmap else if (showInference) mBitmap else (mOriginalBitmap ?: mBitmap)
            if (bmp != null && bmp.width > 0) {
                desiredHeight = (width / (bmp.width.toFloat() / bmp.height.toFloat())).toInt()
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
