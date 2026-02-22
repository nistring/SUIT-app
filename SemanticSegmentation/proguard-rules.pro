# TensorFlow Lite
-keep class org.tensorflow.lite.** { *; }
-keep class com.qualcomm.qti.** { *; }

# JavaCV / Bytedeco - keep classes used at runtime
-keep class org.bytedeco.javacv.FFmpegFrameGrabber { *; }
-keep class org.bytedeco.javacv.AndroidFrameConverter { *; }
-keep class org.bytedeco.javacpp.** { *; }
-keep class org.bytedeco.ffmpeg.** { *; }

# Suppress warnings for desktop-only classes not available on Android
-dontwarn com.jogamp.**
-dontwarn java.awt.**
-dontwarn javax.swing.**
-dontwarn javax.imageio.**
-dontwarn javax.management.**
-dontwarn javafx.**
-dontwarn java.beans.**
-dontwarn java.lang.management.**
-dontwarn org.apache.maven.**
-dontwarn org.bytedeco.opencv.**
-dontwarn org.bytedeco.javacv.**
-dontwarn org.bytedeco.flandmark.**
-dontwarn org.osgi.**
-dontwarn org.slf4j.**

# OpenCV
-keep class org.opencv.** { *; }
