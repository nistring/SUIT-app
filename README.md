# Real-time Brachial Plexus Video Segmentation (TF Lite + QNN)

This app performs real-time video segmentation of the brachial plexus on live ultrasound input. It showcases streaming camera capture, TensorFlow Lite inference, and OpenCV visualization, with acceleration on Qualcomm Hexagon NPU via the TF Lite QNN delegate. The pretrained model is best suited with Sonosite ultrasound machine.

## Build the APK

1. Obtain a compatible `.tflite` model:
   - Export from the training repo and compile via AI Hub using [../Video-object-segmentation-for-SUIT/deploy/export.py](../Video-object-segmentation-for-SUIT/deploy/export.py). Or download pretrained weights from (To be).
2. Copy the `.tflite` file to `src/main/assets/<your_model>.tflite`.
3. In [../gradle.properties](../gradle.properties), set `semanticsegmentation_tfLiteModelAsset=<your_model>.tflite`.
4. Open the parent Android project (the `android` root, not this subfolder) in Android Studio, run Gradle sync, and build the `SemanticSegmentation` target.

## License

This app is released under the [BSD-3 License](LICENSE).

The QNN SDK dependency is also released under a separate license. Please refer to the LICENSE file downloaded with the SDK for details.

Note: The app performs argmax and overlays colored masks on the live preview.
