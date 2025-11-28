# Real-time Brachial Plexus Video Segmentation

This app performs real-time video segmentation of the brachial plexus on live ultrasound input. It showcases streaming camera capture, TensorFlow Lite inference, and OpenCV visualization, with acceleration on Qualcomm Hexagon NPU via the TF Lite QNN delegate. The pretrained model is best suited with Sonosite ultrasound machine.

## Video demo

https://github.com/user-attachments/assets/e5cae33d-6d8e-4ec0-b694-02eec4703e71

I feel sorry that not much is documented. I'll keep updating this.

## Build the APK

1. Obtain a compatible `.tflite` model:
   - Export from the training repo and compile via AI Hub using [../Video-object-segmentation-for-SUIT/deploy/export.py](https://github.com/nistring/Hierarchical-Temporal-BP-Seg/blob/master/deploy/export.py). Or download pretrained weights from (To be).
2. Copy the `.tflite` file to `src/main/assets/<your_model>.tflite`.
3. In `graddle.properties`, set `semanticsegmentation_tfLiteModelAsset=<your_model>.tflite`.
4. Open the parent Android project (the `android` root, not this subfolder) in Android Studio, run Gradle sync, and build the `SemanticSegmentation` target.

## License

This app is released under the BSD-3 License and has basis on [Qualcomm AI hub apps](https://github.com/quic/ai-hub-apps). The QNN SDK dependency is also released under a separate license. Please refer to the LICENSE file downloaded with the SDK for details.

Note: The app performs argmax and overlays colored masks on the live preview.
