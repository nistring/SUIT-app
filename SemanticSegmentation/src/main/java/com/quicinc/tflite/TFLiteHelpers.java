// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;
import android.util.Pair;

import com.qualcomm.qti.QnnDelegate;

import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

public class TFLiteHelpers {
    private static final String TAG = "QualcommTFLiteHelpers";

    public enum DelegateType {
        GPUv2,
        QNN_NPU,
    }

    /**
     * Create a TFLite interpreter from the given model.
     *
     * @param tfLiteModel           The model to load.
     *
     * @param delegatePriorityOrder Delegates, in order they should be registered to the interpreter.
     *
     *                              The "inner array" defines which delegates should be registered when creating the interpreter.
     *                              The order of delegates is the priority in which they are assigned layers.
     *                              For example, if an array contains delegates { QNN_NPU, GPUv2 }, then QNN_NPU will be assigned any
     *                              compatible op first. GPUv2 will then be assigned any ops that QNN_NPU is unable to run.
     *                              And finally, XNNPack will be assigned ops that both QNN_NPU and GPUv2 are unable to run.
     *
     *                              The "outer array" defines the order of delegate list the interpreter should be created with.
     *                              This method will first attempt to create an interpreter using all delegates in the first array.
     *                              If that interpreter fails to instantiate, this method will try to create an interpreter
     *                              using all delegates in the second array. This continues until an interpreter could be successfully
     *                              created & returned, or until all arrays are tried unsuccessfully--which results in an exception.
     *
     * @param numCPUThreads         Number of CPU threads to use for layers on CPU.
     * @param nativeLibraryDir      Android.Context.nativeLibraryDir (native library directory location)
     * @param cacheDir              Android app cache directory.
     * @param modelIdentifier       Unique identifier string for the model being loaded.
     *
     * @return A pair of the created interpreter and associated delegates.
     *         These delegates must be kept in memory until they are no longer needed. Before
     *         deleting, the client must call close() on the returned delegates and interpreter.
     */
    @SuppressWarnings("unchecked")
    public static Pair<Interpreter, Map<DelegateType, Delegate>> CreateInterpreterAndDelegatesFromOptions(
            MappedByteBuffer tfLiteModel,
            DelegateType[][] delegatePriorityOrder,
            int numCPUThreads,
            String nativeLibraryDir,
            String cacheDir,
            String modelIdentifier) {

        // Delegate Storage
        Map<DelegateType, Delegate> delegates = new HashMap<>();

        // All delegates we've tried to instantiate, whether that was successful or not.
        Set<DelegateType> attemptedDelegates = new HashSet<>();

        // Attempt to register delegate pairings in the defined priority order.
        for (DelegateType[] delegatesToRegister : delegatePriorityOrder) {
            // Create delegates for this attempt if we haven't done so already.
            Arrays.stream(delegatesToRegister)
                    .filter(delegateType -> !attemptedDelegates.contains(delegateType))
                    .forEach(delegateType -> {
                        Delegate delegate = CreateDelegate(delegateType, nativeLibraryDir, cacheDir, modelIdentifier);
                        if (delegate != null) {
                            delegates.put(delegateType, delegate);
                        }
                        attemptedDelegates.add(delegateType);
                    });

            // If one or more delegates in this attempt could not be instantiated,
            // skip this attempt.
            if (Arrays.stream(delegatesToRegister).anyMatch(x -> !delegates.containsKey(x))) {
                continue;
            }

            // Create interpreter.
            Interpreter interpreter = CreateInterpreterFromDelegates(
                Arrays.stream(delegatesToRegister).map(
                        delegateType -> new Pair<>(delegateType, delegates.get(delegateType))
                ).toArray(Pair[]::new),
                numCPUThreads,
                tfLiteModel
            );

            // If the interpreter failed to be created, move on to the next attempt.
            if (interpreter == null) {
                continue;
            }

            // Drop & close delegates that were not used by this attempt.
            delegates.keySet().stream()
                    .filter(delegateType -> Arrays.stream(delegatesToRegister).noneMatch(d -> d == delegateType))
                    .collect(Collectors.toSet()) // needed so we don't modify the same object we're looping over
                    .forEach(unusedDelegateType -> {
                        Objects.requireNonNull(delegates.remove(unusedDelegateType)).close();
                    });

            // Return interpreter & associated delegates.
            return new Pair<>(interpreter, delegates);
        }

        throw new RuntimeException("Unable to create an interpreter of any kind for the provided model. See log for details.");
    }



    /**
     * Create an interpreter from the given delegates.
     *
     * @param delegates     Delegate instances to be registered in the interpreter.
     *                      Delegates will be registered in the order of this array.
     * @param numCPUThreads Number of CPU threads to use for layers on CPU.
     * @param tfLiteModel   TFLiteModel to pass to the interpreter.
     * @return An Interpreter if creation is successful, and null otherwise.
     */
    @SuppressWarnings("unchecked")
    public static Interpreter CreateInterpreterFromDelegates(
            final Pair<DelegateType, Delegate>[] delegates,
            int numCPUThreads,
            MappedByteBuffer tfLiteModel) {
        Interpreter.Options tfLiteOptions = new Interpreter.Options();
        tfLiteOptions.setRuntime(Interpreter.Options.TfLiteRuntime.FROM_APPLICATION_ONLY);
        tfLiteOptions.setAllowBufferHandleOutput(true);
        tfLiteOptions.setUseNNAPI(false);
        tfLiteOptions.setNumThreads(numCPUThreads);
        tfLiteOptions.setUseXNNPACK(true); // Fall back to XNNPack (fast CPU implementation) if a layer cannot run on NPU.

        // Register delegates in this interpreter. The first delegate
        // registered will have "first pick" of which operators to run, and so on.
        StringBuilder reg = new StringBuilder();
        for (Pair<DelegateType, Delegate> p : delegates) {
            tfLiteOptions.addDelegate(p.second);
            reg.append(p.first.name()).append(",");
        }
        if (reg.length() > 0) reg.setLength(reg.length() - 1);
        Log.i(TAG, "Registering delegates in order: [" + reg.toString() + "] with CPU threads=" + numCPUThreads);

        try {
            Interpreter i = new Interpreter(tfLiteModel, tfLiteOptions);
            Log.i(TAG, "Interpreter created successfully with delegates: [" + reg.toString() + "] (XNNPACK fallback enabled)");
            i.allocateTensors();
            return i;
        } catch (Exception e) {
            List<String> enabledDelegates = Arrays.stream(delegates).map(x -> x.first.name()).collect(Collectors.toCollection(ArrayList<String>::new));
            enabledDelegates.add("XNNPack");
            Log.e(TAG, "Failed to Load Interpreter with delegates {" + String.join(", ", enabledDelegates) + "} | " + e.getMessage());
            return null;
        }
    }

    /**
     * Load a TFLite model file from assets.
     *
     * @param assets        Android app asset manager.
     * @param filename      File name of the model to load (.tflite).
     * @return The loaded file in MappedByteBuffer format, and a unique file identifier hash string.
     * @throws IOException If the file does not exist or cannot be read.
     */
    public static Pair<MappedByteBuffer, String> loadModelFile(AssetManager assets, String modelFilename)
            throws IOException, NoSuchAlgorithmException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        MappedByteBuffer buffer;
        String hash;

        try (FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();

            // Map the file to a buffer
            buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

            // Compute the hash
            MessageDigest hashDigest = MessageDigest.getInstance("MD5");
            inputStream.skip(startOffset);
            try (DigestInputStream dis = new DigestInputStream(inputStream, hashDigest)) {
                byte[] data = new byte[8192];
                int numRead = 0;
                while (numRead < declaredLength) {
                    numRead += dis.read(data, 0, Math.min(8192, (int)declaredLength - numRead));
                }; // Computing MD5 hash
            }

            // Convert hash to string
            StringBuilder hex = new StringBuilder();
            for (byte b : hashDigest.digest()) {
                hex.append(String.format("%02x", b));
            }
            hash = hex.toString();
        }

        return new Pair<>(buffer, hash);
    }

    /**
     * @param delegateType     The type of delegate to create.
     * @param nativeLibraryDir Native library directory for Android app.
     * @param cacheDir         Android app cache directory.
     * @param modelIdentifier  Unique identifier string for the model being loaded.
     * @return The created delegate if successful, and null otherwise.
     */
    static Delegate CreateDelegate(DelegateType delegateType, String nativeLibraryDir, String cacheDir, String modelIdentifier) {
        if (delegateType == DelegateType.GPUv2) {
            return CreateGPUv2Delegate(cacheDir, modelIdentifier);
        }
        if (delegateType == DelegateType.QNN_NPU) {
            return CreateQNN_NPUDelegate(nativeLibraryDir, cacheDir, modelIdentifier);
        }

        throw new RuntimeException("Delegate creation not implemented for type: " + delegateType.name());
    }

    /**
     * Create and configure the QNN NPU delegate.
     * QNN NPU will be configured for maximum performance (at the cost of device battery life / heat / precision).
     *
     * @param nativeLibraryDir Native library directory for Android app.
     * @param cacheDir         Android app cache directory.
     * @param modelIdentifier  Unique identifier string for the model being loaded.
     * @return The created delegate if successful, and null otherwise.
     */
    static Delegate CreateQNN_NPUDelegate(String nativeLibraryDir, String cacheDir, String modelIdentifier) {
        QnnDelegate.Options qnnOptions = new QnnDelegate.Options();
        qnnOptions.setSkelLibraryDir(nativeLibraryDir);
        qnnOptions.setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_WARN);
        qnnOptions.setCacheDir(cacheDir);
        qnnOptions.setModelToken(modelIdentifier);

        // Check capabilities and configure backend
        if (QnnDelegate.checkCapability(QnnDelegate.Capability.DSP_RUNTIME)) {
            qnnOptions.setBackendType(QnnDelegate.Options.BackendType.DSP_BACKEND);
            qnnOptions.setDspOptions(QnnDelegate.Options.DspPerformanceMode.DSP_PERFORMANCE_BURST, QnnDelegate.Options.DspPdSession.DSP_PD_SESSION_ADAPTIVE);
        } else {
            boolean hasHTP_FP16 = QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16);
            boolean hasHTP_QUANT = QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED);

            if (!hasHTP_FP16 && !hasHTP_QUANT) {
                Log.e(TAG, "QNN with NPU backend is not supported on this device.");
                return null;
            }

            qnnOptions.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND);
            qnnOptions.setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON);
            qnnOptions.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST);

            if (hasHTP_FP16) {
                qnnOptions.setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16);
            }
        }

        Log.i(TAG, "Creating QNN delegate (backend=" + qnnOptions.getBackendType() + ")");
        return new QnnDelegate(qnnOptions);
    }

    /**
     * Create and configure the GPUv2 delegate.
     * GPUv2 will be configured for maximum performance (at the cost of device battery life / heat / precision),
     * and to allow execution in FP16 precision.
     *
     * @param cacheDir         Android app cache directory.
     * @param modelIdentifier  Unique identifier string for the model being loaded.
     * @return The created delegate if successful, and null otherwise.
     */
    static Delegate CreateGPUv2Delegate(String cacheDir, String modelIdentifier) {
        GpuDelegateFactory.Options gpuOptions = new GpuDelegateFactory.Options();
        gpuOptions.setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
        gpuOptions.setPrecisionLossAllowed(false);
        gpuOptions.setSerializationParams(cacheDir, modelIdentifier);

        Log.i(TAG, "Creating GPUv2 delegate (sustained speed, fp16 allowed)");
        return new GpuDelegate(gpuOptions);
    }
}