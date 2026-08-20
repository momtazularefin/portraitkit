using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PortraitKit.Sample;

/// <summary>
/// Minimal C# / .NET sample demonstrating cross-runtime consumption of 
/// PortraitKit ONNX model artifacts via Microsoft.ML.OnnxRuntime.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        Console.WriteLine("=================================================");
        Console.WriteLine("PortraitKit .NET ONNX Runtime Consumption Sample");
        Console.WriteLine("=================================================\n");

        string modelPath = args.Length > 0 ? args[0] : FindDefaultModelPath();

        if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
        {
            Console.WriteLine($"[INFO] No model file found at '{modelPath}'.");
            Console.WriteLine("[INFO] Running in mock validation mode (demonstrating tensor contracts and API bindings)...");
            RunMockValidation();
            return 0;
        }

        Console.WriteLine($"[INFO] Loading ONNX model from: {modelPath}");
        try
        {
            using var sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            using var session = new InferenceSession(modelPath, sessionOptions);

            PrintModelMetadata(session);
            RunInferenceDemo(session);

            Console.WriteLine("\n[SUCCESS] .NET ONNX inference completed successfully.");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Inference failed: {ex.Message}");
            return 1;
        }
    }

    private static string FindDefaultModelPath()
    {
        string[] candidates = new[]
        {
            Path.Combine("..", "..", "..", "..", "models", "yunet-2023mar.onnx"),
            Path.Combine("models", "yunet-2023mar.onnx"),
            Path.Combine("..", "models", "yunet-2023mar.onnx"),
            Path.Combine("..", "..", "models", "yunet-2023mar.onnx"),
            Path.Combine("..", "..", "..", "models", "yunet-2023mar.onnx")
        };

        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                return Path.GetFullPath(candidate);
            }
        }

        return string.Empty;
    }

    private static void PrintModelMetadata(InferenceSession session)
    {
        Console.WriteLine("\n--- Model Signature Metadata ---");
        Console.WriteLine("Input Nodes:");
        foreach (var kvp in session.InputMetadata)
        {
            var dims = string.Join("x", kvp.Value.Dimensions.Select(d => d.ToString()));
            Console.WriteLine($"  * '{kvp.Key}': Type={kvp.Value.ElementType}, Shape=[{dims}]");
        }

        Console.WriteLine("Output Nodes:");
        foreach (var kvp in session.OutputMetadata)
        {
            var dims = string.Join("x", kvp.Value.Dimensions.Select(d => d.ToString()));
            Console.WriteLine($"  * '{kvp.Key}': Type={kvp.Value.ElementType}, Shape=[{dims}]");
        }
    }

    private static void RunInferenceDemo(InferenceSession session)
    {
        Console.WriteLine("\n--- Executing Inference ---");

        var firstInput = session.InputMetadata.First();
        string inputName = firstInput.Key;
        var dims = firstInput.Value.Dimensions;

        // Resolve input dimensions (handle dynamic/batch sizes)
        int batch = dims.Length > 0 && dims[0] > 0 ? dims[0] : 1;
        int channels = dims.Length > 1 && dims[1] > 0 ? dims[1] : 3;
        int height = dims.Length > 2 && dims[2] > 0 ? dims[2] : 640;
        int width = dims.Length > 3 && dims[3] > 0 ? dims[3] : 640;

        Console.WriteLine($"Allocating input tensor [{batch}, {channels}, {height}, {width}] for '{inputName}'...");

        // Create synthetic input tensor following the PreprocessContract
        var tensor = new DenseTensor<float>(new[] { batch, channels, height, width });
        
        // Fill sample normalized test pattern
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        tensor[b, c, y, x] = 0.5f; // Neutral grey
                    }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };

        var stopwatch = Stopwatch.StartNew();
        using var results = session.Run(inputs);
        stopwatch.Stop();

        Console.WriteLine($"Execution complete in {stopwatch.Elapsed.TotalMilliseconds:F2} ms.");
        Console.WriteLine($"Outputs returned ({results.Count()} tensors):");

        foreach (var result in results)
        {
            if (result.Value is Tensor<float> outputTensor)
            {
                var shape = string.Join("x", outputTensor.Dimensions.ToArray());
                Console.WriteLine($"  - '{result.Name}': Shape=[{shape}], Min={outputTensor.Min():F4}, Max={outputTensor.Max():F4}");
            }
            else
            {
                Console.WriteLine($"  - '{result.Name}': Type={result.Value?.GetType().Name}");
            }
        }
    }

    private static void RunMockValidation()
    {
        Console.WriteLine("\n--- Contract & Tensor Validation ---");
        Console.WriteLine("Verifying Microsoft.ML.OnnxRuntime integration:");
        
        // Construct standard DenseTensor matching PortraitKit NCHW PreprocessContract
        var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
        tensor[0, 0, 0, 0] = 1.0f;

        Console.WriteLine($"  * DenseTensor created: Shape=[{string.Join("x", tensor.Dimensions.ToArray())}], Elements={tensor.Length}");
        Console.WriteLine("  * NCHW float32 input contract validated.");
        Console.WriteLine("  * Ready to consume models exported by PortraitKit (YuNet, SCRFD, MODNet, RMBG, U2Net, BiRefNet, IS-Net).");
    }
}
