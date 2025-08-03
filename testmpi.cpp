#include <mpi.h>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;

const int IMG_HEIGHT = 224;
const int IMG_WIDTH = 224;

// Preprocess image to float array (NHWC format)
std::vector<float> preprocess(const cv::Mat& img) {
   cv::Mat rgb_img, resized, float_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    cv::resize(rgb_img, resized, cv::Size(224, 224));
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255);  // ✅ Normalize here

    std::vector<float> input_tensor_values;
    for (int y = 0; y < 224; ++y)
        for (int x = 0; x < 224; ++x)
            for (int c = 0; c < 3; ++c)
                input_tensor_values.push_back(float_img.at<cv::Vec3f>(y, x)[c]);

    return input_tensor_values;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 2) {
        if (world_rank == 0) std::cerr << "Usage: " << argv[0] << " <image_folder>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string image_folder = argv[1];
    std::vector<std::string> all_images;

    // Rank 0 reads all image paths and distributes later
    if (world_rank == 0) {
        for (const auto& entry : fs::directory_iterator(image_folder)) {
            if (entry.is_regular_file()) {
                all_images.push_back(entry.path().string());
            }
        }
    }

    // Broadcast number of images to all ranks
    int total_images = all_images.size();
    MPI_Bcast(&total_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

    
    int images_per_rank = (total_images + world_size - 1) / world_size;
    std::vector<char> serialized_paths;

    if (world_rank == 0) {
        
        for (const auto& path : all_images) {
            int len = path.size();
            serialized_paths.insert(serialized_paths.end(),
                reinterpret_cast<char*>(&len),
                reinterpret_cast<char*>(&len) + sizeof(int));
            serialized_paths.insert(serialized_paths.end(), path.begin(), path.end());
        }
    }

    // Broadcast total size of serialized paths
    int serialized_size = serialized_paths.size();
    MPI_Bcast(&serialized_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate buffer on other ranks and receive
    if (world_rank != 0) serialized_paths.resize(serialized_size);
    MPI_Bcast(serialized_paths.data(), serialized_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Deserialize paths on all ranks
    if (world_rank != 0) {
        all_images.clear();
        size_t pos = 0;
        while (pos < serialized_paths.size()) {
            int len;
            memcpy(&len, &serialized_paths[pos], sizeof(int));
            pos += sizeof(int);
            std::string path(&serialized_paths[pos], len);
            pos += len;
            all_images.push_back(path);
        }
    }
double start_time = MPI_Wtime();
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MPIInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "ratianalModel.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_name_ptr = input_name.get();
    const char* output_name_ptr = output_name.get();

    std::vector<const char*> input_names{input_name_ptr};
    std::vector<const char*> output_names{output_name_ptr};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Each rank processes its chunk of images
    for (int i = world_rank; i < total_images; i += world_size) {
        try {
            cv::Mat img = cv::imread(all_images[i]);
            if (img.empty()) {
                std::cerr << "Rank " << world_rank << ": Failed to load image: " << all_images[i] << std::endl;
                continue;
            }

            std::vector<float> input_tensor_values = preprocess(img);
            std::array<int64_t, 4> input_shape = {1, IMG_HEIGHT, IMG_WIDTH, 3};  

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_values.size(),
                input_shape.data(),
                input_shape.size());

            std::vector<Ort::Value> ort_outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names.data(), &input_tensor, 1,
                output_names.data(), 1);

            float* scores = ort_outputs.front().GetTensorMutableData<float>();
            float confidence = scores[0];
            std::string result = confidence > 0.7f ? "Disease Detected" : "Healthy";

            std::cout << "Rank " << world_rank
                      << " processed: " << all_images[i]
                      << " -> " << result
                      << " (Confidence: " << confidence << ")" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "Rank " << world_rank
                      << " ONNX Runtime error on image " << all_images[i]
                      << ": " << e.what() << std::endl;
        }
    }
double end_time = MPI_Wtime(); // End timing

    if (world_rank == 0) {
        std::cout << "\n=== MPI Execution Time: " << (end_time - start_time) << " seconds ===" << std::endl;
    }
    MPI_Finalize();
    return 0;
}

