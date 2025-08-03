#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
namespace fs = std::filesystem;

const int IMG_HEIGHT = 224;
const int IMG_WIDTH = 224;


std::vector<float> preprocess(const cv::Mat& img) {
    cv::Mat rgb_img, resized, float_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB); 
    cv::resize(rgb_img, resized, cv::Size(IMG_WIDTH, IMG_HEIGHT)); 
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255); 

    std::vector<float> input_tensor_values;
    for (int y = 0; y < IMG_HEIGHT; ++y)
        for (int x = 0; x < IMG_WIDTH; ++x)
            for (int c = 0; c < 3; ++c)
                input_tensor_values.push_back(float_img.at<cv::Vec3f>(y, x)[c]);

    return input_tensor_values;
}

int main() {
auto start_time = std::chrono::high_resolution_clock::now();
    try {
      
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RetinaModel");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        Ort::Session session(env, "ratianalModel.onnx", session_options);

        
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
        auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();
        const char* output_name = output_name_ptr.get();

        std::vector<const char*> input_names{input_name};
        std::vector<const char*> output_names{output_name};

        std::vector<std::string> image_files;
        for (const auto& entry : fs::directory_iterator("./images"))
            image_files.push_back(entry.path().string());

       
        for (int i = 0; i < static_cast<int>(image_files.size()); ++i) {
            try {
                cv::Mat img = cv::imread(image_files[i]);
                if (img.empty()) {
                    std::cerr << "Failed to load image: " << image_files[i] << std::endl;
                    continue;
                }

                std::vector<float> input_tensor_values = preprocess(img);
                std::array<int64_t, 4> input_shape = {1, IMG_HEIGHT, IMG_WIDTH, 3};

                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_tensor_values.data(), input_tensor_values.size(),
                    input_shape.data(), input_shape.size());

                auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                                  &input_tensor, 1, output_names.data(), 1);

                float* scores = output_tensors.front().GetTensorMutableData<float>();
                float confidence = scores[0];
                std::string result = confidence > 0.7f ? "Disease Detected" : "Healthy";

                std::cout << "Image: " << image_files[i]
                          << " -> " << result
                          << " (Confidence Score: " << confidence << ")" << std::endl;

            } catch (const Ort::Exception& e) {
                std::cerr << "Error processing " << image_files[i] << ": " << e.what() << std::endl;
            }
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime failed: " << e.what() << std::endl;
        return 1;
    }

auto end_time = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end_time - start_time;
std::cout << "\n=== Execution Time: " << duration.count() << " seconds ===" << std::endl;

    return 0;
}

