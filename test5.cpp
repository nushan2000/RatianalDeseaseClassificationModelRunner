#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    const std::string model_path = "ratianalModel.onnx";
    const std::string image_path = "images/30.png";  // Path to test image

    // Load image using OpenCV
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }


cv::cvtColor(image, image, cv::COLOR_BGR2RGB);  // 🔥 Add this!

    // Resize and normalize
    cv::resize(image, image, cv::Size(224, 224));
    image.convertTo(image, CV_32F, 1.0 / 255);

    // Change to CHW format and add batch dimension
    // NHWC format
std::vector<float> input_tensor_values;
for (int i = 0; i < 224; ++i) {
    for (int j = 0; j < 224; ++j) {
        cv::Vec3f pixel = image.at<cv::Vec3f>(i, j);
        input_tensor_values.push_back(pixel[0]);  // R
        input_tensor_values.push_back(pixel[1]);  // G
        input_tensor_values.push_back(pixel[2]);  // B
    }
}




    std::vector<int64_t> input_dims = {1, 224, 224, 3};  


    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "inference");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
const char* input_name = input_name_ptr.get();

Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
const char* output_name = output_name_ptr.get();


    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size());

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      &input_name, &input_tensor, 1,
                                      &output_name, 1);

    float* raw_output = output_tensors[0].GetTensorMutableData<float>();
    float prediction = raw_output[0];

    std::cout << "Raw ONNX prediction: " << prediction << std::endl;

    int predicted_class = (prediction > 0.7f) ? 1 : 0;
    std::cout << "Predicted class: " << (predicted_class ? "Disease (True)" : "No Disease (False)") << std::endl;

    return 0;
}
