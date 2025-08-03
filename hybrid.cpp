#include <mpi.h>
#include <omp.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>

namespace fs = std::filesystem;
constexpr int IMG_H = 224, IMG_W = 224;


std::vector<float> preprocess(const cv::Mat& img) {
    cv::Mat rgb,resized,f;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, {IMG_W,IMG_H});
    resized.convertTo(f, CV_32FC3, 1.0/255);

    std::vector<float> v;
    v.reserve(IMG_H*IMG_W*3);
    for(int y=0;y<IMG_H;++y)
        for(int x=0;x<IMG_W;++x)
            for(int c=0;c<3;++c)
                v.push_back(f.at<cv::Vec3f>(y,x)[c]);
    return v;
}


int main(int argc, char** argv)
{
    
    MPI_Init(&argc,&argv);
    int world_size,rank;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(argc<2 && rank==0){
        std::cerr<<"Usage: "<<argv[0]<<" <image_folder>\n";
        MPI_Finalize();
        return 1;
    }
    std::string folder = (argc>1)? argv[1] : "";

   
    std::vector<std::string> all_paths;
    if(rank==0){
        for(const auto& e: fs::directory_iterator(folder))
            if(e.is_regular_file()) all_paths.push_back(e.path().string());
    }

 
    int total = static_cast<int>(all_paths.size());
    MPI_Bcast(&total,1,MPI_INT,0,MPI_COMM_WORLD);

    std::vector<char> blob;
    if(rank==0){
        for(auto& p: all_paths){
            int len = p.size();
            blob.insert(blob.end(),reinterpret_cast<char*>(&len),
                        reinterpret_cast<char*>(&len)+sizeof(int));
            blob.insert(blob.end(),p.begin(),p.end());
        }
    }
    int blob_sz = blob.size();
    MPI_Bcast(&blob_sz,1,MPI_INT,0,MPI_COMM_WORLD);
    if(rank!=0) blob.resize(blob_sz);
    MPI_Bcast(blob.data(),blob_sz,MPI_CHAR,0,MPI_COMM_WORLD);


    if(rank!=0){
        size_t pos=0; all_paths.clear();
        while(pos<blob.size()){
            int len;  std::memcpy(&len,&blob[pos],sizeof(int));  pos+=sizeof(int);
            all_paths.emplace_back(&blob[pos],len);             pos+=len;
        }
    }

  
    double mpi_start = MPI_Wtime();

  
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hybrid");
    Ort::SessionOptions so;  so.SetIntraOpNumThreads(1);        
    Ort::Session session(env,"ratianalModel.onnx",so);
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name  = session.GetInputNameAllocated (0,alloc);
    auto out_name = session.GetOutputNameAllocated(0,alloc);
    const char* in  = in_name.get();
    const char* out = out_name.get();
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);

   
    std::vector<const char*> in_names{in};
    std::vector<const char*> out_names{out};

    double local_start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for(int i=rank; i<total; i+=world_size)
    {
        cv::Mat img = cv::imread(all_paths[i]);
        if(img.empty()){
            #pragma omp critical
            std::cerr<<"Rank "<<rank<<" could not read "<<all_paths[i]<<"\n";
            continue;
        }

        auto tensor_vals = preprocess(img);
        std::array<int64_t,4> shape = {1,IMG_H,IMG_W,3};

        Ort::Value input  = Ort::Value::CreateTensor<float>(mem,
                           tensor_vals.data(), tensor_vals.size(),
                           shape.data(), shape.size());

        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                   in_names.data(), &input, 1,
                                   out_names.data(),1);

        float conf = outputs.front().GetTensorMutableData<float>()[0];
        std::string verdict = conf>0.7f? "Disease Detected" : "Healthy";

        #pragma omp critical
        std::cout<<"Rank "<<rank<<" | "<<all_paths[i]
                 <<" -> "<<verdict<<" ("<<conf<<")\n";
    }

    double local_elapsed = omp_get_wtime() - local_start;

 
    double max_elapsed;
    MPI_Reduce(&local_elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   
    double mpi_total = MPI_Wtime() - mpi_start;
    if(rank==0){
        std::cout<<"\n===== Timing =====\n"
                 <<"MPI wall‑clock (rank0): "<<mpi_total<<" s\n";
                // <<"Slowest rank elapsed : "<<max_elapsed<<" s\n";
    }

    MPI_Finalize();
    return 0;
}

