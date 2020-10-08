#include "ScriptModule.h"

#include "Tensor.h"
#include "utils.h"

namespace torchjs {

Napi::Object ScriptModule::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "ScriptModule",
                  {
                      InstanceMethod("forward", &ScriptModule::forward),
                      InstanceMethod("toString", &ScriptModule::toString),
                  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("ScriptModule", func);
  return exports;
}

ScriptModule::ScriptModule(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<ScriptModule>(info) {
  Napi::HandleScope scope(info.Env());
  Napi::String value = info[0].As<Napi::String>();
  path_ = value;
  module_ = torch::jit::load(value);
  torch::Device device(torch::kCUDA);
  module_.to(torch::kHalf);
  module_.to(device);
  module_.eval();

  // set number of cpus to use
  at::init_num_threads();
  at::set_num_threads(1);
}

Napi::FunctionReference ScriptModule::constructor;

Napi::Value ScriptModule::forward(const Napi::CallbackInfo &info) {
  auto len = info.Length();

  // Convert Node JS buffer to string
  std::vector<torch::jit::IValue> inputs;
  auto fileBuffer = info[0].As<Napi::Buffer<char>>();
  auto array_len = fileBuffer.ElementLength();
  std::vector<uint8_t> dataVector;
  for (decltype(array_len) i = 0; i < array_len; ++i) {
    dataVector.push_back(fileBuffer[i]);
  }
  std::string fileBufferString(dataVector.begin(), dataVector.end());
  inputs.push_back(fileBufferString);

  torch::IValue pred = module_.forward(inputs);

  torch::Tensor outputs = pred.toTuple()->elements()[0].toTensor();

  torch::Tensor boxes = outputs.slice(1, 0, 4).to(torch::kCPU).to(torch::kFloat32);
  torch::Tensor conf = outputs.select(1, 4).to(torch::kCPU).to(torch::kFloat32);
  torch::Tensor cls = outputs.select(1, 5).to(torch::kCPU).to(torch::kInt32);

  auto names = pred.toTuple()->elements()[1].toList();

  Napi::Env env = info.Env();

  Napi::Array classes_array = Napi::Array::New(env, cls.numel());
  auto boxes_array = Napi::TypedArrayOf<float>::New(env, boxes.numel());
  auto scores_array = Napi:: TypedArrayOf<float>::New(env, conf.numel());

  // Get class from list of names
  for ( int i = 0; i < cls.numel(); i++) {
      Napi::String name = Napi::String::New(env, names.get(cls[i].item<int>()).toStringRef());
      classes_array[i] = name;
  }

  memcpy(boxes_array.Data(), boxes.data_ptr(), sizeof(float) * boxes.numel());
  memcpy(scores_array.Data(), conf.data_ptr(), sizeof(float) * conf.numel());

  auto obj = Napi::Object::New(env);
  obj.Set("classes", classes_array);
  obj.Set("scores", scores_array);
  obj.Set("boxes", boxes_array);
  return obj;
}

Napi::Value ScriptModule::toString(const Napi::CallbackInfo &info) {
  return Napi::String::New(info.Env(), "ScriptModule(\"" + path_ + "\")");
}

} // namespace torchjs
