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
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
      device_type = torch::kCUDA;
  } else {
      device_type = torch::kCPU;
  }
  torch::Device device(torch::kCUDA);
  module_.to(device);
  module_.eval();
  // at::init_num_threads();
  // torch::set_num_threads(16);
}

Napi::FunctionReference ScriptModule::constructor;

Napi::Value ScriptModule::forward(const Napi::CallbackInfo &info) {
  auto len = info.Length();
  std::vector<torch::jit::IValue> inputs;
  // TODO: Support other type of IValue, e.g., list
  torch::Device device(torch::kCUDA);
  for (size_t i = 0; i < len; ++i) {
    Tensor *tensor =
        Napi::ObjectWrap<Tensor>::Unwrap(info[i].As<Napi::Object>());
    inputs.push_back(tensor->tensor());
  }
  auto outputs = module_.forward(inputs);
  // TODO: Support other type of IValue
  assert(outputs.isTensor());
  torch::Tensor tensor = outputs.toTensor().to(torch::kCPU);
  Napi::Env env = info.Env();

  auto typed_array = Napi::TypedArrayOf<float>::New(env, tensor.numel());
  memcpy(typed_array.Data(), tensor.data_ptr(), sizeof(float) * tensor.numel());
  return typed_array;
  // return scope.Escape(Tensor::FromTensor(info.Env(), outputs.toTensor()));
}

Napi::Value ScriptModule::toString(const Napi::CallbackInfo &info) {
  return Napi::String::New(info.Env(), "ScriptModule(\"" + path_ + "\")");
}

} // namespace torchjs
