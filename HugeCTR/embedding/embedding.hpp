/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <embedding/embedding_table.hpp>
#include <vector>

namespace embedding {

enum class Stage {
  DPForward,
  DPBackwardIndexCalculation,
  DPLocalReduce,
  DPAllreduce,

  DenseDPForward,
  DenseDPBackwardIndexCalculation,
  DenseDPLocalReduce,
  DenseDPAllReduce,

  HierMPModelForward,
  HierMPNetworkForward,
  HierMPBackwardIndexCalculation,
  HierMPNetworkBackward,
  HierMPLocalReduce,

  MPModelForward,
  MPNetworkdForward,
  MPBackwardIndexCalculation,
  MPNetworkBackward,
  MPLocalReduce,

  DenseMPModelForward,
  DenseMPNetworkForward,
  DenseMPBackwardIndexCalculation,
  DenseMPNetworkBackward,
  DenseMPLocalReduce,
};
inline const char* to_string(Stage s) {
  switch (s) {
    case Stage::DPForward: return "DPForward";
    case Stage::DPBackwardIndexCalculation: return "DPBackwardIndexCalculation";
    case Stage::DPLocalReduce: return "DPLocalReduce";
    case Stage::DPAllreduce: return "DPAllreduce";

    case Stage::DenseDPForward: return "DenseDPForward";
    case Stage::DenseDPBackwardIndexCalculation: return "DenseDPBackwardIndexCalculation";
    case Stage::DenseDPLocalReduce: return "DenseDPLocalReduce";
    case Stage::DenseDPAllReduce: return "DenseDPAllReduce";

    case Stage::HierMPModelForward: return "HierMPModelForward";
    case Stage::HierMPNetworkForward: return "HierMPNetworkForward";
    case Stage::HierMPBackwardIndexCalculation: return "HierMPBackwardIndexCalculation";
    case Stage::HierMPNetworkBackward: return "HierMPNetworkBackward";
    case Stage::HierMPLocalReduce: return "HierMPLocalReduce";

    case Stage::MPModelForward: return "MPModelForward";
    case Stage::MPNetworkdForward: return "MPNetworkdForward";
    case Stage::MPBackwardIndexCalculation: return "MPBackwardIndexCalculation";
    case Stage::MPNetworkBackward: return "MPNetworkBackward";
    case Stage::MPLocalReduce: return "MPLocalReduce";

    case Stage::DenseMPModelForward: return "DenseMPModelForward";
    case Stage::DenseMPNetworkForward: return "DenseMPNetworkForward";
    case Stage::DenseMPBackwardIndexCalculation: return "DenseMPBackwardIndexCalculation";
    case Stage::DenseMPNetworkBackward: return "DenseMPNetworkBackward";
    case Stage::DenseMPLocalReduce: return "DenseMPLocalReduce";
  }
  return "Unknown";
}

class IGroupedEmbeddingOp {
 public:
  virtual ~IGroupedEmbeddingOp() = default;

  virtual void forward_per_gpu(Stage stage, const EmbeddingInput &embedding_input,
                               ILookup *embedding_table, EmbeddingOutput &embedding_output,
                               int batch_size) = 0;

  virtual void backward_per_gpu(Stage stage, const EmbeddingInput &embedding_input,
                                const EmbeddingOutput &top_grad, Wgrad &wgrad, int batch_size) = 0;

  virtual const WgradAttr &get_wgrad_attr() const = 0;

  virtual bool is_valid_stage(Stage stage) const = 0;
};

std::vector<std::unique_ptr<IGroupedEmbeddingOp>> create_grouped_embeddings(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param);

}  // namespace embedding
