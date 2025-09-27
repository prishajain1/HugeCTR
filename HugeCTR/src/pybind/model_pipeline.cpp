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

#include <HugeCTR/include/data_readers/multi_hot/async_data_reader.hpp>
#include <algorithm>
#include <core23/logger.hpp>
#include <core23_network.hpp>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <pybind/model.hpp>
#include <nvtx3/nvToolsExt.h>
#include <resource_managers/resource_manager_core.hpp>
#include <sstream>

namespace HugeCTR {

void Model::exchange_wgrad(size_t device_id) {
  auto& gpu_resource = resource_manager_->get_local_gpu(device_id);
  CudaCPUDeviceContext context(gpu_resource->get_device_id());
  if (resource_manager_->get_global_gpu_count() > 1) {
    exchange_wgrad_->allreduce(device_id, gpu_resource->get_stream());
  }
}

void Model::create_train_network_pipeline(std::vector<std::shared_ptr<Network>>& networks) {
  graph_.train_pipeline_.resize(resource_manager_->get_local_gpu_count());

  auto scheduled_reader = dynamic_cast<SchedulableDataReader*>(train_data_reader_.get());

  for (int local_id = 0; local_id < resource_manager_->get_local_gpu_count(); local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    auto BNET_input_ready_wait = std::make_shared<StreamContextScheduleable>([=] {
      auto stream = gpu_resource->get_stream();
      scheduled_reader->stream_wait_dense_tensors(
          stream, local_id,
          solver_.use_cuda_graph && !scheduled_reader->current_batch_incomplete());
    });

    auto network_forward_and_backward = std::make_shared<StreamContextScheduleable>([=] {
      long long current_batchsize_per_device =
          train_data_reader_->get_current_batchsize_per_device(local_id);
      networks[local_id]->train(current_batchsize_per_device);
    });

    std::vector<std::shared_ptr<Scheduleable>> scheduleable_list;
    if (scheduled_reader) {
      scheduleable_list = {BNET_input_ready_wait, network_forward_and_backward};
    } else {
      scheduleable_list = {network_forward_and_backward};
    }

    auto graph = std::make_shared<GraphScheduleable>(scheduleable_list);
    graph_.train_pipeline_[local_id] = Pipeline{"default", gpu_resource, {graph}};
  }
}

void Model::create_eval_network_pipeline(std::vector<std::shared_ptr<Network>>& networks) {
  graph_.evaluate_pipeline_.resize(resource_manager_->get_local_gpu_count());

  for (int local_id = 0; local_id < static_cast<int>(resource_manager_->get_local_gpu_count());
       local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    auto network_eval = std::make_shared<StreamContextScheduleable>([=] {
      long long current_batchsize_per_device =
          evaluate_data_reader_->get_current_batchsize_per_device(local_id);
      networks[local_id]->eval(current_batchsize_per_device);
    });

    auto cal_metrics = std::make_shared<StreamContextScheduleable>([=] {
      for (auto& metric : metrics_) {
        auto metric_map = networks[local_id]->get_raw_metrics_all().begin()->second;
        metric->local_reduce(local_id, metric_map);
      }
    });

    std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {network_eval, cal_metrics};

    auto graph = std::make_shared<GraphScheduleable>(scheduleable_list);
    graph_.evaluate_pipeline_[local_id] = Pipeline{"default", gpu_resource, {graph}};
  }
}

bool is_first_data_distributor = true;

void Model::create_train_pipeline_with_ebc(std::vector<std::shared_ptr<Network>>& networks) {
  bool is_train = true;
  bool use_graph = solver_.use_cuda_graph;

  graph_.train_pipeline_.resize(resource_manager_->get_local_gpu_count());

#pragma omp parallel for num_threads(resource_manager_->get_local_gpu_count())
  for (int local_id = 0; local_id < static_cast<int>(resource_manager_->get_local_gpu_count());
       local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    // create scheduleable
    auto distribute_data = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_prefetch_in_last_batch(is_train)) return;

      const char* const skip_data_distributor_env = std::getenv("SKIP_DATA_DISTRIBUTOR");
      bool skip_data_distributor =
          (skip_data_distributor_env != nullptr && 1 == std::atoi(skip_data_distributor_env));

      if (is_scheduled_datareader()) {
        if (skip_data_distributor && !is_first_data_distributor) {
          auto stream = gpu_resource->get_stream();
          HCTR_LIB_THROW(cudaStreamSynchronize(stream));
          return;
        }
        if (auto reader =
                dynamic_cast<MultiHot::AsyncDataReader<uint32_t>*>(train_data_reader_.get())) {
          train_data_distributor_->distribute(
              local_id, reader->get_current_sparse_values()[local_id], {},
              train_ddl_output_[local_id], train_data_reader_->get_current_batchsize());
        } else if (auto reader = dynamic_cast<MultiHot::AsyncDataReader<long long>*>(
                       train_data_reader_.get())) {
          train_data_distributor_->distribute(
              local_id, reader->get_current_sparse_values()[local_id], {},
              train_ddl_output_[local_id], train_data_reader_->get_current_batchsize());
        }
        is_first_data_distributor = false;
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       "embedding collection can only be used with AsyncMultiHot DataReader.");
      }
    });
    distribute_data->set_label("EBC: Distribute Sparse Inputs");

    const char* const skip_embedding_env = std::getenv("SKIP_EMBEDDING");
    bool skip_embedding = (skip_embedding_env != nullptr && 1 == std::atoi(skip_embedding_env));

    DataDistributor::Result& ddl_output = solver_.train_inter_iteration_overlap
                                              ? cache_train_ddl_output_[local_id]
                                              : train_ddl_output_[local_id];

    auto ebc_forward = [=, &ddl_output](embedding::Stage stage) {
      if (skip_embedding) return;

      for (auto& ebc : ebc_list_) {
        ebc->forward_per_gpu(stage, is_train, local_id, ddl_output, train_ebc_outptut_[local_id],
                             train_data_reader_->get_full_batchsize());
      }
    };

    auto ebc_backward = [=, &ddl_output](embedding::Stage stage) {
      if (skip_embedding) return;

      for (auto& ebc : ebc_list_) {
        ebc->backward_per_gpu(stage, local_id, ddl_output, train_ebc_outptut_[local_id],
                              train_data_reader_->get_full_batchsize());
      }
    };

    auto ebc_mp_model_forward = std::make_shared<StreamContextScheduleable>([=] {
      ebc_forward(embedding::Stage::MPModelForward);
      ebc_forward(embedding::Stage::HierMPModelForward);
      ebc_forward(embedding::Stage::DenseMPModelForward);
    });

    ebc_mp_model_forward->set_label("EBC: Model-parallel embedding forward");

    auto ebc_mp_network_forward = std::make_shared<StreamContextScheduleable>([=] {
      ebc_forward(embedding::Stage::MPNetworkdForward);
      ebc_forward(embedding::Stage::HierMPNetworkForward);
      ebc_forward(embedding::Stage::DenseMPNetworkForward);
    });

    ebc_mp_network_forward->set_label("EBC: Network-parallel embedding forward");

    auto ebc_mp_network_backward = std::make_shared<StreamContextScheduleable>([=]() {
      ebc_backward(embedding::Stage::MPNetworkBackward);
      ebc_backward(embedding::Stage::HierMPNetworkBackward);
      ebc_backward(embedding::Stage::DenseMPNetworkBackward);
    });

    ebc_mp_network_backward->set_label("EBC: Network-parallel embedding backward");

    auto ebc_mp_backward_index_calculation = std::make_shared<StreamContextScheduleable>([=] {
      ebc_backward(embedding::Stage::MPBackwardIndexCalculation);
      ebc_backward(embedding::Stage::HierMPBackwardIndexCalculation);
      ebc_backward(embedding::Stage::DenseMPBackwardIndexCalculation);
    });

    ebc_mp_backward_index_calculation->set_label("EBC: MP Embedding backward index calculation");

    auto ebc_mp_local_reduce = std::make_shared<StreamContextScheduleable>([=]() {
      ebc_backward(embedding::Stage::HierMPLocalReduce);
      ebc_backward(embedding::Stage::MPLocalReduce);
      ebc_backward(embedding::Stage::DenseMPLocalReduce);
    });

    ebc_mp_local_reduce->set_label("EBC: MP Embedding local reduce");

    auto ebc_mp_update = std::make_shared<StreamContextScheduleable>([=]() {
      if (skip_embedding) return;

      for (auto& ebc : ebc_list_) {
        ebc->update_per_gpu(local_id, embedding::EmbeddingGroupType::SparseModelParallel);
        ebc->update_per_gpu(local_id, embedding::EmbeddingGroupType::DenseModelParallel);
      }
    });
    ebc_mp_update->set_label("EBC: MP Embedding update");

    auto ebc_dp_forward = std::make_shared<StreamContextScheduleable>(
        [=] { ebc_forward(embedding::Stage::DPForward); });
    ebc_dp_forward->set_label("EBC: Data-parallel embedding forward");

    auto ebc_dp_backward_index_calculation = std::make_shared<StreamContextScheduleable>(
        [=] { ebc_backward(embedding::Stage::DPBackwardIndexCalculation); });
    ebc_dp_backward_index_calculation->set_label("EBC: DP Embedding backward index calculation");

    auto ebc_dp_local_reduce = std::make_shared<StreamContextScheduleable>(
        [=]() { ebc_backward(embedding::Stage::DPLocalReduce); });
    ebc_dp_local_reduce->set_label("EBC: DP Embedding local reduce");

    auto ebc_dp_allreduce = std::make_shared<StreamContextScheduleable>(
        [=]() { ebc_backward(embedding::Stage::DPAllreduce); });
    ebc_mp_network_backward->set_label("EBC: DP Embedding allreduce");

    auto ebc_dp_update = std::make_shared<StreamContextScheduleable>([=]() {
      if (skip_embedding) return;

      for (auto& ebc : ebc_list_) {
        ebc->update_per_gpu(local_id, embedding::EmbeddingGroupType::DataParallel);
      }
    });
    ebc_dp_update->set_label("EBC: DP Embedding update");

    const char* const skip_bottom_mlp_env = std::getenv("SKIP_BOTTOM_MLP");
    bool skip_bottom_mlp = (skip_bottom_mlp_env != nullptr && 1 == std::atoi(skip_bottom_mlp_env));

    const char* const skip_top_mlp_env = std::getenv("SKIP_TOP_MLP");
    bool skip_top_mlp = (skip_top_mlp_env != nullptr && 1 == std::atoi(skip_top_mlp_env));

    auto network_init = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_bottom_mlp) return;

      if (networks[local_id]->use_mixed_precision_ &&
          networks[local_id]->optimizer_->get_optimizer_type() != Optimizer_t::SGD) {
        networks[local_id]->conv_weight_(networks[local_id]->train_weight_tensor_half_,
                                         networks[local_id]->train_weight_tensor_);
      }
    });
    network_init->set_label("Network init");

    auto bottom_network_fprop = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_bottom_mlp) return;
      networks[local_id]->prop_layers(networks[local_id]->bottom_layers_, true, is_train);
    });
    bottom_network_fprop->set_label("Bottom MLP fprop");

    auto top_network_fprop = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_top_mlp) return;
      networks[local_id]->prop_layers(networks[local_id]->top_layers_, true, is_train);
    });
    top_network_fprop->set_label("Top MLP fprop");

    auto init_wgrad = std::make_shared<StreamContextScheduleable>([=] {
      networks[local_id]->train_losses_.begin()->second->regularizer_initialize_wgrad(is_train);
    });
    init_wgrad->set_label("Init wgrad");

    auto cal_loss = std::make_shared<StreamContextScheduleable>([=] {
      float rterm = networks[local_id]->train_losses_.begin()->second->regularizer_compute_rterm();
      long long current_batchsize_per_device =
          graph_.is_last_train_batch_
              ? train_data_reader_->get_current_batchsize_per_device(local_id)
              : train_data_reader_->get_full_batchsize() /
                    resource_manager_->get_global_gpu_count();

      networks[local_id]->train_losses_.begin()->second->compute(
          is_train, current_batchsize_per_device, rterm);
    });
    cal_loss->set_label("Calculate loss");

    auto top_network_bprop = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_top_mlp) return;
      networks[local_id]->prop_layers(networks[local_id]->top_layers_, false, is_train);
    });
    top_network_bprop->set_label("Top MLP bprop");

    auto bottom_network_bprop = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_bottom_mlp) return;
      networks[local_id]->prop_layers(networks[local_id]->bottom_layers_, false, is_train);
    });
    bottom_network_bprop->set_label("Bottom MLP bprop");

    auto network_graph = std::make_shared<GraphScheduleable>(
        network_init, bottom_network_fprop, top_network_fprop, init_wgrad, cal_loss,
        top_network_bprop, bottom_network_bprop);

    const char* const skip_allreduce_env = std::getenv("SKIP_ALLREDUCE");
    bool skip_allreduce = (skip_allreduce_env != nullptr && 1 == std::atoi(skip_allreduce_env));

    auto network_exchange_wgrad = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_allreduce) return;
      this->exchange_wgrad(local_id);
    });
    network_exchange_wgrad->set_label("Allreduce wgrad");

    auto update_params =
        std::make_shared<StreamContextScheduleable>([=] { networks[local_id]->update_params(); });
    update_params->set_label("Update params");

    auto sync_back = std::make_shared<StreamContextScheduleable>([] {});
    sync_back->set_label("Sync back");

    if (solver_.train_intra_iteration_overlap) {
      std::string dp_stream = "dp";
      ebc_dp_forward->set_stream(dp_stream);
      ebc_dp_backward_index_calculation->set_stream(dp_stream);
      ebc_mp_backward_index_calculation->set_stream(dp_stream);
      ebc_dp_local_reduce->set_stream(dp_stream);
      ebc_dp_update->set_stream(dp_stream);

      std::string mp_stream = "mp";
      ebc_mp_model_forward->set_stream(mp_stream);
      ebc_mp_network_forward->set_stream(mp_stream);
      ebc_mp_network_backward->set_stream(mp_stream);
      ebc_mp_local_reduce->set_stream(mp_stream);
      ebc_mp_update->set_stream(mp_stream);

      // dp_emb_forward, bmlp_fprop wait for mp_emb_model_forward
      auto done_mp_model_forward = ebc_mp_model_forward->record_done();
      ebc_dp_forward->wait_event({done_mp_model_forward});
      bottom_network_fprop->wait_event({done_mp_model_forward}, use_graph);

      // tmlp_fprop wait for embedding
      auto done_mp_network_forward = ebc_mp_network_forward->record_done();
      auto done_dp_forward = ebc_dp_forward->record_done();
      top_network_fprop->wait_event({done_dp_forward, done_mp_network_forward}, use_graph);

      // mp_emb_bck, dp_emb_bck wait for tmlp bprop
      auto done_top_network_bprop = top_network_bprop->record_done(use_graph);
      ebc_mp_network_backward->wait_event({done_top_network_bprop});
      ebc_dp_local_reduce->wait_event({done_top_network_bprop});

      // mp_local_reduce wait mp_backward_index_calculation
      auto done_ebc_mp_backward_index_calculation =
          ebc_mp_backward_index_calculation->record_done();
      ebc_mp_local_reduce->wait_event({done_ebc_mp_backward_index_calculation});

      // allreduce wait dp local reduce
      auto done_ebc_dp_local_reduce = ebc_dp_local_reduce->record_done();
      network_exchange_wgrad->wait_event({done_ebc_dp_local_reduce});

      // dp update wait allreduce
      auto done_ebc_dp_allreduce = ebc_dp_allreduce->record_done();
      ebc_dp_update->wait_event({done_ebc_dp_allreduce});

      // sync back
      auto done_ebc_dp_update = ebc_dp_update->record_done();
      auto done_ebc_mp_update = ebc_mp_update->record_done();
      sync_back->wait_event({done_ebc_dp_update, done_ebc_mp_update});
    }

    if (!solver_.train_inter_iteration_overlap) {
      std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {
          distribute_data,
          ebc_mp_model_forward,
          ebc_mp_network_forward,
          ebc_dp_forward,
          network_graph,
          ebc_mp_backward_index_calculation,
          ebc_dp_backward_index_calculation,
          ebc_mp_network_backward,
          ebc_dp_local_reduce,
          network_exchange_wgrad,
          ebc_dp_allreduce,
          update_params,
          ebc_mp_local_reduce,
          ebc_mp_update,
          ebc_dp_update,
          sync_back,
      };
      auto done_distribute_data = distribute_data->record_done();
      ebc_mp_model_forward->wait_event({done_distribute_data});

      graph_.train_pipeline_[local_id] = Pipeline{"default", gpu_resource, scheduleable_list};
    } else {
      auto ebc_cache_train_ddl_output =
          std::make_shared<StreamContextScheduleable>([=, &ddl_output] {
            for (auto& ebc : ebc_list_) {
              ebc->cache_ddl_output(local_id, train_ddl_output_[local_id], ddl_output,
                                    train_data_reader_->get_full_batchsize());
            }
          });
      ebc_cache_train_ddl_output->set_label("EBC: Cache DDL output for overlap");

      auto copy_next_iter_network_input = std::make_shared<StreamContextScheduleable>([=]() {
        if (skip_prefetch_in_last_batch(is_train)) return;

        graph_.train_copy_ops_[local_id]->run();
        graph_.train_copy_ops_[local_id + resource_manager_->get_local_gpu_count()]->run();
      });
      copy_next_iter_network_input->set_label("Copy next iter network input");

      std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {
          ebc_cache_train_ddl_output,
          ebc_mp_model_forward,
          ebc_mp_network_forward,
          ebc_dp_forward,
          network_graph,
          ebc_mp_backward_index_calculation,
          ebc_dp_backward_index_calculation,
          distribute_data,
          ebc_mp_network_backward,
          ebc_dp_local_reduce,
          network_exchange_wgrad,
          ebc_dp_allreduce,
          update_params,
          ebc_mp_local_reduce,
          ebc_mp_update,
          ebc_dp_update,
          sync_back,
          copy_next_iter_network_input,
      };
      std::string prefetch_stream = "prefetch";

      auto done_ebc_cache_train_ddl_output = ebc_cache_train_ddl_output->record_done();
      ebc_mp_model_forward->wait_event({done_ebc_cache_train_ddl_output});

      distribute_data->set_absolute_stream(prefetch_stream);
      distribute_data->wait_event({done_ebc_cache_train_ddl_output});

      auto done_distribute_data = distribute_data->record_done();
      ebc_cache_train_ddl_output->wait_event({done_distribute_data});
      graph_.train_pipeline_[local_id] = Pipeline{"default", gpu_resource, scheduleable_list};
    }
  }
}

void Model::train_pipeline_with_ebc() {
  if (graph_.is_first_train_batch_ && solver_.train_inter_iteration_overlap) {
#pragma omp parallel num_threads(number_of_networks())
    {
      size_t id = omp_get_thread_num();
      CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(id)->get_device_id());
      HCTR_CHECK(solver_.use_embedding_collection);
      nvtxRangePushA("Train Batch Data Distribute");
      if (is_scheduled_datareader()) {
        if (auto reader =
                dynamic_cast<MultiHot::AsyncDataReader<uint32_t>*>(train_data_reader_.get())) {
          train_data_distributor_->distribute(id, reader->get_current_sparse_values()[id], {},
                                              train_ddl_output_[id],
                                              train_data_reader_->get_full_batchsize());
        } else if (auto reader = dynamic_cast<MultiHot::AsyncDataReader<long long>*>(
                       train_data_reader_.get())) {
          train_data_distributor_->distribute(id, reader->get_current_sparse_values()[id], {},
                                              train_ddl_output_[id],
                                              train_data_reader_->get_full_batchsize());
        }
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       "embedding collection can only be used with AsyncMultiHot DataReader.");
      }
      nvtxRangePop();
      nvtxRangePushA("Train Copy ops");
      graph_.train_copy_ops_[id]->run();
      graph_.train_copy_ops_[id + resource_manager_->get_local_gpu_count()]->run();
      nvtxRangePop();
      HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream()));
    }

    if (!graph_.is_last_eval_batch_) {
      bool is_train = true;
      nvtxRangePushA("Prefetch next batch");
      long long current_batchsize = read_a_batch(is_train);
      nvtxRangePop();
      if (!current_batchsize) return;
    }
  }

  const bool use_graph = solver_.use_cuda_graph && !train_data_reader_->current_batch_incomplete();
#pragma omp parallel num_threads(resource_manager_->get_local_gpu_count())
  {
    int id = omp_get_thread_num();
    auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
    CudaCPUDeviceContext context(device_id);

    if (use_graph) {
      graph_.train_pipeline_[id].run_graph();
    } else {
      graph_.train_pipeline_[id].run();
    }
  }
}

void Model::create_evaluate_pipeline_with_ebc(std::vector<std::shared_ptr<Network>>& networks) {
  bool is_train = false;
  //  bool use_graph = solver_.use_cuda_graph;
  graph_.evaluate_pipeline_.resize(resource_manager_->get_local_gpu_count());

#pragma omp parallel for num_threads(resource_manager_->get_local_gpu_count())
  for (int local_id = 0; local_id < static_cast<int>(resource_manager_->get_local_gpu_count());
       local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    auto eval_data_distribute = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_prefetch_in_last_batch(is_train)) return;
      if (is_scheduled_datareader()) {
        if (auto reader =
                dynamic_cast<MultiHot::AsyncDataReader<uint32_t>*>(evaluate_data_reader_.get())) {
          eval_data_distributor_->distribute(
              local_id, reader->get_current_sparse_values()[local_id], {},
              evaluate_ddl_output_[local_id], evaluate_data_reader_->get_current_batchsize());
        } else if (auto reader = dynamic_cast<MultiHot::AsyncDataReader<long long>*>(
                       evaluate_data_reader_.get())) {
          eval_data_distributor_->distribute(
              local_id, reader->get_current_sparse_values()[local_id], {},
              evaluate_ddl_output_[local_id], evaluate_data_reader_->get_current_batchsize());
        }
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       "embedding collection can only be used with AsyncMultiHot DataReader.");
      }
    });
    eval_data_distribute->set_label("Distribute Sparse Inputs");

    DataDistributor::Result& ddl_output = solver_.eval_inter_iteration_overlap
                                              ? cache_evaluate_ddl_output_[local_id]
                                              : evaluate_ddl_output_[local_id];

    auto ebc_forward = [=, &ddl_output](embedding::Stage stage) {
      for (auto& ebc : ebc_list_) {
        ebc->forward_per_gpu(stage, is_train, local_id, ddl_output, evaluate_ebc_outptut_[local_id],
                             evaluate_data_reader_->get_full_batchsize());
      }
    };

    auto ebc_mp_model_forward = std::make_shared<StreamContextScheduleable>([=] {
      ebc_forward(embedding::Stage::MPModelForward);
      ebc_forward(embedding::Stage::HierMPModelForward);
      ebc_forward(embedding::Stage::DenseMPModelForward);
    });

    ebc_mp_model_forward->set_label("Model-parallel embedding forward");

    auto ebc_mp_network_forward = std::make_shared<StreamContextScheduleable>([=] {
      ebc_forward(embedding::Stage::MPNetworkdForward);
      ebc_forward(embedding::Stage::HierMPNetworkForward);
      ebc_forward(embedding::Stage::DenseMPNetworkForward);
    });

    ebc_mp_network_forward->set_label("Network-parallel embedding forward");

    auto ebc_dp_forward = std::make_shared<StreamContextScheduleable>(
        [=] { ebc_forward(embedding::Stage::DPForward); });

    ebc_dp_forward->set_label("Data-parallel embedding forward");

    auto network_eval = std::make_shared<StreamContextScheduleable>([=] {
      long long current_batchsize_per_device =
          graph_.is_last_eval_batch_
              ? evaluate_data_reader_->get_current_batchsize_per_device(local_id)
              : evaluate_data_reader_->get_full_batchsize() /
                    resource_manager_->get_global_gpu_count();
      networks[local_id]->eval(current_batchsize_per_device);
    });

    auto network_graph = std::make_shared<GraphScheduleable>(network_eval);
    network_graph->set_label("Dense Network Forward + Loss calc (per GPU)");

    auto cal_metrics = std::make_shared<StreamContextScheduleable>([=] {
      for (auto& metric : metrics_) {
        auto metric_map = networks[local_id]->get_raw_metrics_all().begin()->second;
        metric->local_reduce(local_id, metric_map);
      }
    });
    cal_metrics->set_label("Per-GPU Metric Accumulate");

    if (!solver_.eval_inter_iteration_overlap) {
      graph_.evaluate_pipeline_[local_id] =
          Pipeline{"default",
                   gpu_resource,
                   {eval_data_distribute, ebc_mp_model_forward, ebc_mp_network_forward,
                    ebc_dp_forward, network_graph, cal_metrics}};
    } else {
      auto ebc_cache_eval_ddl_output =
          std::make_shared<StreamContextScheduleable>([=, &ddl_output] {
            for (auto& ebc : ebc_list_) {
              ebc->cache_ddl_output(local_id, evaluate_ddl_output_[local_id], ddl_output,
                                    evaluate_data_reader_->get_full_batchsize());
            }
          });
      ebc_cache_eval_ddl_output->set_label("EBC_Sparse_Input_Staging");

      auto copy_next_iter_network_input = std::make_shared<StreamContextScheduleable>([=]() {
        if (skip_prefetch_in_last_batch(is_train)) return;

        graph_.evaluate_copy_ops_[local_id]->run();
        graph_.evaluate_copy_ops_[local_id + resource_manager_->get_local_gpu_count()]->run();
      });
      copy_next_iter_network_input->set_label("Copy next iter network input");

      std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {
          ebc_cache_eval_ddl_output,
          ebc_mp_model_forward,
          eval_data_distribute,
          ebc_mp_network_forward,
          ebc_dp_forward,
          network_graph,
          cal_metrics,
          copy_next_iter_network_input,
      };

      //      ebc_mp_network_forward->set_stream("side_stream");
      //      ebc_dp_forward->set_stream("side_stream");
      //      network_eval->set_stream("side_stream");
      //      cal_metrics->set_stream("side_stream");

      eval_data_distribute->set_absolute_stream("prefetch");

      auto done_ebc_cache_eval_ddl_output = ebc_cache_eval_ddl_output->record_done();
      eval_data_distribute->wait_event({done_ebc_cache_eval_ddl_output});

      auto done_eval_data_distribute = eval_data_distribute->record_done();
      ebc_cache_eval_ddl_output->wait_event({done_eval_data_distribute});

      //      auto done_ebc_mp_model_forward = ebc_mp_model_forward->record_done();
      //      ebc_mp_network_forward->wait_event({done_ebc_mp_model_forward});
      //
      //      auto done_cal_metrics = cal_metrics->record_done();
      //      copy_next_iter_network_input->wait_event({done_cal_metrics});
      graph_.evaluate_pipeline_[local_id] = Pipeline{"default", gpu_resource, scheduleable_list};
    }
  }
}

void Model::evaluate_pipeline_with_ebc() {
  if (graph_.is_first_eval_batch_ && solver_.eval_inter_iteration_overlap) {
#pragma omp parallel num_threads(number_of_networks())
    {
      size_t id = omp_get_thread_num();
      CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(id)->get_device_id());
      nvtxRangePushA("Eval Batch Data Distribute");
      if (is_scheduled_datareader()) {
        if (auto reader =
                dynamic_cast<MultiHot::AsyncDataReader<uint32_t>*>(evaluate_data_reader_.get())) {
          eval_data_distributor_->distribute(id, reader->get_current_sparse_values()[id], {},
                                             evaluate_ddl_output_[id],
                                             evaluate_data_reader_->get_current_batchsize());
        } else if (auto reader = dynamic_cast<MultiHot::AsyncDataReader<long long>*>(
                       evaluate_data_reader_.get())) {
          eval_data_distributor_->distribute(id, reader->get_current_sparse_values()[id], {},
                                             evaluate_ddl_output_[id],
                                             evaluate_data_reader_->get_current_batchsize());
        }

      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       "embedding collection can only be used with AsyncMultiHot DataReader.");
      }
      nvtxRangePop();
      nvtxRangePushA("Eval Copy ops");
      graph_.evaluate_copy_ops_[id]->run();
      graph_.evaluate_copy_ops_[id + resource_manager_->get_local_gpu_count()]->run();
      nvtxRangePop();
      HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream()));
    }

    // in case we only have 1 batch in eval
    if (!graph_.is_last_eval_batch_){
      bool is_train = false;
      nvtxRangePushA("Prefetch next batch");
      long long current_batchsize = read_a_batch(is_train);
      nvtxRangePop();
      if (!current_batchsize) return;
    }
  }

  const bool use_graph =
      solver_.use_cuda_graph && !evaluate_data_reader_->current_batch_incomplete();
#pragma omp parallel num_threads(number_of_networks())
  {
    size_t id = omp_get_thread_num();
    auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
    CudaCPUDeviceContext context(device_id);

    if (use_graph) {
      graph_.evaluate_pipeline_[id].run_graph();
    } else {
      graph_.evaluate_pipeline_[id].run();
    }
  }
  nvtxRangePushA("Metric Global Reduce (Collective)");
  for (auto& metric : metrics_) {
    metric->global_reduce(number_of_networks());
  }
  nvtxRangePop();
}
}  // namespace HugeCTR
