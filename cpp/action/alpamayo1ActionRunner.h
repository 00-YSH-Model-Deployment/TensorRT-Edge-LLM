/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "action/actionKvBatch.h"
#include "action/actionModelTypes.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/externalWeightManager.h"
#include "tokenizer/tokenizer.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Configuration parsed from the action engine's config.json
struct ActionConfig
{
    float ropeTheta{0.0F};       //!< RoPE base frequency (from rope_theta)
    int32_t mropeSectionH{0};    //!< MRoPE frequency pairs for height dimension (from rope_parameters.mrope_section[1])
    int32_t mropeSectionW{0};    //!< MRoPE frequency pairs for width dimension (from rope_parameters.mrope_section[2])
    int32_t numDecoderLayers{0}; //!< Number of transformer decoder layers (from num_hidden_layers)
    int32_t numTrajTokens{0};    //!< Number of trajectory history tokens (from num_traj_tokens)
    int32_t trajTokenStart{0};   //!< Vocabulary ID of the first trajectory token (from traj_token_start)
    int32_t maxKVCacheCapacity{0}; //!< Maximum KV cache sequence capacity (from builder_config.max_kv_cache_capacity)
    int32_t numKVHeads{0};         //!< Number of key-value heads (from num_key_value_heads)
    int32_t headDim{0};            //!< Head dimension (from head_dim)
};

//! \brief Standalone action / diffusion head for Alpamayo 1 trajectory prediction.
//!
//! Consumes the VLM KV cache after generation and produces future trajectory waypoints via a
//! flow-matching denoising loop.
class Alpamayo1ActionRunner
{
public:
    //! \brief Load action engine, config, and allocate tensors
    //! \param engineDir Path to directory containing action.engine and config.json
    //! \param checkpointDir Original model checkpoint used for runtime weights
    //! \param stream CUDA stream for operations
    //! \param kvCacheConfig KV cache layout from the LLM (from KVCacheManager::Config())
    //! \throws std::runtime_error If engine loading, configuration parsing, or allocation fails.
    //!
    //! config.json must include rope_theta and num_hidden_layers (decoder layer count)
    Alpamayo1ActionRunner(std::string const& engineDir, std::string const& checkpointDir, cudaStream_t stream,
        KVCacheManager::Config const& kvCacheConfig);

    ~Alpamayo1ActionRunner() noexcept;

    //! \brief Get the required context memory size for this engine
    //! \return Required context memory size in bytes
    int64_t getRequiredContextMemorySize() const;

    //! \brief Set shared context memory for the execution context
    //! \param sharedContextMemory Tensor containing the shared device memory (must be on GPU)
    //! \return True on success, false if the tensor is too small
    //! \note The tensor size must be >= getRequiredContextMemorySize(). Must be called before infer().
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    //! \brief Get action head model type
    //! \return Action model type
    action::ActionModelType getModelType() const noexcept
    {
        return mModelType;
    }

    //! \brief Set the random seed used when initializing the diffusion noise trajectory
    //! \param seed Random seed value
    void setNoiseSeed(int32_t seed) noexcept
    {
        mNoiseSeed = seed;
    }

    //! \brief Get the max KV cache capacity the action engine was built with
    //! \return Maximum KV cache capacity (from builder_config in engine config.json)
    int32_t getMaxKVCacheCapacity() const noexcept
    {
        return mConfig.maxKVCacheCapacity;
    }

    //! \brief Run one batched diffusion/flow-matching loop and return future trajectory waypoints for all batch items.
    //! Call preprocess() once per request before this (prefill path).
    //! \param stream CUDA stream for operations
    //! \param kvcache KV cache page pool containing the VLM outputs
    //! \param batch Dense action-local page table, lengths, and positional metadata
    std::vector<std::vector<FutureTrajectoryPoint>> sampleTrajectory(
        cudaStream_t stream, HybridCacheManager const& kvcache, ActionKvBatchView const& batch);

    /*!
     * \brief Preprocess batched token IDs for Alpamayo (e.g. replace <|traj_history|> pads with trajectory
     * tokens), validate batch size against allocated action buffers, and initialize diffusion noise for the request.
     * \return True on success, false if trajectory placeholder fill or noise setup fails
     */
    bool preprocess(LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer);

private:
    // Flow time grid 0 = t_0 < ... < t_n = 1: the action engine runs once per interval, an Euler step from t_k to
    // t_{k+1}. EDGELLM_ALPAMAYO_TIME_GRID (comma list) sets it directly. Otherwise it is the uniform grid of
    // EDGELLM_ALPAMAYO_DENOISE_STEPS steps (default 10) with the steps named in EDGELLM_ALPAMAYO_CACHE_STEPS
    // (comma list, never 0) folded into the computed step before them (action_utils::foldCachedEulerSteps).
    std::vector<float> mTimeGrid{};
    // Scale of the initial noise, EDGELLM_ALPAMAYO_NOISE_SCALE (default 1). 0 starts every flow from the
    // mean action, the PyTorch runner's diffusion_temperature=0: one deterministic trajectory per request.
    float mNoiseScale{1.0F};

    //! \brief Read the flow time grid and noise scale from the environment
    void parseDenoiseSchedule();

    //! \brief Allocate tensors for the action runner
    void allocateTensors(KVCacheManager::Config const& kvCacheConfig);

    //! \brief Reshape batch-sized tensors to activeBatchSize
    //! \return false if any reshape fails
    bool reshapeActionTensorsForActiveBatch(int32_t activeBatchSize);

    //! \brief Initialize the noise trajectory
    //! \param randomSeed Random seed for the noise trajectory
    //! \param activeBatchSize Number of batch rows to fill
    void initializeNoiseTrajectory(int32_t randomSeed, int32_t activeBatchSize);

    //! \brief Set TensorRT input shapes for the current active batch (dynamic batch / seq_length).
    //! \param activeBatchSize Number of active sequences
    void setDynamicInputShapes(int32_t activeBatchSize);

    //! \brief Load model fields from config.json (e.g. rope_theta, num_hidden_layers).
    bool parseModelConfig(std::string const& configPath);

    //! \brief Set the input shapes and the addresses that are the same for every flow step. Skipped while the
    //! batch size and KV length buffer are those of the previous call.
    //! \throws std::runtime_error If a shape or address is rejected
    void bindStaticTensors(int32_t activeBatchSize, void* kvLengthsDevice);

    //! \brief Trajectory buffer holding the state at time-grid point k: the noise buffer for even k (k = 0 is the
    //! initial noise), the denoised buffer for odd k
    void* trajectoryBuffer(int32_t k) noexcept;

    //! \brief Enqueue one action engine call per time-grid interval
    //! \return false if an address is rejected or enqueueV3 fails
    bool enqueueDenoiseSteps(cudaStream_t stream);

    //! \brief Capture enqueueDenoiseSteps for activeBatchSize into a CUDA graph; disables graphs on failure.
    //! The stream must be idle.
    void captureDenoiseGraph(int32_t activeBatchSize, cudaStream_t stream);

    //! \brief Destroy the captured denoise graphs
    void resetDenoiseGraphs() noexcept;

    cudaStream_t mStream{nullptr};
    action::ActionModelType mModelType{action::ActionModelType::ALPAMAYO1};
    int32_t mNoiseSeed{5};  //!< Random seed for diffusion noise trajectory initialization
    ActionConfig mConfig{}; //!< Model configuration parsed from config.json

    AuxStreamSet mAuxStreams{};
    std::unique_ptr<nvinfer1::IRuntime> mRuntime{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
    std::unique_ptr<ExternalWeightManager> mExternalWeights{nullptr};

    rt::Tensor mNoiseTrajectoryDevice;
    rt::Tensor mNoiseTrajectoryHost;
    rt::Tensor mDenoisedTrajectoryDevice;
    rt::Tensor mDenoisedTrajectoryHost;
    rt::Tensor mTimeGridDevice; //!< mTimeGrid, one 256-byte slot per point
    rt::Tensor mTimeGridHost;
    rt::Tensor mRopeCosSinDevice;
    rt::Tensor mPositionIdsHost;
    rt::Tensor mPositionIdsDevice;
    rt::Tensor mRopePositionIdsHost;
    rt::Tensor mRopePositionIdsDevice;

    int32_t mNumWaypoints{64};
    int64_t mRopeHeadDim{128};
    int32_t mMaxSequenceLength{0};
    //! Allocated max batch for action staging (min of engine profile and KV cache maxBatchSize).
    int32_t mMaxActionBatchSize{0};
    std::vector<rt::Tensor> mKCacheLayers;
    std::vector<rt::Tensor> mVCacheLayers;

    int32_t mBoundBatchSize{0};           //!< Batch size of the static bindings, 0 if none are valid
    void const* mBoundKvLengths{nullptr}; //!< KV length buffer bound as kvcache_start_index

    //! CUDA graphs of the whole flow loop, one per batch size (EDGELLM_ALPAMAYO_ACTION_CUDA_GRAPH=1). They bake in
    //! every engine address and the context memory, so they are dropped when the KV length buffer or the context
    //! memory changes.
    struct DenoiseGraph
    {
        int32_t batchSize{0};
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t exec{nullptr};
    };
    bool mUseCudaGraph{false};
    std::vector<DenoiseGraph> mDenoiseGraphs{};
};

} // namespace rt
} // namespace trt_edgellm
