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

#include "common/hashUtils.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/dflashDecodeUtils.h"
#include "runtime/decoding/specCommonStateTracker.h"
#include "runtime/state/externalWeightManager.h"

#include <filesystem>
#include <memory>

namespace trt_edgellm
{
namespace rt
{

class DFlashDecoder final : public DecodingStrategy
{
public:
    DFlashDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
        dflash_utils::CachedBlockDraftRuntimeConfig blockDraftConfig, std::unique_ptr<EngineExecutor> draftExecutor,
        ExternalWeightManager draftWeights, cudaStream_t stream);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kDFlash;
    }

    char const* name() const noexcept override
    {
        return userModeName();
    }

    bool isSpeculative() const noexcept override
    {
        return true;
    }

    DecodingStrategyCapabilities capabilities() const noexcept override
    {
        return {/*.ownsBaseVerificationCudaGraphs=*/true};
    }

    DecodingKvHeadroom requiredKvHeadroom() const override;

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;
    bool initializeForGeneration(DecodingInferenceContext& context) override;
    std::vector<int32_t> const& commonMaterializedStateLengths() const noexcept override;

    int64_t getRequiredContextMemorySize() const noexcept override;
    void setContextMemory(Tensor& memory) override;

    bool hasSystemPromptKVCache(SystemPromptCacheKey const& key) const override;
    void restoreSystemPromptKVCache(SystemPromptCacheKey const& key, int32_t batchIdx, cudaStream_t stream) override;
    bool runSystemPromptPrefill(DecodingInferenceContext& context) override;
    void saveSystemPromptKVCache(SystemPromptCacheKey const& key, std::string const& prompt,
        std::vector<tokenizer::Rank> const& tokenizedPrompt, int32_t promptIdsLength, cudaStream_t stream) override;

    void resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream) override;
    void onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch,
        Tensor& deviceBatchMapping, cudaStream_t stream) override;

private:
    bool runDraftForward(DecodingInferenceContext& context);
    //! Draft-forward steps that do not involve the draft engine: the verify root and the target-hidden rows this
    //! round adds to the draft context. Returns the target-hidden length the draft engine would bind.
    int64_t stageDraftTargets(DecodingInferenceContext& context);
    //! Asks context.externalDraft for this step's proposal and uploads it to mDraftTokenIds; returns the number
    //! of proposed tokens, 0 to leave the step to the draft engine.
    int32_t stageExternalDraft(DecodingInferenceContext& context);
    //! Trajectory-start guard (context.trajFutureStartId >= 0): when every anchor is <|cot_end|> or
    //! <|traj_future_start|>, uploads [<|traj_future_start|>] or no proposal to mDraftTokenIds and returns true, so
    //! the draft engine is skipped; generation stops once <|traj_future_start|> is in the KV.
    bool stageTrajStartProposal(DecodingInferenceContext& context);
    bool prepareBlockDraftVerifyInputs(DecodingInferenceContext& context, bool externalDraft = false);
    bool captureDraftCudaGraphs(cudaStream_t stream);
    bool buildTreeVerifyInputs(DecodingInferenceContext& context);
    bool runBaseVerification(DecodingInferenceContext& context);
    bool executeBaseVerification(DecodingInferenceContext& context, int32_t verifySize);
    void reshapeBaseVerificationForCapture(int32_t batchSize, int32_t verifySize, bool includeTreeMetadata);
    void prepareLinearBaseVerificationMetadata(int32_t batchSize, int32_t verifySize, cudaStream_t stream);
    void copyVerifyTokenIdsToBaseInput(int32_t batchSize, int32_t verifySize, cudaStream_t stream);
    void runBaseVerificationEmbeddingLookup(
        int32_t batchSize, int32_t verifySize, cudaStream_t stream, bool reshapeGemmaPleOutputs);
    bool capturePreparedBaseVerification(int32_t batchSize, int32_t verifySize, cudaStream_t stream);
    void reshapeBaseVerificationInputsOutputs(int32_t batchSize, int32_t verifySize);
    void prepareCommonBaseVerificationInputs(int32_t batchSize, int32_t verifySize);
    void commitAcceptedTreePath(DecodingInferenceContext& context, int32_t verifySize, int32_t maxAcceptLength);
    void bindTargetHiddenDelta(
        int32_t activeBatchSize, int64_t maxDeltaLen, int64_t sourceSeqLen, bool allowLargeDelta, cudaStream_t stream);
    int64_t slideDraftContextWindow(int32_t activeBatchSize, int64_t sourceSeqLen, bool firstRound, cudaStream_t stream);
    bool checkCudaLastError(char const* stage) const;
    bool useDDTree() const noexcept
    {
        return mBlockDraft.treePolicy == dflash_utils::BlockDraftTreePolicy::kDDTree;
    }
    bool causalProposalMask() const noexcept
    {
        return mBlockDraft.proposalAttention == dflash_utils::ProposalAttentionPolicy::kCausal;
    }
    char const* userModeName() const noexcept
    {
        return specDecodeModeName(mBlockDraft.userMode);
    }

    DecodingRuntimeContext& mRuntime;
    HybridCacheManager& mDraftCacheManager;
    dflash_utils::CachedBlockDraftRuntimeConfig mBlockDraft;

    std::unique_ptr<EngineExecutor> mDraftExecutor;
    TensorMap mDraftTensorMap;
    ExternalWeightManager mDraftExternalWeightManager;

    Tensor mDraftInputsEmbeds;        //!< [B, blockSize, draftHiddenSize] FP16
    Tensor mDraftTargetHidden;        //!< Compact scratch for [B, <= blockSize, baseOutputHiddenDim] FP16
    Tensor mDraftPrefillTargetHidden; //!< Lazy scratch for non-compact round-0 target hidden FP16, max batch reserve
    Tensor mDraftOutputLogits;        //!< [B, blockSize, vocabSize] FP32

    //! EDGELLM_DFLASH_CONTEXT_WINDOW=N (0 = off, N <= blockSize): the draft attends only to the last N target
    //! rows, rewritten each round at draft positions [0, N), for drafts trained on a short context window
    //! (z-lab Alpamayo DFlash uses 8).
    int32_t mContextWindow{0};
    Tensor mDraftWindow[2];                //!< [B, mContextWindow, baseOutputHiddenDim] FP16, ping-pong
    int32_t mDraftWindowCurrent{0};        //!< Index of the buffer holding the latest window
    std::vector<int32_t> mDraftWindowFill; //!< Valid rows per batch slot

    Tensor mDraftPackedAttentionMask; //!< [B, blockSize, divUp(blockSize,32)] INT32
    Tensor mDraftAttentionPosId;      //!< [B, blockSize] INT32
    Tensor mDraftContextLengths;      //!< [B] INT32
    Tensor mDraftDeltaLenCommit;      //!< [B] INT32
    Tensor mDraftDeltaLens;           //!< [B] INT32

    Tensor mDraftTokenIds;          //!< [B, blockSize] INT32
    Tensor mHostDraftInputIds;      //!< [B, blockSize] INT32 CPU
    Tensor mHostLastAcceptedTokens; //!< [B] INT32 CPU
    Tensor mHostDeltaLens;          //!< [B] INT32 CPU
    Tensor mLastAcceptedTokens;     //!< [B] INT32 GPU

    //! External draft (context.externalDraft): host staging [B, blockSize] for proposals; slot 0 is the root.
    Tensor mHostExternalDraft;
    int32_t mExternalDraftCooldown{0}; //!< Draft-engine steps to take before trying the external draft again
    dflash_utils::ExternalDraftPolicy mExternalDraftPolicy{dflash_utils::externalDraftPolicyFromEnv()};

    hash_utils::HashMap<SystemPromptCacheKey, SystemPromptKVCache> mSystemPromptKVCacheDraft;

    Tensor mTreeTokenIds;         //!< [B, verifySize] INT32, DDTree node tokens
    Tensor mTreeNodeScores;       //!< [B, verifySize] FP32, DDTree prefix scores
    Tensor mValidCounts;          //!< [B] INT32, DDTree valid node counts
    Tensor mVerifyTokenIds;       //!< [B, verifyTokenCount] INT32
    Tensor mVerifyTreeMask;       //!< [B, verifyTokenCount, verifyTokenCount] INT8
    Tensor mAcceptedTokenIds;     //!< [B, maxAcceptBufferSize] INT32
    Tensor mAcceptedTokenIndices; //!< [B, maxAcceptBufferSize] INT32, verify logits/KV indices
    Tensor mAcceptLength;         //!< [B] INT32
    Tensor mHostAcceptLengths;    //!< [B] INT32 (CPU)
    Tensor mHostAcceptedTokenIds; //!< [B, maxAcceptBufferSize] INT32 (CPU)

    SpecCommonStateTracker mCommonStateTracker;
    Tensor mBuildWorkspace; //!< DDTree build workspace bytes

    //! Draft vocab map [reducedVocabSize] INT32 (GPU). Active when draft
    //! lm_head uses a reduced vocabulary. Sized to zero otherwise.
    Tensor mDraftVocabMappingTable;
    bool mHasDraftVocabMap{false};
};

} // namespace rt
} // namespace trt_edgellm
