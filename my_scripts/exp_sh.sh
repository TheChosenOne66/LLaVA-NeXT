which python
export OMP_NUM_THREADS=20

VISION_MODEL_VERSION="/data_train/huoyuqi/model_zoo/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# DPO Stage
PROMPT_VERSION="qwen_1_5"
SFT_MODEL="/data_train/mm_intern/zjzhao/code/public_repo/llava-onevision-qwen2-7b-ov"
EPOCH=1
beta=0.1

DPO_RUN_NAME="llava-onevision-qwen2-7b-ov_dpo-beta${beta}-epoch${EPOCH}-multi_node"
DPO_CLEAN_NAME="${DPO_RUN_NAME##*/}"
OUTPUT_DIR="work_dirs/${DPO_CLEAN_NAME}"
DATA_PATH="/data_train/mm_intern/duyifan/dataset/ShareGPTVideo/train_video_and_instruction/video_instruction/train/dpo/sft_dpo_17k.jsonl"
mkdir -p $OUTPUT_DIR
echo $DPO_RUN_NAME

export WANDB_API_KEY=""
wandb login $WANDB_API_KEY

export WANDB_NAME=$DPO_RUN_NAME--$SFT_MODEL

export WANDB_PROJECT=LLaVA_NeXT

export WANDB_MODE=online

wandb online

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
    --module \
    llava.train.train_dpo \
    --deepspeed scripts/zero3.json \
    --model_name_or_path=${SFT_MODEL} \
    --dpo_alpha=1.0 \
    --beta=${beta} \
    --gamma=0 \
    --version $PROMPT_VERSION \
    --data_path=$DATA_PATH \
    --image_folder "/data_train/mm_intern/duyifan/dataset/ShareGPTVideo/train_video_and_instruction/train_300k" \
    --video_folder "/data_train/mm_intern/duyifan/dataset/ShareGPTVideo/train_video_and_instruction/train_300k" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --unfreeze_mm_vision_tower True \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $DPO_CLEAN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    2>&1 | tee $OUTPUT_DIR/train.log