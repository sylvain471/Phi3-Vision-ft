import os
import torch
import transformers
from peft import LoraConfig, get_peft_model
import ast
# If you get rid of AutoProcessor, the code dosen't work.
from transformers import AutoProcessor, BitsAndBytesConfig
import sys

#from phi3_vision import Phi3VForCausalLM, Phi3VConfig, Phi3VProcessor

from configuration_phi3_v import Phi3VConfig
from modeling_phi3_v import Phi3VForCausalLM
from processing_phi3_v import Phi3VProcessor
#from image_embedding_phi3_v import Phi3ImageEmbedding

from trainer import Phi3VTrainer
from data import make_supervised_data_module
from params import DataArguments, ModelArguments, TrainingArguments
from train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=["self_attn", "lm_head"], verbose=True):
    linear_cls = torch.nn.modules.Linear
    lora_module_names = []
    lora_namespan_exclude += ["vision_model", "img_projection"]
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, linear_cls):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.lora_enable:
        training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            load_in_4bit=training_args.bits==4,
            load_in_8bit=training_args.bits==8,
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["img_projection"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))
    
    config = Phi3VConfig.from_pretrained(model_args.model_id)

    if training_args.disable_flash_attn2:
        config._attn_implementation = "eager"

    model = Phi3VForCausalLM.from_pretrained(
        model_args.model_id,
        config=config,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir, 
        **bnb_model_from_pretrained_args
    )

    # rank0_print(model)

    model.config.use_cache = False

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    processor = Phi3VProcessor.from_pretrained(model_args.model_id,
                                               cache_dir=training_args.cache_dir, 
                                               padding_side='right', 
                                               model_max_length=training_args.max_seq_length)
    

    # use unk rather than eos token to prevent endless generation
    processor.tokenizer.pad_token = processor.tokenizer.unk_token
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
    processor.tokenizer.padding_side = 'right'

    model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    
    # When using LoRA, the model is rapped once more.
    if training_args.lora_enable:
        vision_tower = model.model.model.vision_embed_tokens.img_processor.vision_model
        vision_tower.to(dtype=compute_dtype, device=training_args.device)

        data_args.is_multimodal = True

        if not training_args.tune_img_projector:
            for p in model.model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = False
        else:
            for p in model.model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = True

        if training_args.freeze_vision_tower:
            for p in model.model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = False
        else:
            for p in model.model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = True


        if training_args.bits in [4, 8]:
            model.model.model.vision_embed_tokens.img_processor.to(dtype=compute_dtype, device=training_args.device)

    else:
        vision_tower = model.model.vision_embed_tokens.img_processor.vision_model
        vision_tower.to(dtype=compute_dtype, device=training_args.device)

        data_args.is_multimodal = True

        if not training_args.tune_img_projector:
            for p in model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = False
        else:
            for p in model.model.vision_embed_tokens.img_projection.parameters():
                p.requires_grad = True

        if training_args.freeze_vision_tower:
            for p in model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = False
        else:
            for p in model.model.vision_embed_tokens.img_processor.vision_model.parameters():
                p.requires_grad = True


        if training_args.bits in [4, 8]:
            model.model.vision_embed_tokens.img_processor.to(dtype=compute_dtype, device=training_args.device)

    model.config.multimodal_lr = training_args.multimodal_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_module():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module.weight = module.weight.to(torch.bfloat16)

    data_module = make_supervised_data_module(processor=processor,
                                              data_args=data_args)

    trainer = Phi3VTrainer(
        model=model,
        processor=processor,
        args=training_args,
        **data_module
    )

    trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()