
from baukit import TraceDict, get_module
from models import *
from preprocess import *
import sys
import torch
import numpy as np
import json
import random
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, logging
import sys
from torchvision.ops.boxes import box_area

logging.set_verbosity_warning()
import warnings

torch.autograd.set_detect_anomaly(True)

sys.path.append('../eval_mm')
from vqa import VQA
from vqa_eval import VQAEval


def load_model(model_name, cur_dataset):

    """
    A function that loads the model and a corresponding model_helper. Refer to model.py for more detail.

    Parameters:
    model_name: The name of the model you are attempting to load
    cur_dataset: The name of dataset you are attempting to load

    Returns: 
    model_helper: A helper class that contains the model as well as other functionality.
    """
    if model_name == "llava_ov":
        from llava.model.builder import load_pretrained_model
        
        pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"

        
        model_name = "llava_qwen"
        device_map = "auto"
        llava_model_args = {
                "multimodal": True,
                # "image_aspect_ratio":"pad"
            }
        ###For finetuned models

        # overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}
        # overwrite_config = {}
        # overwrite_config["image_aspect_ratio"] = "pad"
        # llava_model_args["overwrite_config"] = overwrite_config


        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
        #tokenizer, model, image_processor, max_length = load_pretrained_model("/home/zhaobin/LLaVA-NeXT/checkpoints/Mhalu_sft", pretrained, "llava_qwen_lora", device_map=device_map, **llava_model_args)

        # ###TODO:DELETE, FOR BLINK
        # tokenizer, model, image_processor, max_length = load_pretrained_model(lora_path, pretrained, "llava_qwen_lora", device_map=device_map, **llava_model_args)

        model.eval()
        model.requires_grad_(False)
        model_helper = llavaOVHelper(model, tokenizer, image_processor, cur_dataset)
    if model_name == "Qwen-VL":
        
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id

        model_helper = QwenHelper(model, tokenizer, cur_dataset)

    if model_name == "ViLA":
        from peft import PeftModel, PeftConfig

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        model_name = get_model_name_from_path("Efficient-Large-Model/Llama-3-VILA1.5-8b")
        tokenizer, model, image_processor, context_len = load_pretrained_model("Efficient-Large-Model/Llama-3-VILA1.5-8b", model_name, None)
        model_helper = ViLAHelper(model, tokenizer, image_processor, cur_dataset)

    if model_name == "idefics2":
        
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        processor.image_processor.do_image_splitting = False
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            device_map="auto"
        )

        model_helper = Idefics2Helper(model, processor, cur_dataset)

    return model_helper


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L15
def gather_last_attn_activations(inputs, model_helper):

    """
    A function that performs a forward pass and extract the activation at certain location of the layer.

    Parameters:
    inputs: input to the model. Created with model_helper
    model_helper

    Returns: 
    td: The attention activations.
    result: The output logits from forward method.
    """

    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], retain_input=True, retain_output=True) as td:                
        result = model_helper.forward(inputs)
    return td, result


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L65
def split_activations_by_head(activations, model_config):

    """
    The model concatenate the output of multi-headed attention to a single vector. This function splits this vector back to different heads.
    From

    Parameters:
    activations: From gather_last_attn_activations
    model_config: Refer to model.py

    Returns: 
    the activation partitioned by attention heads
    """

    new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
    activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
    return activations.to("cuda")


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L46
def get_last_mean_head_activations(dataset, model_helper, N_TRIALS = 50, shot=4, no_mean=False):

    """
    This function extracts the activation of the last input token.

    Parameters:
    dataset: a iterable item suitable for model_helper.format_func. Essentially a dataloader.
    model_helper:
    N_TRIALS: How many example to average the activation over
    shot: Number of shots per example
    no_mean: Whether you want to take the mean of the examples or save it for other preprocess

    Returns: 
    mean_activations: It has the dimension of (layer, head, Token_len, residual_dim) or (N_TRIALS, layer, head, Token_len, residual_dim). Token_len is set to 1 in this case.
    """

    activation_storage = None

    for n in tqdm(range(N_TRIALS)):

        text, image_list, _, _ = model_helper.format_func(dataset, None, num_shot=shot, model_helper=model_helper)
        inputs = model_helper.insert_image(text, image_list)
        activations_td, result= gather_last_attn_activations(inputs, model_helper)


        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_helper.model_config) for layer in model_helper.model_config['attn_hook_names']]).permute(0,2,1,3)
        ###Extracting only the activation of the last input_token, as seen in the -1 indexing
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)
        if activation_storage is None:
            activation_storage = cur_activation
        else:

            activation_storage = torch.vstack((activation_storage, cur_activation))
    if no_mean:
        return activation_storage
    
    mean_activations = activation_storage.mean(dim=0)
    
    return mean_activations


def reinforce(mean_activations, model_helper, reinforce_data, eval_data):

    """
    This function performs Reinforce to select the attentions that encodes ICL examples.

    Parameters:
    mean_activations: From get_last_mean_head_activations
    model_helper:
    reinforce_data: Dataset used during reinforce optimization
    eval_data: Dataset used for Validation

    Returns: 
    bernoullis: A tensor of bernoullis variable. One variable for each attention heads. Each denote the probability of selecting this attention head.
    """

    num_layer = model_helper.model_config["n_layers"]
    num_heads = model_helper.model_config["n_heads"]
    lr = 0.1
    eps = 1e-3
    epoch = 600

    #(num_layer, num_head)
    bernoullis = [torch.neg(torch.ones(num_heads)).requires_grad_() for _ in range(num_layer)]
    optim = torch.optim.Adam(bernoullis, lr=lr)
    with torch.set_grad_enabled(True):

        for epoch in tqdm(range(epoch)):
            
            loss_list = []
            saved_log_probs = []

            text, image_list, target_out, _ = model_helper.format_func(reinforce_data, None, num_shot=0, model_helper=model_helper)
            new_input = model_helper.insert_image(text, image_list)

            if type(target_out)==list:
                target_out = target_out[0]

            if model_helper.space:
                target_out = " " + target_out

            target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx].unsqueeze(dim=0).to("cuda")
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)


            ###Sampling the distribution many times to reduce variance.
            for _ in range(32):

                ##Current sample
                sampled = prob_dist.sample()
                saved_log_probs.append(prob_dist.log_prob(sampled))

                with torch.no_grad():
                    out_logit = reinforce_activation_replacement(new_input, mean_activations, model_helper, sampled, last_token_only=True)
                    task_loss = torch.nn.functional.cross_entropy(out_logit, target_token)
                    loss_list.append(task_loss)

            #print(model_helper.tokenizer.decode(out_logit[0].argmax(dim=-1)), model_helper.tokenizer.decode(target_token[0]), flush=True)

            policy_loss = []
            loss_list = torch.tensor(loss_list)
            loss_list = (loss_list - loss_list.mean())/(loss_list.std() + eps)

            for log_prob, R in zip(saved_log_probs, loss_list):
                policy_loss.append(log_prob * R)

            optim.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optim.step()
            torch.cuda.empty_cache()
            if epoch % 50 == 0:
                validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch)
    return bernoullis


def validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch, sampled=None):

    with torch.no_grad():
        if sampled is None:
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
            sampled = prob_dist.sample()

        loss_list = []
        for item in eval_data:
            text, image_list, target_out, _ = model_helper.format_func(None, item, num_shot=0, split="test", model_helper=model_helper)
            new_input = model_helper.insert_image(text, image_list)

            if model_helper.space:
                target_out = " " + target_out
            target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx].unsqueeze(dim=0).to("cuda")


            out_logit = reinforce_activation_replacement(new_input, mean_activations, model_helper, sampled, last_token_only=True)
            task_loss = torch.nn.functional.cross_entropy(out_logit, target_token)

            loss_list.append(task_loss)


        print(f"validation loss at {epoch} epoch:", torch.tensor(loss_list).mean())
    return torch.tensor(loss_list).mean().item()


def avg_reinforce(mean_activations, model_helper, reinforce_data, eval_data):

    """
    This function performs Reinforce to select the attentions that encodes ICL examples.
    It computes the average loss over all target tokens instead of the first generated token.

    Parameters:
    mean_activations: From get_last_mean_head_activations
    model_helper:
    reinforce_data: Dataset used during reinforce optimization
    eval_data: Dataset used for Validation

    Returns: 
    bernoullis: A tensor of bernoullis variable. One variable for each attention heads. Each denote the probability of selecting this attention head.
    """

    num_layer = model_helper.model_config["n_layers"]
    num_heads = model_helper.model_config["n_heads"]
    lr = 0.1
    eps = 1e-3
    epoch = 600

    #(num_layer, num_head)
    bernoullis = [torch.neg(torch.ones(num_heads)).requires_grad_() for _ in range(num_layer)]
    optim = torch.optim.Adam(bernoullis, lr=lr)
    with torch.set_grad_enabled(True):

        for epoch in tqdm(range(epoch)):
            
            loss_list = []
            saved_log_probs = []

            text, image_list, target_out, _ = model_helper.format_func(reinforce_data, None, num_shot=0, model_helper=model_helper)

            if type(target_out)==list:
                target_out = target_out[0]


            ###Constructing a label for taking average loss
            if model_helper.space:
                            target_out = " " + target_out
            # target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx].unsqueeze(dim=0).to("cuda")
            input_full = model_helper.insert_image(text, image_list, gt=target_out)
            labels = input_full[0].clone()
            target_len = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0].shape[0]
            labels[:, :-target_len] = -100


            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)

            ###Sampling the distribution many times to reduce variance. Each 
            for _ in range(8):

                ##Current sample
                sampled = prob_dist.sample()
                saved_log_probs.append(prob_dist.log_prob(sampled))

                with torch.no_grad():
                    out= reinforce_activation_replacement(input_full, mean_activations, model_helper, sampled, last_token_only=True, gt=labels, intervention_token=-target_len-1)
                    loss_list.append(out)

            policy_loss = []
            loss_list = torch.tensor(loss_list)
            loss_list = (loss_list - loss_list.mean())/(loss_list.std() + eps)

            for log_prob, R in zip(saved_log_probs, loss_list):
                policy_loss.append(log_prob * R)

            optim.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optim.step()
            torch.cuda.empty_cache()
            if epoch % 50 == 0:
                print(policy_loss.item())
                validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch)
    return bernoullis


def reinforce_activation_replacement(model_input, avg_activations, model_helper, sampled, last_token_only=True, gt=None, intervention_token=None):

    """
    This function performs Reinforce to select the attentions that encodes ICL examples.

    Parameters:
    model_input: Input to the forward function. Refer to model.py
    avg_activations:get_last_mean_head_activations
    model_helper:
    sampeld:
    last_token_only:

    Returns: 
    output: The logit of the first output token
    """

    ###This function returns a list of locations to perform intervention on based on sampled. List((layer, head, token_idx)). Token_idx is default to -1, meaning we always perform intervention on the generated token
    intervention_locations = reinforce_intervention_location(sampled)


    intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                model=model_helper.model, model_config=model_helper.model_config,
                                                batched_input=False, last_token_only=last_token_only, split_idx=model_helper.split_idx, intervention_token=intervention_token)

    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn, retain_grad=True) as td: 
        if gt is None:               
            output = model_helper.forward(model_input, labels=gt).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
        else:
            output = model_helper.forward(model_input, labels=gt).loss

    return output


def reinforce_intervention_location(sampled, categorical=None, token_idx = -1):
    intervention_locations = []
    #(layer, head)

    patch_idx = torch.nonzero(sampled)
    count = 0
    for _ in patch_idx:
        cur_layer = _[0]
        cur_head = _[1]
        intervention_locations.append((cur_layer, cur_head, -1))

    return intervention_locations


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/874d6e93c099d71fe4a2d76551fab233e60062c2/src/utils/intervention_utils.py#L16
def last_replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, model_config, batched_input=False, last_token_only=False, patching=False, replace_layer = 0, split_idx=2, intervention_token=None):

    """
    This function performs intervention on during generation.

    This function defaults to perform intervention during the full generation. To perform intervention on certain token/generation step, modify the function accordingly.
    """


    if patching:
        edit_layers = [replace_layer]
    else:
        edit_layers = [x[0] for x in layer_head_token_pairs]


    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[split_idx])

        token_len = inputs[0].shape[1]
        if current_layer in edit_layers:
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            
            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

            # Patch activations only at the last token for interventions like


            # cloned_inputs = inputs.clone()

            if last_token_only:

                for (layer,head_n, token_n) in layer_head_token_pairs:

                    if layer == current_layer:
   
                        #cloned_inputs[-1,-1,head_n] = avg_activations[layer,head_n,0]
                        inputs[-1,-1,head_n] = avg_activations[layer,head_n,0]

            elif intervention_token is not None:
                for (layer,head_n, token_n) in layer_head_token_pairs:

                    if layer == current_layer:
   
                        #cloned_inputs[-1,intervention_token,head_n] = avg_activations[layer,head_n,0]
                        inputs[-1,intervention_token,head_n] = avg_activations[layer,head_n,0]

            #cloned_inputs = cloned_inputs.view(*original_shape)
            inputs = inputs.view(*original_shape)

            proj_module = get_module(model, layer_name)

            out_proj = proj_module.weight

            #new_output = torch.matmul(cloned_inputs, out_proj.T)
            new_output = torch.matmul(inputs, out_proj.T)

            return new_output
        else:
            return output
    return rep_act


def fv_intervention_natural_text(model_input, model_helper, max_new_tokens=10, return_item="both", intervention_locations=None, avg_activations=None):

    """
    This function is a wrapper of generation intervention
    """

    #Text form to avoid for-loop inside eval loop
    clean_output, intervention_output = "None", "None"

    if return_item == "clean" or return_item == "both":
    
        clean_output = model_helper.generate(model_input, max_new_tokens)


    if return_item == "interv" or return_item == "both":
        
        intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                    model=model_helper.model, model_config=model_helper.model_config,
                                                    batched_input=False, last_token_only=True, split_idx=model_helper.split_idx)
            
        with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn):     
                intervention_output = model_helper.generate(model_input, max_new_tokens)

    return clean_output, intervention_output


def eval_vqa(cur_dataset, results_path, answers):
    ds_collections = {
        'vizwiz_val': {
        'train': '../data/vizwiz/vizwiz_train.jsonl',
        'test': '../data/vizwiz/vizwiz_val.jsonl',
        'question': '../data/vizwiz/vizwiz_val_questions.json',
        'annotation': '../data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
        'okvqa_val': {
            'train': '../data/okvqa/okvqa_train.jsonl',
            'test': '../data/okvqa/okvqa_val.jsonl',
            'question': '../data/okvqa/OpenEnded_mscoco_val2014_questions.json',
            'annotation': '../data/okvqa/mscoco_val2014_annotations.json',
            'metric': 'vqa_score',
            'max_new_tokens': 10,
        }
    }
    if answers is not None:
        result_file = open(results_path, 'w')
        result_file.write(json.dumps(answers))
        result_file.close()


    vqa = VQA(ds_collections[cur_dataset]['annotation'],
                ds_collections[cur_dataset]['question'])
    results = vqa.loadRes(
        resFile=results_path,
        quesFile=ds_collections[cur_dataset]['question'])
    vqa_scorer = VQAEval(vqa, results, n=2)
    vqa_scorer.evaluate()
    print(vqa_scorer.accuracy)