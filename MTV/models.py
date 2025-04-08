from mtv_utils import *
from preprocess import *
from PIL import Image
import torch
import copy

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token
from qwen_vl_utils import process_vision_info

def load_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
    except:
        return image_file
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out



class ModelHelper:
    def __init__(self):

        """
        self.model:The loaded model
        self.tokenizer: The loaded tokenizer
        self.model_config: The architecture of the model. Might need to do print(model) see how to initialize
        self.format_func: The format function for the current dataset
        self.space: Whether the model output will have a leading space
        self.cur_dataset: Name of the current dataset
        self.split_idx: The index of "layer" when you parse "attn_hook_names" with "."
        self.nonspecial_idx: The index in which the generated tokens are not special token. Used to skip special token and construct the current target output for loss calculation.
        """


    #Always return a single variable. If both text and image is returned, return in tuple
    def insert_image(self, text, image_list):

        """
        Returns an object that is the input to forward and generate.
        """
        pass
    #Takes the output of insert_image
    def forward(self, model_input, labels=None):

        """
        Forwrad function wrapper
        """

        pass
    #Takes the output of insert image
    def generate(self, model_input, max_new_tokens):

        """
        Generate function wrapper
        """
        pass


class llavaOVHelper(ModelHelper):

    def __init__(self, model, tokenizer, processor, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_config = {"n_heads":model.model.config.num_attention_heads,
                    "n_layers":model.model.config.num_hidden_layers,
                    "resid_dim":model.model.config.hidden_size,
                    "name_or_path":model.model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.model.config.num_hidden_layers)],
                    "mlp_hook_names": [f'model.layers.{layer}.mlp.down_proj' for layer in range(model.model.config.num_hidden_layers)]}
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.nonspecial_idx = 0
        self.modalities = None


    def insert_image(self, text, image_list, gt=None):

        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        if gt is not None:
            prompt_question = prompt_question + gt

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")

        if image_list == []:
            return (input_ids, None, None)
        
        path = image_list[0]

        if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')): 
            self.modalities = ["video"]
        
        image_list = load_images(image_list)
        image_sizes = [image.size for image in image_list]

        image_tensors = process_images(image_list, self.processor, self.model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=self.model.device) for _image in image_tensors]


        return (input_ids, image_tensors, image_sizes)
    

    def forward(self, model_input, labels=None):

        result = self.model(model_input[0],
            images=model_input[1],
            image_sizes=model_input[2],
            labels=labels) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    

    def generate(self, model_input, max_new_tokens):

        cont = self.model.generate(
            model_input[0],
            images=model_input[1],
            image_sizes=model_input[2],
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
        )
        
        return self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    def vqascore(self, model_input, answer='Yes'):
        with torch.inference_mode():
            outputs = self.model.generate(
                model_input[0],
                images=model_input[1],
                image_sizes=model_input[2],
                do_sample=False,
                temperature=0,
                max_new_tokens=1,
                modalities=self.modalities,
                output_scores=True,
                return_dict_in_generate=True
            )
        scores = outputs.scores[0]
        probs = torch.nn.functional.softmax(scores, dim=-1)
        yes_token_id = self.processor.tokenizer.encode(answer)[0]
        lm_prob = probs[0, yes_token_id].item()
        
        return lm_prob


class QwenHelper(ModelHelper):

    def __init__(self, model, tokenizer, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = {"n_heads":model.transformer.config.num_attention_heads,
                    "n_layers":model.transformer.config.num_hidden_layers,
                    "resid_dim":model.transformer.config.hidden_size,
                    "name_or_path":model.transformer.config._name_or_path,
                    "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.transformer.config.num_hidden_layers)],
                    "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.transformer.config.num_hidden_layers)]}
        self.format_func = get_format_func(cur_dataset)
        self.space = True
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.nonspecial_idx = 0
        self.question_lookup = None

    def insert_image(self, text, image_list):

        text = text.replace("<image>", "<img></img>")
        text = text.split("</img>")

        new_text = ""
        for text_split, image in zip(text[:-1], image_list):
            new_text += f"{text_split}{image}</img>"
        return self.tokenizer(new_text + text[-1], return_tensors='pt', padding='longest')
    

    def forward(self, model_input, labels=None):

        result = self.model(input_ids=model_input["input_ids"].to(self.model.device),
                attention_mask=model_input["attention_mask"].to(self.model.device)) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    

    def generate(self, model_input, max_new_tokens):

        generated_output = self.model.generate(
                input_ids=model_input["input_ids"].to(self.model.device),
                attention_mask=model_input["attention_mask"].to(self.model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,)
        
        return self.tokenizer.batch_decode(generated_output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()
    
    
class ViLAHelper(ModelHelper):

    def __init__(self, model, tokenizer, image_processor, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = {"n_heads":model.llm.model.config.num_attention_heads,
                        "n_layers":model.llm.model.config.num_hidden_layers,
                        "resid_dim":model.llm.model.config.hidden_size,
                        "name_or_path":model.llm.model.config._name_or_path,
                        "attn_hook_names":[f'llm.model.layers.{layer}.self_attn.o_proj' for layer in range(model.llm.model.config.num_hidden_layers)],
                        "layer_hook_names":[f'llm.model.layers.{layer}' for layer in range(model.llm.model.config.num_hidden_layers)]}
    
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 0
        self.question_lookup = None


    ##No need to change the image token since it's the same as default
    def insert_image(self, text, image_list):

        text = text.replace("<image>", "<image>\n")

        if image_list is not None:
            images = load_images(image_list)
            images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        else:
            images_tensor = None

        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
            

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        return (input_ids, images_tensor, stopping_criteria, stop_str)
    

    def forward(self, model_input, labels=None): 
    
        result = self.model(model_input[0], images=[model_input[1]], labels=labels)
        return result
    

    def generate(self, model_input, max_new_tokens):

        if model_input[1] is None:
            output = self.model.generate(
                model_input[0],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                use_cache=True,
                stopping_criteria=[model_input[2]])    
        else:

            output = self.model.generate(
                    model_input[0],
                    images=[
                        model_input[1],
                    ],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    min_new_tokens=1,
                    use_cache=True,
                    stopping_criteria=[model_input[2]])
    
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        output = output.strip()
        if output.endswith(model_input[3]):
            output = output[: -len(model_input[3])]
        output = output.strip()
        return output 
    

class Idefics2Helper(ModelHelper):

    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_config = {"n_heads":model.model.text_model.config.num_attention_heads,
                        "n_layers":model.model.text_model.config.num_hidden_layers,
                        "resid_dim":model.model.text_model.config.hidden_size,
                        
                        "name_or_path":model.model.text_model.config._name_or_path,
                        "attn_hook_names":[f'model.text_model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.text_model.config.num_hidden_layers)],
                        "layer_hook_names":[f'model.text_model.layers.{layer}' for layer in range(model.model.text_model.config.num_hidden_layers)]}
    
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 1


    def insert_image(self, text, image_list):

        opened_images = load_images(image_list)
        inputs = self.processor(text=[text], images=[opened_images], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs


    def forward(self, model_input, labels=None):
        result = self.model(**model_input)
        return result
    

    def generate(self, model_input, max_new_tokens):

        with torch.inference_mode():
            output = self.model.generate(
                    **model_input,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,)
        
        output = self.processor.batch_decode(output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()
        return output
    
class Qwen2Helper(ModelHelper):
    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_config = {"n_heads":model.model.config.num_attention_heads,
                    "n_layers":model.model.config.num_hidden_layers,
                    "resid_dim":model.model.config.hidden_size,
                    "name_or_path":model.model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.model.config.num_hidden_layers)],
                    "mlp_hook_names": [f'model.layers.{layer}.mlp.down_proj' for layer in range(model.model.config.num_hidden_layers)]}
        self.format_func = get_format_func(cur_dataset)
        self.cur_dataset = cur_dataset
        self.nonspecial_idx = 0
        self.split_idx = 2
        self.question_lookup = None

        self.space = False

        self.all_heads = []
        for layer in range(28):
            for head in range(28):
                self.all_heads.append((layer, head, -1))

    def insert_image(self, text, image_list, gt=None):

        if gt is not None:
            text = text + gt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in image_list
                ] + [
                    {"type": "text", "text": text}
                ]
            }
        ]


        formatted_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[formatted_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # print(f'input_height {inputs["image_grid_thw"][0][1]*14}')
        # print(f'input_width {inputs["image_grid_thw"][0][2]*14}')
        return inputs

    def forward(self, model_input, labels=None):

        # input_ids = model_input["input_ids"].to(self.model.device)
        # attention_mask = model_input["attention_mask"].to(self.model.device)

        result = self.model(
            **model_input
        )  

        return result

    def generate(self, model_input, max_new_tokens):

    
        generated_output = self.model.generate(
            **model_input, max_new_tokens=max_new_tokens, do_sample=False
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_input.input_ids, generated_output)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    def vqascore(self, model_input, answer='Yes'):
        with torch.inference_mode():
            outputs = self.model.generate(
                **model_input,
                max_new_tokens=1,
                do_sample=False, # Odd that greedy decoding seems necessary for some reason to get the logprobs
                output_scores=True,
                return_dict_in_generate=True
            )
        scores = outputs.scores[0]

        probs = torch.nn.functional.softmax(scores, dim=-1)
        yes_token_id = self.processor.tokenizer.encode(answer)[0]
        lm_prob = probs[0, yes_token_id].item()
        
        return lm_prob





