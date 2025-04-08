#### 
import json
import random

vizwiz_prompt = """First carefully understand the given examples. 
Then use the given image and answer the question in the same way as the examples. 
If the question can not be answered, respond unanswerable. """

wheat_head_prompt = """# Introduction
The dataset focuses on detecting wheat heads in agricultural images to assist with monitoring and analysis. It contains annotations for the following class:

- **Wheat Heads**: Wheat heads are the parts of the wheat plant where the grains are formed.

# Object Classes

## Wheat Heads

### Description
The wheat head is part of the wheat plant, distinguished by its elongated shape and spikelets containing grains. They may appear amidst leaves and stems, often seen in dense clusters or isolated. Wheat heads may be green or yellow in color depending on their moisture content.

### Instructions
- Annotate the entire wheat head, capturing the full elongated shape from the base near the stem to the tip, even if partially obscured by other parts of the plant.
- Ensure the bounding box includes the entire visible head while excluding leaves and stems unless they are integral to the shape.
- Do not annotate if the head is less than 20% visible, or if it is unclear whether the object is a wheat head.
- Avoid annotating overly blurred objects that cannot be confidently identified as wheat heads.
- Cross-referencing other instances can help disambiguate partially visible wheat heads."""

wood_defects_prompt = """# Introduction
The purpose of this dataset is to help localize and classify different defects that are often found on wood planks used in manufacturing, or other wood products. Wood will often have knots, cracks, or holes in it, all of which inform the quality and condition of the wood. 

- **Crack**: A crack in the wood.
- **Holes**: A hole in the wood.
- **Knot with Crack**: A knot with a crack in it.
- **Dead Knot**: A darker colored knot.
- **Live Knot**: A lighter colored knot.

# Object Classes
## Crack
### Description
This class contains cracks in the wood. It specifically does not contain cracks that occur in knots.

### Instructions
- Draw a bounding box around the entire crack. If a crack has several clearly distinct sections to it, draw boxes around those individually. Try to make the box as tight around the crack as possible. 
- Cracks will often appear as dark lines in the wood that do not necessarily follow the wood grain. Do not tag cracks that appear in knots, as those belong to the knot with crack category. 
- If a crack extends a long way out of a knot, more than the diameter of the knot, draw a box around the part of the crack not in the knot.

## Holes
### Description
This class contains holes in wood. The two major kinds of holes are: the hole a termite or nail may leave behind, small and concentrated; or a hole that has the same dark color as a crack but just doesn't go anywhere, instead having a circular shape. These holes can be very small, so look carefully!

### Instructions
- Draw a tight bounding box around the hole. If there are clusters of holes, mark each individually. 
- Make sure not to tag holes that appear in knots.

## Knot with Crack
### Description
This is a knot that has a crack in it. It doesn't matter if the knot is dead or alive; if it has a crack in it, it's a knot with crack.

### Instructions
- Draw a tight bounding box around the knot that has the crack in it. Most of the time, the crack will be contained entirely within the knot. 
- However, if the crack extends outside of the knot, do not enlarge the bounding box to contain the rest of the crack. The important thing is that the box is tight around the knot.

## Dead Knot
### Description
This knot belonged to a branch that died before the tree was cut down. Dead knots typically have strong discoloration around at least some part of their boundary, indicating that its growth with the tree was poor. Often this discoloration is in the form of a very dark ring around the knot. Other times, there is just a sudden and sharp contrast with the color of the rest of the wood, that isn't necessarily shaped as a ring. The discoloration is sharp and sudden. It can look like the boundary of rot.

### Instructions
- Draw a tight bounding box around the knot. If there is clear discoloration, ensure that it is contained within the bounding box. 
- For example, if the knot has a dark ring, ensure that the ring is entirely contained within the box.

## Live Knot
### Description
This a knot that belonged to a living branch. These knots may be a different color from the rest of the wood, but the boundary between them and the wood is typically softer than it is for a dead knot. The knot looks like it is part of the wood, solidly attached all the way around. If the wood has grain around the knot, the grain moves smoothly around the knot. These knots can have rings, but their boundaries are soft.

### Instructions
- Draw a bounding box around the knot. Some knots are clear as to where they end and the rest of the wood begins. For those knots, draw a tight bounding box. 
- Other knots blend into the general grain of the wood. They will, however, always have a circular region where the branch grew. For those kinds of knots, do your best to draw the bounding box around the circular region and not around the rest of the warp of the grain."""

x_ray_prompt="""# Introduction
This dataset provides X-ray images of human hands for medical studies. It aims to segment and identify key anatomical features in hand radiographs. The classes are:
- **DIP**: Distal Interphalangeal joint, the farthest joint in the fingers.
- **MCP**: Metacarpophalangeal joint, located at the base of each finger.
- **PIP**: Proximal Interphalangeal joint, the middle joint in the fingers.
- **Radius**: One of the two main bones in the forearm, on the thumb side.
- **Ulna**: The second forearm bone, located on the pinky side.
- **Wrist**: Carpal bones at the base of the hand connecting to the forearm.

# Object Classes

## DIP
### Description
The DIP (or Distal Interphalangeal) joint is located near the tip of the finger, between the last two phalanges. It appears as a small gap representing the joint space on the X-ray.

### Instructions
- Focus on the end joints of each digit.
- Ensure the bounding box encapsulates the visible joint space of the DIP.
- Avoid labeling any part of the bone shafts; the focus is on the joint itself.

## MCP
### Description
The MCP (or Metacarpophalangeal) joint is found at the knuckle, connecting fingers to the palm. It appears as a prominent gap near the base of each finger.

### Instructions
- Identify the knuckle joints visible where the fingers meet the palm.
- Draw the bounding box around the joint area, including any visible joint space.
- Exclude the phalange or metacarpal bone sections; only annotate the joint gap.

## PIP
### Description
The PIP (or Proximal Interphalangeal) joint is located between the first and second phalanges, appearing as a clear gap between the finger bones. It is below the DIP and above the MCP.

### Instructions
- Locate the middle joint in each finger.
- Capture the visible gap between the bones, indicative of the PIP joint.
- Do not extend the annotation to cover the phalanges themselves.

## Radius
### Description
The Radius is the thicker and shorter of the two forearm bones, located on the thumb side. It appears as a robust and continuous bone on the X-ray.

### Instructions
- Identify the forearm bone on the thumb side, distinguishable by its thicker dimensions.
- Cover the entire visible section of the Radius, from wrist to near the elbow.
- Ensure surrounding soft tissues or overlapping bones are not included.

## Ulna
### Description
The Ulna runs parallel to the Radius and is typically longer and thinner, extending to the elbow.

### Instructions
- Locate the forearm bone on the side opposite the thumb.
- Draw the bounding box to include the full visible length of the Ulna.
- Distinguish it from the Radius to avoid overlap in the annotation.

## Wrist
### Description
The Wrist comprises multiple small carpal bones at the base of the hand, forming the connection to the forearm.

### Instructions
- Focus on the cluster of bones at the base of the palm.
- Surround all visible carpal bones in the bounding box.
- Avoid marking the Radius or Ulna sections extending into the hand.
"""
####

def open_data(dataset_name, path):

    jsonl_format_dataset = ["vizwiz", "okvqa", "natural_ret", "robovl"]
    list_format_dataset = ["flower", "cub", "dtd"]

    with open(path, 'r') as json_file:
        if dataset_name in jsonl_format_dataset:
            dataset = [json.loads(json_string) for json_string in list(json_file)]
        elif dataset_name in list_format_dataset:
            dataset = json.load(json_file)
    return dataset


### Each format function should return (full_text, image_list, answer, question_id)
def get_format_func(cur_dataset):

    if cur_dataset == "vizwiz":
        return format_vizwiz
    if cur_dataset == "natural_ret":
        return format_natural_ret
    if cur_dataset == "robovl":
        return format_robovl
    if cur_dataset == "okvqa":
        return format_okvqa
    if cur_dataset == "flower":
        return format_flower
    if cur_dataset == "cub":
        return format_cub
    if cur_dataset == "dtd":
        return format_dtd

def natural_ret_balance(all_data, num_shot):
    yes_samples = []
    no_samples = []

    # sampled = random.sample(all_data, 20)  # Sample more than needed to ensure we find enough of each
    random.shuffle(all_data)
    for item in all_data:
        
        # Break early if we have enough samples
        if len(yes_samples) == num_shot and len(no_samples) == num_shot:
            break
        elif len(yes_samples) == num_shot:
            if item['label'] == 'No':
                no_samples.append(item)
        elif len(no_samples) == num_shot:
            if item['label'] == 'Yes':
                yes_samples.append(item)
        else:
            if item['label'] == 'Yes':
                yes_samples.append(item)
            if item['label'] == 'No':
                no_samples.append(item)

    total_samples = yes_samples + no_samples
    random.shuffle(total_samples)
    return total_samples

def format_natural_ret(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = '{} Answer with Yes or No.'
    image_list = []
    
    if cur_item is None:
        data = random.sample(all_data, 1)[0]
    else:
        data = cur_item
    image = data['image']
    question = data['question']
    label = data['label']
    question_id = data['question_id']
    
    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = natural_ret_balance(all_data, num_shot)
        for sample in sampled_data:
            few_shot_prompt += prompt.format(sample['question']) + f" {sample['label']}\n"
            image_list.append(sample["image"])
    
    full_text = few_shot_prompt + prompt.format(question)
    
    image_list.append(image)
    
    return full_text, image_list, label, question_id
    

def format_robovl(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    # with open()
    prompt = wood_defects_prompt+ '\n{}'
    image_list = []
    
    if cur_item is None:
        data = random.sample(all_data, 1)[0]
    else:
        data = cur_item
    image = data['image']
    question = data['question']
    label = data['label']
    question_id = data['question_id']
    
    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            few_shot_prompt += prompt.format(sample['question']) + f" {sample['label']}\n"
            image_list.append(sample["image"])
    
    full_text = few_shot_prompt + prompt.format(question)

    # print(f'Full Text {full_text}')
    
    image_list.append(image)
    
    return full_text, image_list, label, question_id


# def format_nuimages(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
#     prompt = '<image>\n{}' # Experiment with different prompt structures here.
#     image_list = []
    
#     if cur_item is None:
#         data = random.sample(all_data, 1)[0]
#     else:
#         data = cur_item
        
#     image = data['image']
#     question = data['question']
#     label = data['label']
#     question_id = data['question_id']
    
#     few_shot_prompt = ''
#     if num_shot > 0:
#         sampled_data = natural_ret_balance(all_data)
#         for sample in sampled_data:
#             few_shot_prompt += prompt.format(sample['question']) + f" {sample['label']}\n"
#             image_list.append(sample["image"])
    
#     full_text = few_shot_prompt + prompt.format(question)
    
#     image_list.append(image)
    
#     return full_text, image_list, label, question_id
####All return format will be in the form (Text, list of images, Answer, Question_id)
def format_vizwiz(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = '<image>{} Answer:'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']

    few_shot_prompt = ''
    if num_shot > 0:

        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(sample['question']) + f" {sample['answer']}"
            image_list.append("../" + sample["image"])


    full_text = vizwiz_prompt + few_shot_prompt + prompt.format(question)
    image_list.append("../" + image)

    return full_text, image_list, answer, question_id


def format_okvqa(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = '<image>{} Answer:'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(sample['question']) + f"{sample['answer']}"
            image_list.append(sample["image"])

    
    full_text = few_shot_prompt + prompt.format(question)
    image_list.append(image)

    return full_text, image_list, answer, question_id


def format_flower(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1


def format_cub(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1
    

def format_dtd(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1