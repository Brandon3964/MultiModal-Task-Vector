# MTV ⚡⚡
[NeurIPS 2024] Official Code for the Paper "Multimodal Task Vectors Enable Many-Shot  Multimodal In-Context Learning"
---
We present **Multimodal Task Vector(MTV)**, a novel technique for compressing many-shot multimodal in-context examples. We find that this approach not only outperform vanille ICL for Large Multimodal Language Models but also require significantly less time and memory. 

More details can be found in our [paper](https://arxiv.org/abs/2406.15334).

<p align="center">
  <img src=MTV/data/teaser.png width="500"/>
</p>

### Method Description
---
Our method consists of three steps. The first step performs some amount of forward pass on ICL examples and take the average activations of these forward pass. The second step consists of running REINFORCE to locate the attention heads in the Language Backbone that capture the given multimodal tasks. During zero-shot inference, intervention is performed on the selected attention heads to replace the current activations with the average activations, in which we called the Multimodal Task Vector.
<p align="center">
  <img src=MTV/data/method.png height="600"/>
</p>

### 💻 Setup
---

#### Datasets
For Vizwiz and OKVQA, please follow the instruction in the Qwen-VL repository. For Flower, CUB, and DTD, please download the images from their respective official websites. We provide the 2-way 1 shot text annotations in the data file.

#### Models
1. For the models used in the paper, please follow the installation steps outlined in their official repository.
2. Install this package by David Bau at Northeastern University.
```bash
git+https://github.com/davidbau/baukit@main#egg=baukit
```
3. Install `flash-attention`
Please refer to models.py if you would like to use custom models.

### 📝 Citation
---
If you found our work useful, please consider starring and citing. Thank you!
```latex
@article{huang2024multimodal,
  title={Multimodal Task Vectors Enable Many-Shot Multimodal In-Context Learning},
  author={Huang, Brandon and Mitra, Chancharik and Arbelle, Assaf and Karlinsky, Leonid and Darrell, Trevor and Herzig, Roei},
  booktitle={Advances in neural information processing systems (NeurIPS)},
  year={2024}
}
```
