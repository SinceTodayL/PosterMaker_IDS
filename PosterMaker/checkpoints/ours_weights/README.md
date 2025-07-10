---
license: other
license_name: stabilityai-ai-community
license_link: >-
  https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE.md
language:
- en
base_model:
- alimama-creative/SD3-Controlnet-Inpainting
- stabilityai/stable-diffusion-3-medium
pipeline_tag: text-to-image
library_name: diffusers
tags:
- alimama-creative
- stable-diffusion
---


# PosterMaker

![demo images](assets/tesear.png)

PosterMaker is Accepted by CVPR25, please visit [Project page](https://poster-maker.github.io/) to learn more details.



## Model

![pomethodster](assets/method.png)

PosterMaker is an advanced framework for generating promotional product posters with high text rendering and fidelity. Utilizing TextRenderNet for precise character-level text control and SceneGenNet for maintaining product fidelity, PosterMaker excels in creating visually appealing posters. Through a two-stage training strategy to optimize text rendering and background generation separately, PosterMaker outperforms existing methods significantly.

For more technical details, please refer to the [Research paper](https://arxiv.org/abs/2504.06632).

  
### Model Weight

Introduce the model names and weights

| Model Name | Weight Name | Download Link |
| --- | --- | --- |
| TextRenderNet_v1 | textrender_net-0415.pth | [HuggingFace](https://huggingface.co/alimama-creative/PosterMaker/tree/main) |
| SceneGenNet_v1 | scenegen_net-0415.pth | [HuggingFace](https://huggingface.co/alimama-creative/PosterMaker/tree/main) |
| SceneGenNet_v1 with Reward Learning | scenegen_net-rl-0415.pth | [HuggingFace](https://huggingface.co/alimama-creative/PosterMaker//tree/main) |
| TextRenderNet_v2 | textrender_net-1m-0415.pth | [HuggingFace](https://huggingface.co/alimama-creative/PosterMaker/tree/main) |
| SceneGenNet_v2 | scenegen_net-1m-0415.pth | [HuggingFace](https://huggingface.co/alimama-creative/PosterMaker/tree/main) |

**NOTE:** TextRenderNet_v2 is trained with more data for training in the Stage 1, resulting in better text rendering effects. Related details can be found in Section 8 of the Supplementary Materials.


### Known Limitations
The current model exhibits the following known limitations stemming from processing strategies applied to textual elements and captions during constructing our training dataset:

**Text** 
- During training, we restrict texts to 7 lines of up to 16 characters each, and the same applies during inference.
- The training data comes from e-commerce platforms, resulting in relatively simple text colors and font styles with limited design diversity. This leads to similarly simple styles in the inference outputs.


**Layout**
- Only horizontal text boxes are supported (since the amount of vertical text boxes was insufficient, we excluded them from training data)
- Text box must maintain aspect ratios proportional to content length for optimal results (derived from tight bounding box annotations in training)
- No automatic text wrapping within boxes (multi-line text was split into separate boxes during training)

**Prompt Behavior**
- Text content should not be specified in prompts (to match the training setting).
- Limited precise control over text attributes. For poster generation, we expect the model to automatically determine text attributes like fonts and colors. Thus, descriptions about text attributes were intentionally suppressed in training captions.


## Citation
If you find PosterMaker useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{gao2025postermakerhighqualityproductposter,
          title={PosterMaker: Towards High-Quality Product Poster Generation with Accurate Text Rendering}, 
          author={Yifan Gao and Zihang Lin and Chuanbin Liu and Min Zhou and Tiezheng Ge and Bo Zheng and Hongtao Xie},
          year={2025},
          eprint={2504.06632},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2504.06632},
}
```

## LICENSE
The model is based on SD3 finetuning; therefore, the license follows the original [SD3 license](https://huggingface.co/stabilityai/stable-diffusion-3-medium#license).
