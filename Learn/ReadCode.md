## Reading Code 



### pipelines/pipeline_sd3.py

#### class: StableDiffusion3ControlNetPipeline

##### Function: 

* _get_t5_prompt_embeds

* _get_clip_prompt_embeds

这两个函数都是将 `prompt` 转化为向量的函数，

首先，确定设备和数据类型

```Python
device = device or self._execution_device
dtype = dtype or self.text_encoder.dtype
```

然后处理 `prompt` 格式以及确定 `batch_size`

再利用已有模型转换

```Python
 text_inputs = self.tokenizer_3(
      prompt,
      padding="max_length",
      max_length=self.tokenizer_max_length,
      truncation=True,
      add_special_tokens=True,
      return_tensors="pt",
)
```

如果发现长度太长（超过给定值），就打log

最后按照这个prompt要生成一张图片，就复制几遍这个嵌入向量

```Python
prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
prompt_embeds = prompt_embeds.view(
      batch_size * num_images_per_prompt, seq_len, -1
)
```



* encode_prompt

这个函数利用上面两个函数，正式将 `prompt` 转化为向量

Stable Diffusion 中，会同时使用 `CLIP-1, CLIP-2, T5` 这几个模型，然后将结果融合在一起，作为向量输入

这样能综合使用他们的优点



* check_inputs

属于前置输入校验层，检查文本输入、图像尺寸的一致性、合法性



* prepare_latents

首先计算 `latent vector` 的形状

然后用高斯分布随机取样一个该形状的向量，这个向量就是整个模型的起点，是整个扩散模型一开始的随机向量



* prepare_image
* prepare_image_with_mask
* prepare_image_vae_cond

图像预处理，包括扩展数量（根据生成图片数量）、编码图像（送进 VAE 得到 latent 表示）、检查准备掩码

其中 `prepare_image_vae_cond` 不拼接 mask，只是将 image 转换到 latent 空间、扩展数量等

但是 `prepare_image_with_mask` 会 concat，通道维度的拼接，即通道数 +1

```Python
control_image = torch.cat([image_latents, mask], dim=1)
```

mask 可以进行空间引导，用0/1表示，比如可以控制模型在哪一块不要动



* ==\_\_call\_\_==

这个是函数从 `prompt` 生成图片的全流程

函数中分步骤写出了整个流程，需要仔细阅读！