# %%
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import RobertaTokenizer, ClapTextModelWithProjection
import torch

# %%
model = "cvssp/audioldm"
device = "cuda"

# %%
vae = AutoencoderKL.from_pretrained(model, subfolder="vae").to(device)
tokenizer = RobertaTokenizer.from_pretrained(model, subfolder="tokenizer")
text_encoder = ClapTextModelWithProjection.from_pretrained(model, subfolder="text_encoder").to(device)
# unet = UNet2DConditionModel(
#     sample_size=128,
#     in_channels=8,
#     out_channels=8,
#     center_input_sample=False,
#     flip_sin_to_cos=True,
#     freq_shift=0,
#     cross_attention_dim=768,
#     block_out_channels=[128, 256, 384, 640],
#     class_embeddings_concat=True,
#     projection_class_embeddings_input_dim=512,
#     time_embedding_type="positional",
#     down_block_types=["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
#     attention_head_dim=8,
#     act_fn="silu",
#     #class_embed_type="simple_projection",
#     num_class_embeds=None,
# ).to(device)
unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet").to(device)
scheduler = DDPMScheduler.from_pretrained(model, subfolder="scheduler")

# %%
prompt = ["a photograph of an astronaut riding a horse"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise
batch_size = 1  # Batch size for inference

# %%
text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(device)).last_hidden_state

# %%
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(device)).last_hidden_state

# %%
# print the shape of the embeddigns
print(text_embeddings.shape)
print(uncond_embeddings.shape)
text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=1)

# %%
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(device)

# %%
latents = latents * scheduler.init_noise_sigma

# %%
from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        #prints the dimensions of text embeddings
        print(text_embeddings.shape)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# %%
