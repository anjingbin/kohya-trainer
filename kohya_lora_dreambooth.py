# -*- coding: utf-8 -*-

"""# V. Training Model


"""

import os

accelerate_config = "/Users/anjingbin/ai-ex/kohya-trainer/accelerate_config/config.yaml"

from accelerate.utils import write_basic_config
if not os.path.exists(accelerate_config):
  write_basic_config(save_location=accelerate_config)

        
v2 = False #@param {type:"boolean"}
v_parameterization = False #@param {type:"boolean"}
project_name = "jiangshuying" #@param {type:"string"}
pretrained_model_name_or_path = "/Users/anjingbin/ai-ex/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.ckpt" #@param {type:"string"}
vae = ""  #@param {type:"string"}
#@markdown You need to register parent folder and not where `train_data_dir` located
train_folder_directory = "/Users/anjingbin/ai-ex/kohya-trainer/loratrain/image" #@param {'type':'string'}
# %store train_folder_directory
reg_folder_directory = "/Users/anjingbin/ai-ex/kohya-trainer/loratrain/image" #@param {'type':'string'}
# %store reg_folder_directory
output_dir = "/Users/anjingbin/ai-ex/kohya-trainer/loratrain/output" #@param {'type':'string'}
resume_path =""
inference_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/"


os.makedirs(output_dir, exist_ok=True)

# Commented out IPython magic to ensure Python compatibility.
#@title ## 5.2. Define Specific LoRA Training Parameters
# %store -r

#@markdown ## LoRA - Low Rank Adaptation Dreambooth

#@markdown Some people recommend setting the `network_dim` to a higher value.
network_dim = 128 #@param {'type':'number'}
#@markdown For weight scaling in LoRA, it is better to set `network_alpha` the same as `network_dim` unless you know what you're doing. A lower `network_alpha` requires a higher learning rate. For example, if `network_alpha = 1`, then `unet_lr = 1e-3`.
network_alpha = 128 #@param {'type':'number'}
network_module = "networks.lora"

#@markdown `network_weights` can be specified to resume training.
network_weights = "" #@param {'type':'string'}

#@markdown By default, both Text Encoder and U-Net LoRA modules are enabled. Use `network_train_on` to specify which module to train.
network_train_on = "both" #@param ['both','unet_only', 'text_encoder_only'] {'type':'string'}

#@markdown It is recommended to set the `text_encoder_lr` to a lower learning rate, such as `5e-5`, or to set `text_encoder_lr = 1/2 * unet_lr`.
learning_rate = 1e-4 #@param {'type':'number'}
unet_lr = 0 #@param {'type':'number'}
text_encoder_lr = 5e-5 #@param {'type':'number'}
lr_scheduler = "constant" #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"] {allow-input: false}

#@markdown If `lr_scheduler = cosine_with_restarts`, update `lr_scheduler_num_cycles`.
lr_scheduler_num_cycles = 1 #@param {'type':'number'}
#@markdown If `lr_scheduler = polynomial`, update `lr_scheduler_power`.
lr_scheduler_power = 1 #@param {'type':'number'}

#@markdown Check the box to not save metadata in the output model.
no_metadata = False #@param {type:"boolean"}
training_comment = "this comment will be stored in the metadata" #@param {'type':'string'}

print("Loading network module:", network_module)
print(f"{network_module} dim set to:", network_dim)
print(f"{network_module} alpha set to:", network_alpha)

if network_weights == "":
  print("No LoRA weight loaded.")
else:
  if os.path.exists(network_weights):
    print("Loading LoRA weight:", network_weights)
  else:
    print(f"{network_weights} does not exist.")
    network_weights =""

if network_train_on == "unet_only":
  print("Enabling LoRA for U-Net.")
  print("Disabling LoRA for Text Encoder.")

print("Global learning rate: ", learning_rate)

if network_train_on == "unet_only":
  print("Enable LoRA for U-Net")
  print("Disable LoRA for Text Encoder")
  print("UNet learning rate: ", unet_lr) if unet_lr != 0 else ""
if network_train_on == "text_encoder_only":
  print("Disabling LoRA for U-Net")
  print("Enabling LoRA for Text Encoder")
  print("Text encoder learning rate: ", text_encoder_lr) if text_encoder_lr != 0 else ""
else:
  print("Enabling LoRA for U-Net")
  print("Enabling LoRA for Text Encoder")
  print("UNet learning rate: ", unet_lr) if unet_lr != 0 else ""
  print("Text encoder learning rate: ", text_encoder_lr) if text_encoder_lr != 0 else ""

print("Learning rate Scheduler:", lr_scheduler)

if lr_scheduler == "cosine_with_restarts":
  print("- Number of cycles: ", lr_scheduler_num_cycles)
elif lr_scheduler == "polynomial":
  print("- Power: ", lr_scheduler_power)

# Printing the training comment if metadata is not disabled and a comment is present
if not no_metadata:
  if training_comment: 
    training_comment = training_comment.replace(" ", "_")
    print("Training comment:", training_comment)
else:
  print("Metadata won't be saved")

# Commented out IPython magic to ensure Python compatibility.
from prettytable import PrettyTable
import textwrap
import yaml

# %store -r

#@title ## 5.3. Start LoRA Dreambooth
#@markdown ### Define Parameter

train_batch_size = 6 #@param {type:"number"}
num_epochs = 20 #@param {type:"number"}
caption_extension = '.txt' #@param {'type':'string'}
mixed_precision = "fp16" #@param ["no","fp16","bf16"] {allow-input: false}
save_precision = "fp16" #@param ["float", "fp16", "bf16"] {allow-input: false}
save_n_epochs_type = "save_n_epoch_ratio" #@param ["save_every_n_epochs", "save_n_epoch_ratio"] {allow-input: false}
save_n_epochs_type_value = 3 #@param {type:"number"}
save_model_as = "safetensors" #@param ["ckpt", "pt", "safetensors"] {allow-input: false}
resolution = 512 #@param {type:"slider", min:512, max:1024, step:128}
enable_bucket = True #@param {type:"boolean"}
min_bucket_reso = 320 if resolution > 640 else 256
max_bucket_reso = 1280 if resolution > 640 else 1024
cache_latents = True #@param {type:"boolean"}
max_token_length = 225 #@param {type:"number"}
clip_skip = 2 #@param {type:"number"}
use_8bit_adam = True #@param {type:"boolean"}
gradient_checkpointing = False #@param {type:"boolean"}
gradient_accumulation_steps = 1 #@param {type:"number"}
seed = 0 #@param {type:"number"}
logging_dir = "/Users/anjingbin/ai-ex/kohya-trainer/loratrain/logs"
log_prefix = project_name
additional_argument = "--shuffle_caption --xformers" #@param {type:"string"}
print_hyperparameter = True #@param {type:"boolean"}
prior_loss_weight = 1.0
# %cd {repo_dir}

train_command=f"""
accelerate launch --config_file={accelerate_config} --num_cpu_threads_per_process=8 train_network.py \
  {"--v2" if v2 else ""} \
  {"--v_parameterization" if v2 and v_parameterization else ""} \
  --network_dim={network_dim} \
  --network_alpha={network_alpha} \
  --network_module={network_module} \
  {"--network_weights=" + network_weights if network_weights else ""} \
  {"--network_train_unet_only" if network_train_on == "unet_only" else ""} \
  {"--network_train_text_encoder_only" if network_train_on == "text_encoder_only" else ""} \
  --learning_rate={learning_rate} \
  {"--unet_lr=" + format(unet_lr) if unet_lr !=0 else ""} \
  {"--text_encoder_lr=" + format(text_encoder_lr) if text_encoder_lr !=0 else ""} \
  {"--no_metadata" if no_metadata else ""} \
  {"--training_comment=" + training_comment if training_comment and not no_metadata else ""} \
  --lr_scheduler={lr_scheduler} \
  {"--lr_scheduler_num_cycles=" + format(lr_scheduler_num_cycles) if lr_scheduler == "cosine_with_restarts" else ""} \
  {"--lr_scheduler_power=" + format(lr_scheduler_power) if lr_scheduler == "polynomial" else ""} \
  --pretrained_model_name_or_path={pretrained_model_name_or_path} \
  {"--vae=" + vae if vae else ""} \
  {"--caption_extension=" + caption_extension if caption_extension else ""} \
  --train_data_dir={train_folder_directory} \
  --reg_data_dir={reg_folder_directory} \
  --output_dir={output_dir} \
  --prior_loss_weight={prior_loss_weight} \
  {"--resume=" + resume_path if resume_path else ""} \
  {"--output_name=" + project_name if project_name else ""} \
  --mixed_precision={mixed_precision} \
  --save_precision={save_precision} \
  {"--save_every_n_epochs=" + format(save_n_epochs_type_value) if save_n_epochs_type=="save_every_n_epochs" else ""} \
  {"--save_n_epoch_ratio=" + format(save_n_epochs_type_value) if save_n_epochs_type=="save_n_epoch_ratio" else ""} \
  --save_model_as={save_model_as} \
  --resolution={resolution} \
  {"--enable_bucket" if enable_bucket else ""} \
  {"--min_bucket_reso=" + format(min_bucket_reso) if enable_bucket else ""} \
  {"--max_bucket_reso=" + format(max_bucket_reso) if enable_bucket else ""} \
  {"--cache_latents" if cache_latents else ""} \
  --train_batch_size={train_batch_size} \
  --max_token_length={max_token_length} \
  {"--use_8bit_adam" if use_8bit_adam else ""} \
  --max_train_epochs={num_epochs} \
  {"--seed=" + format(seed) if seed > 0 else ""} \
  {"--gradient_checkpointing" if gradient_checkpointing else ""} \
  {"--gradient_accumulation_steps=" + format(gradient_accumulation_steps) } \
  {"--clip_skip=" + format(clip_skip) if v2 == False else ""} \
  --logging_dir={logging_dir} \
  --log_prefix={log_prefix} \
  {additional_argument}
  """

debug_params = ["v2", \
                "v_parameterization", \
                "network_dim", \
                "network_alpha", \
                "network_module", \
                "network_weights", \
                "network_train_on", \
                "learning_rate", \
                "unet_lr", \
                "text_encoder_lr", \
                "no_metadata", \
                "training_comment", \
                "lr_scheduler", \
                "lr_scheduler_num_cycles", \
                "lr_scheduler_power", \
                "pretrained_model_name_or_path", \
                "vae", \
                "caption_extension", \
                "train_folder_directory", \
                "reg_folder_directory", \
                "output_dir", \
                "prior_loss_weight", \
                "resume_path", \
                "project_name", \
                "mixed_precision", \
                "save_precision", \
                "save_n_epochs_type", \
                "save_n_epochs_type_value", \
                "save_model_as", \
                "resolution", \
                "enable_bucket", \
                "min_bucket_reso", \
                "max_bucket_reso", \
                "cache_latents", \
                "train_batch_size", \
                "max_token_length", \
                "use_8bit_adam", \
                "num_epochs", \
                "seed", \
                "gradient_checkpointing", \
                "gradient_accumulation_steps", \
                "clip_skip", \
                "logging_dir", \
                "log_prefix", \
                "additional_argument"]

if print_hyperparameter:
    table = PrettyTable()
    table.field_names = ["Hyperparameter", "Value"]
    for params in debug_params:
        if params != "":
            if globals()[params] == "":
                value = "False"
            else:
                value = globals()[params]
            table.add_row([params, value])
    table.align = "l"
    print(table)

    arg_list = train_command.split()
    mod_train_command = {'command': arg_list}
    
    train_folder = os.path.dirname(output_dir)
    
    # save the YAML string to a file
    with open(str(train_folder)+'/dreambooth_lora_cmd.yaml', 'w') as f:
        yaml.dump(mod_train_command, f)

f = open("./train.sh", "w")
f.write(train_command)
f.close()



