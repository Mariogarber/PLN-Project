# Base Model

Selected Base model is `google/mt5-base`. Transformer model Seq2Seq type, include encoder and decoder. When trained, the model recieve a first word that represents the 
task he should do, and the rest of the input. It encode the input and use a decoder to inference the most probable token next.

# QLoRA Config based

## "Q" Config
Model loaded using 4 bits. However, they were `nf4` bit-type, so good representation. When compute, they cast to `bfloat16`, to add stability on training.

## LoRA Config

### First Approach

- Rank of matrix: 8
- Lora Alpha: 16
- No bias
- Dropout: 0.05
- Target Modules = Q and V

### Second Approach

- Rank of matrix: 8
- Lora Alpha: 16
- No bias
- Dropout: 0.05
- Target Modules = Q, V, K, O

# Custom Loss

Because we need to detoxify a dataset, we can use several loss function to finetune our model.

## Original Loss Function

Just use the original loss function that Mt5 implemented to try to reconstruct the sentences without any toxic word. However, some limitations of this is that the model 
does not care about generate or not toxic words, because it do not penalize them

## Toxic Penalization

We add a new component to the loss function, this component requires a list of toxic tokens, and compute a penalization if that tokens had higher probabilities on logits layer. We need to work with logits to ensure that the gradiant of this component does not just disapear. We define a gamma parameter to set the importance of the toxic penalization related with the original loss.

# Limitations

## Identity Mapping
Sometimes input are so similar to outputs, and if we do not compute the toxic penalization or the loss correctly, and if we do not define the task, the model just become a identity mapping, because is the easiest and the lowest-loss solution. 

## Not Defined task
MT5 tends to copy the input when no task is defined, soy we need to add the task before any input sequence. In our case, our task is just `detoxify: `.

# Tries

## First Try

The first try use the original function and the first lora config. I do not compute the metrics because the model was just copying the input. I do not add the task prefix to the inputs.

## Second Try

I add a toxic optimization of 0.15. Also I add the task before the input. Used the first lora config. The result is again the model copying the input IDK why.