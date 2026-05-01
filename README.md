# [Perpetual] WORK-IN-PROGRESS
<b><i>NOTICE: This repo is under heavy development and is likely to break and/or things not always work as expected. If you run into any errors, please ensure ComfyUI and this repo is fully updated to the latest version and reapply any node from this repo in question. If you still run into any errors, or are experiencing any other issues, please post your issue so that I can look into it.</i></b><br><br>
Don't expect everything to function perfectly.<br>
Please check back frequently as I am making constant updates and improvements...<br><br>
Here you will find a collection of custom nodes for ComfyUI with the intent of enhancing your film production workflow.<br><br>
Let me preface by saying that I am not a developer by trade, nor do I have a background in programming. I come from a traditional filmmaking background, with my focus being writing, directing, and cinematography.<br>
With that said, I have been following the AI scene for quite some time now, working behind the scene on ways to implement AI into my own personal workflow and finding ways to utilize it as a tool, rather than try to fight it's constant progression - a battle that we cannot win.<br><br>
The current highlight of this collection is a massive LTXV all-in-one node, Film Auteur (LTXV), which was designed with LTX 2.3 as the backbone.<br>
What started off as a simple idea to create a node for injecting reference images into LTX quickly became a highly ambitious project.<br>
It's not perfect, but it works.<br>
At first glance, I'm sure the node looks overwhelming, with so much packed into it, but I assure you it's really not that bad, and can easily be broken down into sections to better understand it.<br>

# Nodes

<b>triXope Film Auteur (LTXV)</b> - one node to rule them all!<br>
A complete production-ready suite for LTX 2.3 featuring:<br>
- Text-to-Video<br>
- Image-to-Video<br>
- Image Reference-to-Video (experimental work-in-progress)<br>
- Audio-to-Video<br>
- Audio Reference (with ID-LoRA)<br>
- Ollama integration for prompt enhancement (Gemma 4 26b recommended)<br>
- Normalized Attention Guidance (NAG) integration for negative prompting with low CFG
- Length input as seconds (calculated & converted to frame count internally based on fps)<br>
- Multi-shot inferencing using "|" separators between prompts<br>
- Image input accepts image batch for storyboard processing (1 shot per image coinciding with multi-prompt input) or reference images<br>
- LTXV Add Guide & LTX Add Video IC-LoRA Guide fully implemented under the hood for added control & consistency over reference images<br>
- Inifinite (truly) length by use of autoregressive chunking and built-in sliding context windows<br>
- Up to 3 sampling stages for built-in upsampling (model2_opt if wanted for stages 2 & 3)<br>
- Temporal upscaling option (enable to double framerate and visual refinement)<br>
- Face restoration to help with cleaning up faces and removing artifacts<br>
(place face restore model with .pt or .pth file extension in "facerestore_models" folder (eg. <a href="https://huggingface.co/models? search=codeformer">codeformer</a>, <a href="https://huggingface.co/models?search=gfpgan">GFPGAN</a>, etc.))<br>
- Built-in sageattention and fp16 accumulation (must be installed to use)<br>
- Built in chunk feed forward (to assist in computational efficiency)<br>
- Unload models & clear cache<br>
- Built in stage 1 preview<br>
- Internal Real-Time ETA counter (with assist node)<br>
Just plug in your models, optional reference images and/or audio, set your desired paramerters, send it out to your preferred video save or combine node, and you're good-to-go.<br><br>

<b>triXope LTXV Real-Time ETA:</b><br>
- An assist node for triXope Film Auteur (LTXV) to display the countdown timer.<br>
Note: This node is NOT precise and is intended as an aid to provide a rough estimation of the total time it will take to complete a generation - more useful with longer generations.<br>
The node is not a high priority, but I am open to suggestions for ways to improve the accuracy.<br><br>

<b>triXope ColorFX 🎬</b>
Post-processing suite featuring:<br>
- Color Correction<br>
- LUX Processing<br>
- Color Enhancements<br>
- Blur Effects<br>
- Stylistic Effects<br>

<b>triXope Directory File Selector 📂 (Live):</b><br>
- Scrub local directory
- Select file from list populated by selected directory
- Output file path for further processing of audio, image, video, etc.

Note: Refer to the tooltips on each node for important information.<br><br>
More nodes coming soon...<br>

# Installation
ComfyUI Manager (recommended method)<br><br>
Manual Method<br>
1. cd custom_nodes<br>
2. git clone https://github.com/triXope/ComfyUI-triXope.git<br>
3. Restart ComfyUI<br>

# Known Issues / Bugs
- Temporal Upscaler is a work-in-progress and still not properly engaged.
