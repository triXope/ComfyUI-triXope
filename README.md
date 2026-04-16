Here you will find a collection of custom nodes for ComfyUI with the intent of enhancing your film production workflow.

Let me preface by saying that I am not a developer by trade, nor do I have a background in programming. I come from a traditional filmmaking background, with my focus being writing, directing, and cinematography.

With that said, I have been following the AI scene for quite some time now, working behind the scene on ways to implement AI into my own personal workflow and finding ways to utilize it as a tool, rather than try to fight it's constant progression - a battle that we cannot win.

The current highlight of this collection is a massive LTXV all-in-one node, Film Auteur (LTXV), which was designed with LTX 2.3 as the backbone.
What started off as a simple idea to create a node for injecting reference images into LTX quickly became a highly ambitious project.
It's not perfect, but it works.
At first glance, I'm sure the node looks overwhelming, with so much packed into it, but I assure you it's really not that bad, and can easily be broken down into sections to better understand it.

What the node does/features:

• Text-to-Video <br>
• Image-to-Video <br>
• Image Reference-to-Video <br>
• Audio-to-Video <br>
• Audio Reference (with ID-LoRA) <br>
• Ollama integration for prompt enhancement (I recommend Gemma 4) <br>
• Length input as seconds (calculated & converted to frame count internally based on fps) <br>
• Multi-shot inferencing using "|" separators between prompts <br>
• first_frame input accepts image batch for storyboard processing (1 shot per image coinciding with multi-prompt input) <br>
• Inifinite (truly) length by use of autoregressive chunking and built-in sliding context windows <br>
• Up to 3 sampling stages for built-in upsampling (model2_opt if wanted for stages 2 & 3) <br>
• Temporal upscaling option (double framerate and visual refinement) <br>
• Face restoration to help with cleaning up faces and removing artifacts (place face restore model with .pt or .pth file extension in "facerestore_models" folder (eg. codeformer, GFPGAN, etc.)) <br>
• Built-in sageattention and fp16 accumulation (must be installed to use) <br>
• Built in chunk feed forward (to assist in computational efficiency) <br>

Requirements: Download and place face restore model with .pt or .pth file extension in "facerestore_models" folder (eg. codeformer, GFPGAN, etc.)

Note: Refer to the tooltips for important information.

Just plug in your models, optional reference images and/or audio, set your desired paramerters, send it out to your preferred video save or combine node, and you're good-to-go.

More nodes coming soon...
