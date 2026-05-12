import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "triXope.FilmAuteur_LTXV.Groups",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FilmAuteur_LTXV") {

            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (onSerialize) onSerialize.apply(this, arguments);
                if (this.widgets && o.widgets_values) {
                    let cleanValues = [];
                    for (let i = 0; i < this.widgets.length; i++) {
                        if (!this.widgets[i].isCustomGrouperBtn) {
                            cleanValues.push(o.widgets_values[i]);
                        }
                    }
                    o.widgets_values = cleanValues;
                }
            };

            // --- 1. DYNAMIC UI RESIZING (THE ENFORCER) ---
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                if (onResize) onResize.apply(this, arguments);
                if (!this.widgets) return;

                let textWidgets = this.widgets.filter(w => (w.type === "customtext" || (w.inputEl && w.inputEl.tagName === "TEXTAREA")) && !w.hidden);
                if (textWidgets.length === 0) return;

                for (let w of textWidgets) {
                    w.computeSize = function(width) { return [width, 0]; };
                }

                let minNodeSize = this.computeSize([size[0], 0]);
                let fixedHeight = minNodeSize[1];

                let availableHeight = size[1] - fixedHeight - 15; 
                let heightPerBox = Math.max(40, availableHeight / textWidgets.length);

                for (let w of textWidgets) {
                    if (!w.options) w.options = {};
                    w.options.height = heightPerBox;

                    w.computeSize = function(width) { return [width, heightPerBox]; };

                    if (w.inputEl) {
                        let finalHeight = (heightPerBox - 10) + "px";
                        w.inputEl.style.setProperty("height", finalHeight, "important");
                        w.inputEl.style.setProperty("min-height", finalHeight, "important");
                        w.inputEl.style.setProperty("max-height", finalHeight, "important");
                    }
                }
            };

            // --- 2. THE LOAD PROTECTOR ---
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                if (onConfigure) onConfigure.apply(this, arguments);
                if (o.size) {
                    this._true_saved_size = [o.size[0], o.size[1]];
                    if (this.widgets) {
                        for (let w of this.widgets) {
                            if (w.type === "customtext" || (w.inputEl && w.inputEl.tagName === "TEXTAREA")) {
                                w.computeSize = function(width) { return [width, 40]; };
                            }
                        }
                    }
                }
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.properties = this.properties || {};

                const groupDefinitions = [
                    { btnName: "grp_mode", label: "Mode Select", widgets: ["video_mode", "image_strength", "img_compression", "audio_select", "identity_guidance_scale"] },
                    { btnName: "grp_prompting", label: "Prompting", widgets: ["character_descriptions", "location_description", "scene_descriptions", "use_ollama", "ollama_url", "ollama_model"] },
                    { btnName: "grp_specs", label: "Video Specs", widgets: ["seed_number", "control_before_generate", "target_width", "target_height", "length_in_seconds", "frame_rate"] },
                    { btnName: "grp_sampling", label: "Primary Sampling", widgets: ["primary_sampler_name", "primary_cfg", "primary_steps", "eta", "bongmath", "enable_nag"] },
                    { btnName: "grp_refinement", label: "Upscale & Refine", widgets: ["spatial_upscale", "spatial_passes", "spatial_sampler", "spatial_cfg", "spatial_sigmas", "temporal_upscale", "temporal_denoise", "restore_faces", "facerestore_model", "facedetection", "codeformer_fidelity", "face_restore_color_match", "face_restore_edge_blur", "face_restore_blend"] },
                    { btnName: "grp_performance", label: "Performance", widgets: ["enable_fp16_accumulation", "sage_attention", "autoregressive_chunking", "chunk_size_seconds", "context_window_seconds", "chunks_feedforward", "clear_models_and_cache"] },
                    { btnName: "grp_preview", label: "Preview", widgets: ["enable_preview", "stage1_preview"] }
                ];

                const WIDGET_TOOLTIPS = {
                    "video_mode": "text: generates from prompt only.\nimage: uses image as the first frame.\nreference: uses image globally as a style/concept reference.",
                    "image_strength": "Strength of the image conditioning. Values over 1.0 may cause artifacts or burning.",
                    "img_compression": "Compression applied to input images (CRF). 0 bypasses.",
                    "audio_select": "internal: uses LTX native generated audio.\nsource: encodes the provided audio track.\nreference: uses audio for ID-LoRA voice guidance.",
                    "identity_guidance_scale": "Strength of identity guidance for ID-LoRA.",
                    "character_descriptions": "Provide a detailed description for each character (overridden by image reference).",
                    "location_description": "Provide a detailed description of the location(s).",
                    "scene_descriptions": 'Provide a detailed description for each shot, separated by "|" (eg. shot 1 | shot 2 | shot 3). Note: length_in_seconds must be evenly divisible by total number of shots.',
                    "use_ollama": "Use local Ollama to visually describe inputs and revamp the prompt.",
                    "seed_number": "The specific generation seed.",
                    "control_before_generate": "Dictates how the seed changes BEFORE generating.",
                    "target_width": "Target width of the final video.",
                    "target_height": "Target height of the final video.",
                    "length_in_seconds": "Total video length in seconds. In multi-shot mode, this will automatically round to the nearest whole number evenly divisible by your shot count.",
                    "frame_rate": "Target frames per second.",
                    "primary_steps": "Enter a single number for steps (e.g., 20), or a comma-separated list for manual sigmas (e.g., 1.0, 0.995, 0.99, 0.9875, 0.975, 0.65, 0.28, 0.07, 0.0).",
                    "eta": "Calculated noise amount to be added, then removed, after each step.",
                    "bongmath": "Injects BONGMATH parameter into extra_options.",
                    "enable_nag": "Enable Normalized Attention Guidance (NAG) to dramatically improve prompt adherence using optimal hidden settings.",
                    "spatial_upscale": "Enable spatial upscaling to increase resolution.",
                    "spatial_passes": "Number of upscaling stages. 1 = One 2x upscale pass, 2 = Two 2x upscale passes (4x total resolution boost).",
                    "temporal_upscale": "[TEMPORARILY DISABLED] Triggers the temporal upscaler on or off.",
                    "temporal_denoise": "Use a low denoise value between 0.15 and 0.35 for optimal temporal upscaling results.",
                    "restore_faces": "Apply CodeFormer face restoration to all frames. Requires a valid model selected below.",
                    "facerestore_model": "Select the CodeFormer Face Restore Model.",
                    "facedetection": "Face detection model.",
                    "codeformer_fidelity": "Balance between quality and identity. 0 is high quality, 1 is high fidelity.",
                    "face_restore_color_match": "Automatically match the hue, saturation, and luminance of the restored face to the original degraded face.",
                    "face_restore_edge_blur": "Apply a soft alpha feathering to the edges of the restored face before pasting to eliminate harsh boundary lines.",
                    "face_restore_blend": "Opacity of the restored face. Lower values significantly reduce video flickering by anchoring to the original frame.",
                    "enable_fp16_accumulation": "Enable torch.backends.cuda.matmul.allow_fp16_accumulation.",
                    "sage_attention": "Patch comfy attention to use sageattn.",
                    "autoregressive_chunking": "Automatically flush VRAM and outpaint the video in chunks if the length exceeds the chunk size.",
                    "chunk_size_seconds": "The max duration (in seconds) generated in a single pass before flushing VRAM.",
                    "context_window_seconds": "Seconds of previous video the model can 'see'. Caps render time! Set equal to chunk_size to keep rendering times perfectly flat.",
                    "chunks_feedforward": "Number of chunks to split the feedforward activations into to reduce peak VRAM usage.",
                    "clear_models_and_cache": "Frees up VRAM by forcefully unloading models and emptying the cache at the end of the generation.",
                    "enable_preview": "Enable or disable the animated preview after the first processing stage."
                };

                const toggleWidget = (w, visible) => {
                    if (!w) return;
                    if (w.type === "converted-widget") return;

                    w.hidden = !visible;
                    if (w.element) w.element.style.display = visible ? "" : "none";
                    if (w.inputEl) w.inputEl.style.display = visible ? "" : "none";

                    if (visible) {
                        if (w.hasOwnProperty('origComputeSize')) {
                            w.computeSize = w.origComputeSize;
                        } else {
                            delete w.computeSize; 
                        }
                        w.tooltip = WIDGET_TOOLTIPS[w.name] || "";

                        if (w.inputEl) w.inputEl.title = w.tooltip;
                        if (w.element) w.element.title = w.tooltip;
                    } else {
                        if (!w.hasOwnProperty('origComputeSize')) {
                            w.origComputeSize = w.hasOwnProperty('computeSize') ? w.computeSize : undefined;
                        }
                        w.computeSize = () => [0, 0];

                        w.y = undefined;
                        w.last_y = undefined;

                        w.tooltip = null; 
                    }
                };

                setTimeout(() => {
                    let tempCtx = document.createElement("canvas").getContext("2d");
                    tempCtx.font = LiteGraph.WIDGET_TEXT_FONT || "12px Arial";
                    let longestLabelWidth = 0;
                    for (let d of groupDefinitions) {
                        let w = tempCtx.measureText("▼ " + d.label).width;
                        if (w > longestLabelWidth) longestLabelWidth = w;
                    }

                    for (let def of groupDefinitions) {
                        let dummyIndex = this.widgets.findIndex(w => w.name === def.btnName);

                        if (dummyIndex !== -1) {
                            let dummyWidget = this.widgets[dummyIndex];
                            toggleWidget(dummyWidget, false); 

                            let propKey = "groupState_" + def.btnName;
                            if (this.properties[propKey] === undefined) {
                                this.properties[propKey] = false; 
                            }
                            let isExpanded = this.properties[propKey];

                            let btn = this.addWidget("button", (isExpanded ? "▼ " : "▶ ") + def.label, null, () => {
                                let oldMinY = this.computeSize()[1]; 

                                isExpanded = !isExpanded;
                                this.properties[propKey] = isExpanded; 
                                btn.name = (isExpanded ? "▼ " : "▶ ") + def.label;

                                for (let wName of def.widgets) {
                                    let targetW = this.widgets.find(w => w.name === wName);
                                    toggleWidget(targetW, isExpanded);
                                }

                                let newMinSize = this.computeSize();
                                let deltaY = newMinSize[1] - oldMinY;

                                this.setSize([
                                    Math.max(this.size[0], newMinSize[0]), 
                                    Math.max(newMinSize[1], this.size[1] + deltaY) 
                                ]);

                                app.graph.setDirtyCanvas(true, true);
                            });

                            btn.isCustomGrouperBtn = true;

                            btn.draw = function(ctx, node, widget_width, y, H) {
                                // 1. Draw the native button background
                                ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#222";
                                ctx.fillRect(15, y, widget_width - 30, H);

                                // 2. Setup native text styling
                                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#AFAFAF";
                                ctx.font = LiteGraph.WIDGET_TEXT_FONT || "12px Arial";
                                ctx.textAlign = "left"; // Override default centering

                                // 3. Calculate perfectly centered left-justified anchor
                                let x_start = (widget_width / 2) - (longestLabelWidth / 2);
                                ctx.fillText(this.name, x_start, y + H * 0.7);
                            };

                            this.widgets.pop(); 
                            this.widgets.splice(dummyIndex, 0, btn);

                            for (let wName of def.widgets) {
                                let targetW = this.widgets.find(w => w.name === wName);
                                toggleWidget(targetW, isExpanded);
                            }
                        }
                    }

                    // --- FINAL LOAD-SIZE CALCULATION ---

                    // 1. Identify text boxes
                    let textWidgets = this.widgets.filter(w => w.type === "customtext" || (w.inputEl && w.inputEl.tagName === "TEXTAREA"));

                    // 2. Temporarily zero them out to find the height of purely static UI (pins/buttons)
                    for (let w of textWidgets) {
                        w.computeSize = function(width) { return [width, 0]; };
                    }

                    // 3. Calculate "Compact Height": Static UI + (40px per box) + buffer
                    let bootMinSize = this.computeSize([this.size[0], 0]);
                    let compactHeight = bootMinSize[1] + (textWidgets.length * 40) + 20;

                    let finalW = this.size[0];
                    let finalH = compactHeight;

                    // 4. Apply the saved size if it exists, otherwise use compactHeight
                    if (this._true_saved_size) {
                        finalW = this._true_saved_size[0];
                        finalH = this._true_saved_size[1];
                        delete this._true_saved_size; 
                    }

                    // 5. Force the node to the final size and trigger the layout engine
                    this.setSize([finalW, finalH]);
                    if (this.onResize) {
                        this.onResize(this.size);
                    }

                    app.graph.setDirtyCanvas(true, true);

                }, 250);

                return r;
            };
        }
    }
});

// --- DYNAMIC NATIVE MP4 PREVIEW RECEIVER ---
api.addEventListener("trixope_ltxv_preview", (event) => {
    const data = event.detail;
    const node = app.graph.getNodeById(data.node);

    if (node) {
        let previewWidget = node.widgets && node.widgets.find(w => w.name === "stage1_preview");

        if (!previewWidget) {
            const videoEl = document.createElement("video");
            videoEl.style.width = "100%";
            videoEl.style.objectFit = "contain";
            videoEl.controls = true; 
            videoEl.autoplay = true;
            videoEl.loop = true;
            videoEl.muted = true;

            previewWidget = node.addDOMWidget("stage1_preview", "preview", videoEl, {
                serialize: false,
                hideOnZoom: false
            });

            previewWidget.computeSize = function(width) {
                if (!this.element || !this.element.getAttribute('src')) {
                    this.element.style.height = "0px";
                    return [width, 0];
                }

                let height = (width * 9) / 16; 
                if (this.element && this.element.videoWidth > 0) {
                    const ratio = this.element.videoHeight / this.element.videoWidth;
                    height = width * ratio;
                }

                this.element.style.height = height + "px";
                return [width, height + 10]; 
            };

            if (node.properties["groupState_grp_preview"] === false) {
                previewWidget.origComputeSize = previewWidget.computeSize;
                previewWidget.computeSize = () => [0, 0];
                previewWidget.hidden = true;
                if (previewWidget.element) previewWidget.element.style.display = "none";
                previewWidget.y = undefined;
                previewWidget.last_y = undefined;
            }

            const origOnResize = node.onResize;
            node.onResize = function(size) {
                if (origOnResize) origOnResize.apply(this, arguments);
                if (previewWidget.element) {
                    let ratio = 9 / 16;
                    if (previewWidget.element.videoWidth > 0) {
                        ratio = previewWidget.element.videoHeight / previewWidget.element.videoWidth;
                    }
                    previewWidget.element.style.height = (size[0] * ratio) + "px";
                }
            };

            const origOnRemoved = node.onRemoved;
            node.onRemoved = function() {
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
                if (previewWidget.element) {
                    previewWidget.element.pause();
                    previewWidget.element.removeAttribute('src');
                    previewWidget.element.load();
                }
            };
        }

        previewWidget.element.src = api.apiURL(`/view?filename=${data.filename}&type=${data.type}&t=${Date.now()}`);

        previewWidget.element.onloadedmetadata = () => {
            const currentWidth = node.size[0];
            const currentHeight = node.size[1];
            const idealSize = node.computeSize([currentWidth, currentHeight]);

            const newHeight = Math.max(currentHeight, idealSize[1]);

            node.setSize([currentWidth, newHeight]);
            app.graph.setDirtyCanvas(true, true);
        };

        previewWidget.element.play().catch(e => console.warn("Video autoplay blocked by browser: ", e));
    }
});

api.addEventListener("executing", (event) => {
    const nodeId = event.detail;
    if (!nodeId) return;

    const node = app.graph.getNodeById(nodeId);
    if (node && node.type === "FilmAuteur_LTXV") {
        let toggleWidget = node.widgets && node.widgets.find(w => w.name === "enable_preview");
        let previewWidget = node.widgets && node.widgets.find(w => w.name === "stage1_preview");

        if (toggleWidget && !toggleWidget.value && previewWidget) {
            if (previewWidget.element) {
                previewWidget.element.pause();
                previewWidget.element.removeAttribute('src');
                previewWidget.element.load();
            }
            node.removeWidget(previewWidget);
            app.graph.setDirtyCanvas(true, true);
        }
    }
});

api.addEventListener("trixope-global-seed", (event) => {
    let nodes = app.graph._nodes_by_id;

    for (let id in event.detail.seed_map) {
        let node = nodes[id];
        if (node && node.widgets) {
            const w = node.widgets.find((w) => w.name === 'seed_number' && w.type === 'number');
            if (w && event.detail.seed_map[id] !== undefined) {
                w.value = event.detail.seed_map[id];
            }
        }
    }
    app.graph.setDirtyCanvas(true, true);
});
