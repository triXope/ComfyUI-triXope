import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "triXope.FilmAuteur_LTX.Groups",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FilmAuteur_LTX") {

            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                // 1. Run standard serialization first
                if (onSerialize) onSerialize.apply(this, arguments);

                // 2. Reach directly into the JSON save file (o) and fix the port labels!
                const fixJsonLabels = (jsonSlots, liveSlots) => {
                    if (!jsonSlots || !liveSlots) return;
                    for (let i = 0; i < jsonSlots.length; i++) {
                        let jSlot = jsonSlots[i];
                        let lSlot = liveSlots[i];
                        
                        // If the port was collapsed, inject its true stored name into the JSON
                        if (lSlot && lSlot.has_old_label && lSlot.old_label) {
                            jSlot.label = lSlot.old_label;
                        } 
                        // Failsafe: Never let ComfyUI save a blank string!
                        else if (!jSlot.label || jSlot.label.trim() === "") {
                            jSlot.label = jSlot.name; 
                        }
                    }
                };
                
                // Repair the inputs and outputs in the save object
                fixJsonLabels(o.inputs, this.inputs);
                fixJsonLabels(o.outputs, this.outputs);

                // 3. Clean up the custom widget values (Original functionality)
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

            // --- 3. RGTHREE-STYLE CONNECTIONS COLLAPSER ---
            const oldGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
                if (oldGetExtraMenuOptions) oldGetExtraMenuOptions.apply(this, arguments);
                options.push(null); // Separator line
                options.push({
                    content: this.properties.collapse_connections ? "🟢 Expand Connections" : "▶ Collapse Connections",
                    callback: () => {
                        // 1. Log the starting height
                        let oldMinY = this.computeSize()[1];
                        
                        this.properties.collapse_connections = !this.properties.collapse_connections;

                        // 2. Toggle Labels to prevent text overlap (with Fallback Rescue)
                        const toggleLabels = (slots) => {
                            if (!slots) return;
                            for (let cxn of slots) {
                                if (this.properties.collapse_connections) {
                                    if (!cxn.has_old_label) {
                                        cxn.has_old_label = true;
                                        // SAFE FALLBACK: Never trap an empty string as the true label!
                                        cxn.old_label = (!cxn.label || cxn.label.trim() === "") ? cxn.name : cxn.label;
                                    }
                                    cxn.label = " ";
                                } else {
                                    if (cxn.has_old_label) {
                                        cxn.has_old_label = false;
                                        cxn.label = cxn.old_label || cxn.name;
                                        delete cxn.old_label;
                                    } else if (!cxn.label || cxn.label.trim() === "") {
                                        cxn.label = cxn.name;
                                    }
                                }
                            }
                        };
                        toggleLabels(this.inputs);
                        toggleLabels(this.outputs);

                        // 3. Calculate the exact delta shift and dynamically resize!
                        let newMinY = this.computeSize()[1];
                        let deltaY = newMinY - oldMinY;
                        
                        this.setSize([
                            this.size[0], 
                            Math.max(newMinY, this.size[1] + deltaY) 
                        ]);
                        app.graph.setDirtyCanvas(true, true);
                    }
                });
            };

            const oldGetInputPos = nodeType.prototype.getInputPos;
            nodeType.prototype.getInputPos = function(slotNumber) {
                const fn = oldGetInputPos || this.constructor.prototype.getInputPos || LiteGraph.LGraphNode.prototype.getInputPos;
                return fn.call(this, this.properties.collapse_connections ? 0 : slotNumber);
            };

            const oldGetOutputPos = nodeType.prototype.getOutputPos;
            nodeType.prototype.getOutputPos = function(slotNumber) {
                const fn = oldGetOutputPos || this.constructor.prototype.getOutputPos || LiteGraph.LGraphNode.prototype.getOutputPos;
                return fn.call(this, this.properties.collapse_connections ? 0 : slotNumber);
            };

            const oldGetConnectionPos = nodeType.prototype.getConnectionPos;
            nodeType.prototype.getConnectionPos = function(isInput, slotNumber, out) {
                const fn = oldGetConnectionPos || this.constructor.prototype.getConnectionPos || LiteGraph.LGraphNode.prototype.getConnectionPos;
                return fn.call(this, isInput, this.properties.collapse_connections ? 0 : slotNumber, out);
            };

            // THE MIND TRICK: Safely calculate height without negative math crashes!
            const oldComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function(out) {
                let tempInp, tempOut;
                
                // If collapsed, temporarily trick LiteGraph into thinking there is only 1 slot
                if (this.properties.collapse_connections) {
                    tempInp = this.inputs;
                    tempOut = this.outputs;
                    this.inputs = tempInp && tempInp.length > 0 ? [tempInp[0]] : [];
                    this.outputs = tempOut && tempOut.length > 0 ? [tempOut[0]] : [];
                }

                // LiteGraph safely calculates the exact minimum height needed for the UI widgets!
                let size = oldComputeSize.apply(this, arguments);

                // Instantly restore the real slots before the UI renders
                if (this.properties.collapse_connections) {
                    this.inputs = tempInp;
                    this.outputs = tempOut;
                }
                
                return size;
            };
            // ----------------------------------------------

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.properties = this.properties || {};

                const groupDefinitions = [
                    { btnName: "grp_mode", label: "Mode Activation", widgets: ["primary_sampling", "spatial_upscale", "temporal_upscale", "restore_faces", "enable_colorfx", "enable_preview", "enable_final_video", "video_mode", "image_strength", "img_compression", "audio_select", "identity_guidance_scale"] },
                    { btnName: "grp_prompting", label: "Prompting", widgets: ["character_descriptions", "location_description", "scene_descriptions", "use_ollama", "ollama_url", "ollama_model"] },
                    { btnName: "grp_specs", label: "Video Specs", widgets: ["seed_number", "control_before_generate", "target_resolution", "length_in_seconds", "frame_rate"] },
                    { btnName: "grp_sampling", label: "Primary Sampling", widgets: ["primary_sampler_name", "primary_cfg", "primary_steps", "eta", "bongmath", "enable_nag"] },
                    { btnName: "grp_refinement", label: "Upscale & Refine", widgets: ["spatial_passes", "spatial_sampler", "spatial_cfg", "spatial_sigmas", "temporal_denoise", "facerestore_model", "facedetection", "codeformer_fidelity", "face_restore_color_match", "face_restore_edge_blur", "face_restore_blend"] },
                    
                    // --- NESTED COLOR FX GROUPS ---
                    { btnName: "grp_cfx_main", label: "Color FX", widgets: ["enable_color_correction", "enable_lut_processing", "enable_enhancements", "enable_blur_effects", "enable_stylistic_effects"] },
                    { btnName: "grp_cfx_color", label: "Color Correction", parent: "grp_cfx_main", widgets: ["hdr_intensity", "shadow_intensity", "highlight_intensity", "gamma", "brightness", "contrast", "saturation", "enhance_color"] },
                    { btnName: "grp_cfx_lut", label: "LUT Processing", parent: "grp_cfx_main", widgets: ["lut_name", "lut_strength", "lut_log_process"] },
                    { btnName: "grp_cfx_enhancements", label: "Enhancements", parent: "grp_cfx_main", widgets: ["sharpness", "edge_enhance_strength", "detail_enhance_strength"] },
                    { btnName: "grp_cfx_blur", label: "Blur Effects", parent: "grp_cfx_main", widgets: ["blur_radius", "gaussian_blur_radius", "radial_blur_strength", "radial_blur_center_x", "radial_blur_center_y", "radial_blur_focus_spread", "radial_blur_steps"] },
                    { btnName: "grp_cfx_stylistic", label: "Stylistic Effects", parent: "grp_cfx_main", widgets: ["chromatic_aberration_r_x", "chromatic_aberration_r_y", "chromatic_aberration_b_x", "chromatic_aberration_b_y", "chromatic_blur_amount", "simple_film_grain_intensity", "simple_film_grain_monochrome", "scanline_intensity", "vignette_intensity", "vignette_center_x", "vignette_center_y", "soft_light_opacity", "soft_light_blur_radius"] },
                    // ---------------------------
                    
                    { btnName: "grp_performance", label: "Performance", widgets: ["enable_fp16_accumulation", "sage_attention", "autoregressive_chunking", "chunk_size_seconds", "context_window_seconds", "chunks_feedforward", "clear_models_and_cache"] },
                    { btnName: "grp_output", label: "Output", widgets: ["stage1_preview", "final_video"] }
                ];

                // THE TOOLTIPS
                const WIDGET_TOOLTIPS = {
                    "primary_sampling": "Enable to run the primary UNet generation. Disable to bypass directly to upscaling or output decoding.",
                    "spatial_upscale": "Enable spatial upscaling to increase resolution.",
                    "temporal_upscale": "Triggers the temporal upscaler on or off.",
                    "restore_faces": "Apply CodeFormer face restoration to all frames. Requires a valid model selected below.",
                    "enable_colorfx": "Enable the cinematic Color FX post-processing mastering suite.",
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
                    "target_resolution": "Target resolution of the final video.",
                    "length_in_seconds": "Total video length in seconds. In multi-shot mode, this will automatically round to the nearest whole number evenly divisible by your shot count.",
                    "frame_rate": "Target frames per second.",
                    "primary_steps": "Enter a single number for steps (e.g., 20), or a comma-separated list for manual sigmas (e.g., 1.0, 0.995, 0.99, 0.9875, 0.975, 0.65, 0.28, 0.07, 0.0).",
                    "eta": "Calculated noise amount to be added, then removed, after each step.",
                    "bongmath": "Injects BONGMATH parameter into extra_options.",
                    "enable_nag": "Enable Normalized Attention Guidance (NAG) to dramatically improve prompt adherence using optimal hidden settings.",
                    "spatial_passes": "Number of upscaling stages. 1 = One 2x upscale pass, 2 = Two 2x upscale passes (4x total resolution boost).",
                    "temporal_denoise": "Use a low denoise value between 0.15 and 0.35 for optimal temporal upscaling results.",
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

                // --- HIERARCHICAL VISIBILITY ENGINE ---
                const refreshGroupVisibility = () => {
                    let oldMinY = this.computeSize()[1];

                    for (let def of groupDefinitions) {
                        // 1. Is this group's parent expanded? (If no parent, assume true)
                        let isParentExpanded = def.parent ? this.properties["groupState_" + def.parent] : true;
                        
                        // 2. Is this specific group expanded?
                        let isGroupExpanded = this.properties["groupState_" + def.btnName];
                        
                        // 3. Contents are only visible if BOTH this group AND its parent are expanded
                        let contentsVisible = isParentExpanded && isGroupExpanded;

                        // Safely toggle the subgroup's UI button itself based on the parent's visibility
                        if (def.parent) {
                            let btn = this.widgets.find(w => w.originalName === def.btnName);
                            if (btn) toggleWidget(btn, isParentExpanded);
                        }

                        // Safely toggle all input widgets inside this group
                        for (let wName of def.widgets) {
                            let targetW = this.widgets.find(w => w.name === wName);
                            if (targetW) toggleWidget(targetW, contentsVisible);
                        }
                    }

                    // Dynamically calculate the delta required to fit the new UI state
                    let newMinSize = this.computeSize();
                    let deltaY = newMinSize[1] - oldMinY;
                    this.setSize([
                        Math.max(this.size[0], newMinSize[0]), 
                        Math.max(newMinSize[1], this.size[1] + deltaY) 
                    ]);
                    app.graph.setDirtyCanvas(true, true);
                };

                setTimeout(() => {
                    // Check if loaded in collapsed state and hide labels to prevent overlap
                    if (this.properties.collapse_connections) {
                        const wipe = (slots) => {
                            if (!slots) return;
                            for (let cxn of slots) {
                                if (!cxn.has_old_label) {
                                    cxn.has_old_label = true;
                                    // SAFE FALLBACK: Rescue corrupted JSON saves!
                                    cxn.old_label = (!cxn.label || cxn.label.trim() === "") ? cxn.name : cxn.label;
                                }
                                cxn.label = " ";
                            }
                        }
                        wipe(this.inputs);
                        wipe(this.outputs);
                    } else {
                        // If the workflow loaded expanded, but labels were corrupted by a prior save, rescue them!
                        const rescue = (slots) => {
                            if (!slots) return;
                            for (let cxn of slots) {
                                if (!cxn.label || cxn.label.trim() === "") {
                                    cxn.label = cxn.name;
                                    cxn.has_old_label = false;
                                }
                            }
                        }
                        rescue(this.inputs);
                        rescue(this.outputs);
                    }
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

                            // Generate the actual interactive button
                            let btn = this.addWidget("button", (isExpanded ? "▼ " : "▶ ") + def.label, null, () => {
                                this.properties[propKey] = !this.properties[propKey];
                                btn.name = (this.properties[propKey] ? "▼ " : "▶ ") + def.label;
                                refreshGroupVisibility();
                            });

                            btn.isCustomGrouperBtn = true;
                            btn.originalName = def.btnName; // Critical for finding it during the global visibility pass!

                            btn.draw = function(ctx, node, widget_width, y, H) {
                                let isSub = def.parent !== undefined;
                                let indent = isSub ? 20 : 15; // Indent subgroups visually!
                                let fillWidth = widget_width - indent - 15;

                                ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#222";
                                ctx.fillRect(indent, y, fillWidth, H);

                                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#AFAFAF";
                                ctx.font = LiteGraph.WIDGET_TEXT_FONT || "12px Arial";
                                ctx.textAlign = "left"; 

                                let x_start = (widget_width / 2) - (longestLabelWidth / 2) + (isSub ? 10 : 0);
                                ctx.fillText(this.name, x_start, y + H * 0.7);
                            };

                            this.widgets.pop(); 
                            this.widgets.splice(dummyIndex, 0, btn);
                        }
                    }
                    
                    // Run a final master visibility pass to sync all parents and children perfectly
                    refreshGroupVisibility();

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

            if (node.properties["groupState_grp_output"] === false) {
                previewWidget.origComputeSize = previewWidget.computeSize;
                previewWidget.computeSize = () => [0, 0];
                previewWidget.hidden = true;
                if (previewWidget.element) previewWidget.element.style.display = "none";
                previewWidget.y = undefined;
                previewWidget.last_y = undefined;
            }

            if (!node._hasPreviewResizeHook) {
                node._hasPreviewResizeHook = true;
                const origOnResize = node.onResize;
                node.onResize = function(size) {
                    if (origOnResize) origOnResize.apply(this, arguments);
                    const pWidget = this.widgets && this.widgets.find(w => w.name === "stage1_preview");
                    if (pWidget && pWidget.element) {
                        let ratio = (pWidget.element.videoWidth > 0) ? (pWidget.element.videoHeight / pWidget.element.videoWidth) : (9 / 16);
                        pWidget.element.style.height = (size[0] * ratio) + "px";
                    }
                };
            }

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

// --- MASTER FINAL VIDEO PREVIEW RECEIVER (AUTO-REPLACE PREVIEW) ---
api.addEventListener("trixope_ltxv_final_video", (event) => {
    const data = event.detail;
    const node = app.graph.getNodeById(data.node);

    if (node) {
        // Automatically unload and remove the mid-generation stage 1 preview widget if active
        let stage1Widget = node.widgets && node.widgets.find(w => w.name === "stage1_preview");
        if (stage1Widget) {
            if (stage1Widget.element) {
                stage1Widget.element.pause();
                stage1Widget.element.removeAttribute('src');
                stage1Widget.element.load();
            }
            node.removeWidget(stage1Widget);
        }

        let finalWidget = node.widgets && node.widgets.find(w => w.name === "final_video");

        if (!finalWidget) {
            const videoEl = document.createElement("video");
            videoEl.style.width = "100%";
            videoEl.style.objectFit = "contain";
            videoEl.controls = true; 
            videoEl.autoplay = true;
            videoEl.loop = true;
            videoEl.muted = true;

            finalWidget = node.addDOMWidget("final_video", "preview", videoEl, {
                serialize: false,
                hideOnZoom: false
            });

            finalWidget.computeSize = function(width) {
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

            if (node.properties["groupState_grp_output"] === false) {
                finalWidget.origComputeSize = finalWidget.computeSize;
                finalWidget.computeSize = () => [0, 0];
                finalWidget.hidden = true;
                if (finalWidget.element) finalWidget.element.style.display = "none";
            }
        }

        finalWidget.element.src = api.apiURL(`/view?filename=${data.filename}&type=${data.type}&t=${Date.now()}`);
        finalWidget.element.onloadedmetadata = () => {
            node.setSize([node.size[0], Math.max(node.size[1], node.computeSize([node.size[0], node.size[1]])[1])]);
            app.graph.setDirtyCanvas(true, true);
        };
        finalWidget.element.play().catch(e => console.warn("Master playback autoplay blocked: ", e));
    }
});

api.addEventListener("executing", (event) => {
    const nodeId = event.detail;
    if (!nodeId) return;

    const node = app.graph.getNodeById(nodeId);
    if (node && node.type === "FilmAuteur_LTX") {
        let toggleWidget = node.widgets && node.widgets.find(w => w.name === "enable_preview");
        let previewWidget = node.widgets && node.widgets.find(w => w.name === "stage1_preview");
        let finalWidget = node.widgets && node.widgets.find(w => w.name === "final_video");

        if (toggleWidget && !toggleWidget.value && previewWidget) {
            if (previewWidget.element) {
                previewWidget.element.pause();
                previewWidget.element.removeAttribute('src');
                previewWidget.element.load();
            }
            node.removeWidget(previewWidget);
        }
        // Added cleanup routine
        if (finalWidget) {
            if (finalWidget.element) {
                finalWidget.element.pause();
                finalWidget.element.removeAttribute('src');
                finalWidget.element.load();
            }
            node.removeWidget(finalWidget);
        }
        app.graph.setDirtyCanvas(true, true);
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
