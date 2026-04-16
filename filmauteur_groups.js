import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "triXope.FilmAuteur_LTXV.Groups",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only target our specific node
        if (nodeData.name === "FilmAuteur_LTXV") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onConfigure = nodeType.prototype.onConfigure;
            
            // Hook into the loading sequence to capture the TRUE saved size before LiteGraph mangles it
            nodeType.prototype.onConfigure = function(info) {
                if (onConfigure) onConfigure.apply(this, arguments);
                if (info && info.size) {
                    this._true_saved_size = [...info.size];
                }
            };

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Ensure the native properties memory bank exists
                this.properties = this.properties || {};

                const groupDefinitions = [
                    { btnName: "grp_input_controls", label: "Input", widgets: ["bypass_img_ref", "bypass_first_frame", "load_audio_from_file", "bypass_audio_ref", "image_ref_str", "first_frame_str", "identity_guidance_scale"] },
                    { btnName: "grp_ollama_enhance", label: "Enhance", widgets: ["use_ollama", "ollama_url", "ollama_model"] },
                    { btnName: "grp_timeline_controls", label: "Timeline", widgets: ["noise_seed", "control_after_generate", "target_width", "target_height", "length_in_seconds", "frame_rate"] },
                    { btnName: "grp_sampling", label: "Sampling", widgets: ["sampling_stages", "primary_sampler_name", "primary_cfg", "primary_steps", "upsample_sampler_name", "upsample_cfg", "upsample_manual_sigmas", "eta", "bongmath", "autoregressive_chunking", "chunk_size_seconds", "context_window_seconds"] },
                    { btnName: "grp_refinement", label: "Refinement", widgets: ["temporal_upscale", "restore_faces", "facerestore_model", "facedetection", "codeformer_fidelity", "face_restore_color_match", "face_restore_edge_blur", "face_restore_blend"] },
                    { btnName: "grp_vram_optimization", label: "VRAM", widgets: ["enable_fp16_accumulation", "sage_attention", "chunks"] }
                ];

                const toggleWidget = (w, visible) => {
                    if (!w) return;
                    w.hidden = !visible;
                    if (w.element) w.element.style.display = visible ? "" : "none";
                    if (w.inputEl) w.inputEl.style.display = visible ? "" : "none";
                };

                // Execute after ComfyUI populates the widgets array and loads saved JSON data
                setTimeout(() => {
                    
                    for (let def of groupDefinitions) {
                        let dummyIndex = this.widgets.findIndex(w => w.name === def.btnName);
                        
                        if (dummyIndex !== -1) {
                            let dummyWidget = this.widgets[dummyIndex];
                            toggleWidget(dummyWidget, false);

                            // 1. STATE PERSISTENCE: Check the save file for this specific group
                            let propKey = "groupState_" + def.btnName;
                            if (this.properties[propKey] === undefined) {
                                this.properties[propKey] = false; // Default to collapsed for brand new nodes
                            }
                            let isExpanded = this.properties[propKey];

                            // 2. Create the button dynamically
                            let btn = this.addWidget("button", (isExpanded ? "▼ " : "▶ ") + def.label, null, () => {
                                // Capture the node's required height BEFORE we show/hide anything
                                let oldMinY = this.computeSize()[1]; 

                                isExpanded = !isExpanded;
                                this.properties[propKey] = isExpanded; // Save the new state to memory!
                                btn.name = (isExpanded ? "▼ " : "▶ ") + def.label;
                                
                                for (let wName of def.widgets) {
                                    let targetW = this.widgets.find(w => w.name === wName);
                                    toggleWidget(targetW, isExpanded);
                                }
                                
                                // Calculate the height delta and apply it smoothly to the current size
                                let newMinSize = this.computeSize();
                                let deltaY = newMinSize[1] - oldMinY;
                                
                                this.setSize([
                                    Math.max(this.size[0], newMinSize[0]), 
                                    Math.max(newMinSize[1], this.size[1] + deltaY) // Preserves your manual stretches!
                                ]);
                                
                                app.graph.setDirtyCanvas(true, true);
                            });

                            this.widgets.pop(); 
                            this.widgets.splice(dummyIndex, 0, btn);

                            // 3. Apply the saved visibility state to the widgets on boot
                            for (let wName of def.widgets) {
                                let targetW = this.widgets.find(w => w.name === wName);
                                toggleWidget(targetW, isExpanded);
                            }
                        }
                    }
                    
                    // Final boot resizing: Force the node to respect your TRUE saved dimensions
                    let bootMinSize = this.computeSize();
                    let finalW = this.size[0];
                    let finalH = this.size[1];

                    // If we intercepted a saved size during boot, use it!
                    if (this._true_saved_size) {
                        finalW = this._true_saved_size[0];
                        finalH = this._true_saved_size[1];
                        delete this._true_saved_size; // Clean up memory
                    }

                    this.setSize([
                        Math.max(bootMinSize[0], finalW), 
                        Math.max(bootMinSize[1], finalH)
                    ]);
                    
                    app.graph.setDirtyCanvas(true, true);
                    
                }, 250); 

                return r;
            };
        }
    }
});