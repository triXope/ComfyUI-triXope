import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "triXope.ColorFX.Groups",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ColorFX") {

            // --- 1. THE SAVE PROTECTOR (Fixes the lost values) ---
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (onSerialize) onSerialize.apply(this, arguments);
                if (this.widgets && o.widgets_values) {
                    let cleanValues = [];
                    for (let i = 0; i < this.widgets.length; i++) {
                        // Strip our custom buttons out of the save data so the index stays aligned!
                        if (!this.widgets[i].isCustomGrouperBtn) {
                            cleanValues.push(o.widgets_values[i]);
                        }
                    }
                    o.widgets_values = cleanValues;
                }
            };

            // --- 2. THE LOAD PROTECTOR (Fixes the lost node size) ---
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                if (onConfigure) onConfigure.apply(this, arguments);
                if (o.size) {
                    this._true_saved_size = [o.size[0], o.size[1]];
                }
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.properties = this.properties || {};

                // Map our Python string inputs exactly to the widget names
                const groupDefinitions = [
                    { btnName: "grp_color_correction", label: "Color Correction", widgets: ["hdr_intensity", "shadow_intensity", "highlight_intensity", "gamma", "brightness", "contrast", "saturation", "enhance_color"] },
                    { btnName: "grp_lut", label: "LUT Processing", widgets: ["lut_name", "lut_strength", "lut_log_process"] },
                    { btnName: "grp_enhancements", label: "Enhancements", widgets: ["sharpness", "edge_enhance_strength", "detail_enhance_strength"] },
                    { btnName: "grp_blur", label: "Blur Effects", widgets: ["blur_radius", "gaussian_blur_radius", "radial_blur_strength", "radial_blur_center_x", "radial_blur_center_y", "radial_blur_focus_spread", "radial_blur_steps"] },
                    { btnName: "grp_stylistic", label: "Stylistic Effects", widgets: ["chromatic_aberration_r_x", "chromatic_aberration_r_y", "chromatic_aberration_b_x", "chromatic_aberration_b_y", "chromatic_blur_amount", "simple_film_grain_intensity", "simple_film_grain_monochrome", "scanline_intensity", "vignette_intensity", "vignette_center_x", "vignette_center_y", "soft_light_opacity", "soft_light_blur_radius"] }
                ];

                const toggleWidget = (w, visible) => {
                    if (!w) return;
                    w.hidden = !visible;
                    if (w.element) w.element.style.display = visible ? "" : "none";
                    if (w.inputEl) w.inputEl.style.display = visible ? "" : "none";

                    if (visible) {
                        if (w.hasOwnProperty('origComputeSize')) w.computeSize = w.origComputeSize;
                        else delete w.computeSize;
                    } else {
                        if (!w.hasOwnProperty('origComputeSize')) {
                            w.origComputeSize = w.hasOwnProperty('computeSize') ? w.computeSize : undefined;
                        }
                        w.computeSize = () => [0, 0];
                        w.y = undefined;
                        w.last_y = undefined;
                    }
                };

                setTimeout(() => {
                    if (!this.widgets) return;

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
                                    if (targetW) toggleWidget(targetW, isExpanded);
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
                                ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#222";
                                ctx.fillRect(15, y, widget_width - 30, H);
                                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#AFAFAF";
                                ctx.font = LiteGraph.WIDGET_TEXT_FONT || "12px Arial";
                                ctx.textAlign = "left"; 
                                let x_start = (widget_width / 2) - (longestLabelWidth / 2);
                                ctx.fillText(this.name, x_start, y + H * 0.7);
                            };

                            this.widgets.pop(); 
                            this.widgets.splice(dummyIndex, 0, btn);

                            for (let wName of def.widgets) {
                                let targetW = this.widgets.find(w => w.name === wName);
                                if (targetW) toggleWidget(targetW, isExpanded);
                            }
                        }
                    }

                    // --- NATIVE SIZE RESTORE FIX ---
                    let bootMinSize = this.computeSize([this.size[0], 0]);
                    let finalW = this.size[0];
                    let finalH = bootMinSize[1];

                    if (this._true_saved_size) {
                        finalW = this._true_saved_size[0];
                        finalH = this._true_saved_size[1];
                        delete this._true_saved_size; 
                    }

                    this.setSize([finalW, finalH]);
                    // -------------------------------

                    app.graph.setDirtyCanvas(true, true);
                }, 250);

                return r;
            };
        }
    }
});