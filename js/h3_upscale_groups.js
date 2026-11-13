// triXope - H3 Upscale: collapsible control groups and subgroups (same pattern as filmauteur_groups.js).
//
// Python declares one dummy widget per group (grp_*). On creation this extension hides each dummy and puts a
// real button bar in its place; clicking the bar expands/collapses the group's widgets and resizes the node.
// Open/closed state lives in node.properties (groupState_*), the node size is remembered for every
// open/closed combination (node.properties.h3_sizes), and the helper buttons are stripped from
// widgets_values on save so workflow files stay compatible.
//
// The node width is stored once per node (h3_width) instead of per open/closed combination: a workflow
// saved with every group collapsed used to reload with the width the node happened to have while
// collapsed, which is how the user's width was lost. The height is still remembered per combination.
//
// Stage order inside the node (top to bottom, and the order it runs):
//   H3 Upscale -> Face Detail -> Refine -> DLSS5 -> Color FX (Color Correction, LUT, Enhancements, Blur, Stylistic)
// Every group and subgroup starts collapsed.
import { app } from "../../scripts/app.js";

const GROUPS = [
    { btnName: "grp_h3_upscale", label: "H3 Upscale", enable: "h3_upscale_mode", widgets: [
        "h3_upscale_mode", "upscale_model", "latent_upscale_model", "scale_to_megapixels", "tile_width", "tile_height",
        "tile_padding", "seam_fix_width", "seam_fix_denoise", "seam_fix_padding", "seed", "control_after_generate",
        "steps", "sampler_name", "scheduler", "denoise", "diffusion_mix", "edge_protect", "detail_retention",
        "latent_noise_offset", "redraw_only", "align_to_source", "locks_follow_denoise",
        "pixel_batch_overlap"] },
    { btnName: "grp_face_detail", label: "Face Detail", enable: "enable_face_detailer", widgets: [
        "enable_face_detailer", "same_face", "face_denoise", "face_padding", "face_size", "face_detector",
        "face_detector_model", "face_confidence", "manual_face_boxes", "sam3_prompt", "reference_strength",
        "reference_pose_match", "reference_flow_align", "face_structure_lock", "face_chroma_lock",
        "face_seamless_blend", "frontal_only_threshold", "face_detect_stride"] },
    { btnName: "grp_refine", label: "Refine", enable: "enable_refiner", widgets: [
        "enable_refiner", "refine_amount", "refine_radius"] },
    { btnName: "grp_dlss5", label: "DLSS5", enable: "enable_dlss5", widgets: [
        "enable_dlss5", "dlss5_scale_to_megapixels", "nr_preset", "nr_style", "nr_intensity",
        "local_tone_strength",
        "local_structure_strength", "skin_structure_strength", "automatic_mask", "dlss_model_preset", "motion",
        "scene_change_threshold", "warmup_frames", "runtime_dir", "verify_neural_rendering"] },

    // --- Color FX (parent) and its subgroups ---
    { btnName: "grp_cfx_main", label: "Color FX", enable: "enable_colorfx", widgets: [
        "enable_colorfx", "enable_color_correction", "enable_lut_processing", "enable_enhancements",
        "enable_blur_effects", "enable_stylistic_effects"] },
    { btnName: "grp_cfx_color", label: "Color Correction", parent: "grp_cfx_main", enable: "enable_color_correction", widgets: [
        "tone_lock", "chroma_lock", "hdr_intensity", "shadow_intensity", "highlight_intensity", "gamma",
        "brightness", "contrast", "saturation", "enhance_color"] },
    { btnName: "grp_cfx_lut", label: "LUT Processing", parent: "grp_cfx_main", enable: "enable_lut_processing", widgets: [
        "lut_name", "lut_strength", "lut_log_process"] },
    { btnName: "grp_cfx_enhancements", label: "Enhancements", parent: "grp_cfx_main", enable: "enable_enhancements", widgets: [
        "sharpness", "edge_enhance_strength", "detail_enhance_strength"] },
    { btnName: "grp_cfx_blur", label: "Blur Effects", parent: "grp_cfx_main", enable: "enable_blur_effects", widgets: [
        "blur_radius", "gaussian_blur_radius", "radial_blur_strength", "radial_blur_center_x",
        "radial_blur_center_y", "radial_blur_focus_spread", "radial_blur_steps"] },
    { btnName: "grp_cfx_stylistic", label: "Stylistic Effects", parent: "grp_cfx_main", enable: "enable_stylistic_effects", widgets: [
        "chromatic_aberration_r_x", "chromatic_aberration_r_y", "chromatic_aberration_b_x",
        "chromatic_aberration_b_y", "chromatic_blur_amount", "simple_film_grain_intensity",
        "simple_film_grain_monochrome", "scanline_intensity", "vignette_intensity", "vignette_center_x",
        "vignette_center_y", "soft_light_opacity", "soft_light_blur_radius"] },
];

const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const stateProp = (def) => "groupState_" + def.btnName;

function toggleWidget(w, visible) {
    if (!w) return;
    if (w.type === "converted-widget") return;

    w.hidden = !visible;
    if (w.element) w.element.style.display = visible ? "" : "none";
    if (w.inputEl) w.inputEl.style.display = visible ? "" : "none";

    if (visible) {
        if (hasOwn(w, "origComputeSize")) {
            w.computeSize = w.origComputeSize;
        } else {
            delete w.computeSize;
        }
        if (hasOwn(w, "origTooltip")) w.tooltip = w.origTooltip;
    } else {
        if (!hasOwn(w, "origComputeSize")) {
            w.origComputeSize = hasOwn(w, "computeSize") ? w.computeSize : undefined;
        }
        if (!hasOwn(w, "origTooltip")) w.origTooltip = w.tooltip;
        w.computeSize = () => [0, 0];
        w.y = undefined;
        w.last_y = undefined;
        w.tooltip = null;
    }
}

function stateKey(node) {
    return GROUPS.map((d) => (node.properties[stateProp(d)] ? "1" : "0")).join("");
}

function remember(node, key, size) {
    node.properties.h3_sizes = node.properties.h3_sizes || {};
    node.properties.h3_sizes[key] = [size[0], size[1]];
}

// --- Connections collapse/expand, identical to film_auteur_h3_editor.js and screenplay_editor.js ---
// LiteGraph's computeSize counts every socket row, hidden or not, so the node reclaims the space itself.
function slotHeight() {
    const configured = typeof LiteGraph !== "undefined" ? Number(LiteGraph.NODE_SLOT_HEIGHT) : 0;
    return Number.isFinite(configured) && configured > 0 ? configured : 20;
}

function socketInputs(node) {
    return (node.inputs || []).filter((slot) => !slot.widget);
}

function applyCollapseState(node) {
    const collapsed = !!(node.properties && node.properties.collapse_connections);
    const sockets = socketInputs(node);
    const keepVisible = collapsed ? sockets[0] : null;

    (node.inputs || []).forEach((slot) => {
        // Widget inputs keep their rows: only real sockets collapse.
        if (slot.widget) return;
        slot.hidden = !!collapsed && slot !== keepVisible;
    });
    (node.outputs || []).forEach((slot, index) => {
        slot.hidden = !!collapsed && index > 0;
    });
}

function widgetContentFloor(node) {
    let bottom = 0;
    for (const widget of node.widgets || []) {
        const y = Number(widget.last_y ?? widget.y);
        const height = typeof widget.computeSize === "function"
            ? Number((widget.computeSize(node.size?.[0]) || [])[1])
            : 0;
        if (Number.isFinite(y) && y >= 0 && Number.isFinite(height) && height > 0) {
            bottom = Math.max(bottom, y + height);
        }
    }
    return bottom > 0 ? Math.ceil(bottom + 10) : 0;
}

function syncLayout(node) {
    if (typeof node.computeSize !== "function") return;
    const minSize = node.computeSize();
    if (!Array.isArray(minSize)) return;
    const width = Math.max(Number(node.size?.[0]) || 0, Number(minSize[0]) || 0);
    const height = Math.max(Number(minSize[1]) || 0, widgetContentFloor(node));
    if (height <= 0) return;
    node.setSize([width || 340, height]);
    app.graph?.setDirtyCanvas(true, true);
}

function scheduleLayout(node) {
    // Let LiteGraph apply the hidden-slot state before measuring; measuring synchronously can read
    // the previous connection-row height.
    requestAnimationFrame(() => {
        requestAnimationFrame(() => syncLayout(node));
    });
}

function toggleConnections(node) {
    if (node.__h3Busy) return;
    node.properties = node.properties || {};
    node.properties.collapse_connections = !node.properties.collapse_connections;
    applyCollapseState(node);
    scheduleLayout(node);
}

function installSizing(node) {
    if (node.__h3CollapseSizing || typeof node.computeSize !== "function") return;
    const originalComputeSize = node.computeSize.bind(node);
    node.__h3CollapseSizing = true;
    node.computeSize = function (...args) {
        const size = originalComputeSize(...args);
        if (!Array.isArray(size) || !this.properties?.collapse_connections) return size;

        const sockets = socketInputs(this);
        const outputs = this.outputs || [];
        const totalRows = Math.max(sockets.length, outputs.length, 1);
        const visibleRows = Math.max(
            sockets.filter((slot) => !slot.hidden).length,
            outputs.filter((slot) => !slot.hidden).length,
            1
        );
        const hiddenRows = Math.max(0, totalRows - visibleRows);
        if (!hiddenRows) return size;

        const floor = widgetContentFloor(this);
        return [size[0], Math.max(size[1] - hiddenRows * slotHeight(), floor)];
    };
}

// pixel_batch_overlap used to sit at the end of the node's widget list, below every Color FX
// control, because it was added there and never moved. It now sits inside the H3 Upscale group, so
// every workflow saved before the move has its widgets_values one step out from that slot onwards:
// the overlap value is still the last one, and everything from the slot down is where the widget
// after it expects to be. Put them back once per file - `h3_overlap_in_group` marks a file that is
// already in the new order, so a workflow saved by this version is never migrated twice.
// Workflows saved before pixel_batch_frames was removed carry one more value after the overlap;
// `extra` is that count.
function migrateOverlapValues(values, slot, extra = 0) {
    if (!Array.isArray(values) || slot <= 0) return values;
    const overlapAt = values.length - 1 - Math.max(0, extra);
    if (overlapAt <= slot) return values;
    return values.slice(0, slot).concat([values[overlapAt]], values.slice(slot, overlapAt));
}

app.registerExtension({
    name: "triXope.H3Upscale.Groups",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const inp = (nodeData && nodeData.input) || {};
        const declared = Object.assign({}, inp.required || {}, inp.optional || {});
        if (!("grp_h3_upscale" in declared)) return;   // not this node (name-independent check)

        // --- keep the helper buttons out of widgets_values (the dummy widgets keep the saved slots) ---
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            if (onSerialize) onSerialize.apply(this, arguments);
            // The values written below are in the moved order: record that, so loading this file
            // again does not shift them a second time.
            this.properties = this.properties || {};
            this.properties.h3_overlap_in_group = true;
            if (o) {
                o.properties = o.properties || {};
                o.properties.h3_overlap_in_group = true;
            }
            if (this.widgets && o.widgets_values) {
                const clean = [];
                for (let i = 0; i < this.widgets.length; i++) {
                    if (!this.widgets[i].isCustomGrouperBtn) clean.push(o.widgets_values[i]);
                }
                o.widgets_values = clean;
            }
        };

        // --- remember the size that was saved with the workflow (and that this node was loaded, not new) ---
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            if (onConfigure) onConfigure.apply(this, arguments);
            this._h3Loaded = true;
            this.properties = this.properties || {};
            if (o && Array.isArray(o.widgets_values) && !this.properties.h3_overlap_in_group
                    && this.widgets) {
                const slot = this.widgets.findIndex((w) => w.name === "pixel_batch_overlap");
                // In the new order this slot already holds the overlap itself, a number; in the
                // old order it holds the group header that used to sit there, which is not one.
                // A workflow saved by this version carries the marker instead, above.
                if (slot > 0 && typeof o.widgets_values[slot] !== "number") {
                    const extra = Math.max(0, o.widgets_values.length - this.widgets.length);
                    const fixed = migrateOverlapValues(o.widgets_values, slot, extra);
                    if (fixed !== o.widgets_values) {
                        for (let i = 0; i < this.widgets.length && i < fixed.length; i++) {
                            if (!this.widgets[i].isCustomGrouperBtn) this.widgets[i].value = fixed[i];
                        }
                        o.widgets_values = fixed;
                    }
                    this.properties.h3_overlap_in_group = true;
                }
            }
            if (o && o.size) this._true_saved_size = [o.size[0], o.size[1]];
            applyCollapseState(this);
            if (this.__h3Ready && this._true_saved_size) {
                // configure() arrived after the initial layout pass: restore the saved size now.
                this.__h3Busy = true;
                this.setSize([Math.max(this._true_saved_size[0], 340),
                              Math.max(this.size[1], this._true_saved_size[1])]);
                this.__h3Busy = false;
                delete this._true_saved_size;
                remember(this, stateKey(this), this.size);
            }
            scheduleLayout(this);
        };

        // --- remember manual resizes for the current open/closed combination ---
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            if (onResize) onResize.apply(this, arguments);
            if (this.__h3Ready && !this.__h3Busy && this.properties) {
                const current = size || this.size;
                remember(this, stateKey(this), current);
                // The width is a property of the node, not of a group state.
                this.properties.h3_width = Math.round(current[0]) || this.properties.h3_width;
            }
        };

        // --- collapse/expand the connection sockets (same behaviour as the Film Auteur nodes) ---
        const oldGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            if (oldGetExtraMenuOptions) oldGetExtraMenuOptions.apply(this, arguments);
            if (!options) return;

            const connectionOption = options.find(
                (option) => option && typeof option.content === "string" && /connections/i.test(option.content)
            );
            if (connectionOption) {
                if (!connectionOption.__h3ResizeWrapped && typeof connectionOption.callback === "function") {
                    const originalCallback = connectionOption.callback;
                    const targetNode = this;
                    connectionOption.callback = function (...args) {
                        const result = originalCallback.apply(this, args);
                        scheduleLayout(targetNode);
                        return result;
                    };
                    connectionOption.__h3ResizeWrapped = true;
                }
                return;
            }

            options.push(null); // Separator
            options.push({
                content: this.properties?.collapse_connections
                    ? "🟢 Expand Connections"
                    : "▶ Collapse Connections",
                callback: () => toggleConnections(this),
            });
        };

        const oldGetInputPos = nodeType.prototype.getInputPos;
        nodeType.prototype.getInputPos = function (slotNumber) {
            const fn = oldGetInputPos || this.constructor?.prototype?.getInputPos || LiteGraph.LGraphNode.prototype.getInputPos;
            const isCollapsed = this.properties && this.properties.collapse_connections;
            return fn.call(this, isCollapsed ? 0 : slotNumber);
        };

        const oldGetOutputPos = nodeType.prototype.getOutputPos;
        nodeType.prototype.getOutputPos = function (slotNumber) {
            const fn = oldGetOutputPos || this.constructor?.prototype?.getOutputPos || LiteGraph.LGraphNode.prototype.getOutputPos;
            const isCollapsed = this.properties && this.properties.collapse_connections;
            return fn.call(this, isCollapsed ? 0 : slotNumber);
        };

        const oldGetConnectionPos = nodeType.prototype.getConnectionPos;
        nodeType.prototype.getConnectionPos = function (isInput, slotNumber, out) {
            const fn = oldGetConnectionPos || this.constructor?.prototype?.getConnectionPos || LiteGraph.LGraphNode.prototype.getConnectionPos;
            const isCollapsed = this.properties && this.properties.collapse_connections;
            return fn.call(this, isInput, isCollapsed ? 0 : slotNumber, out);
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.properties = this.properties || {};

            // install before the first layout pass so a restored "collapsed" state is measured correctly
            installSizing(this);
            applyCollapseState(this);

            const refreshGroupVisibility = () => {
                const oldKey = this.__h3Key;
                if (oldKey) remember(this, oldKey, this.size);        // size of the state we are leaving
                const oldMinY = this.computeSize()[1];

                for (const def of GROUPS) {
                    // a subgroup's header and contents only show while its parent group is expanded
                    const parentOpen = def.parent ? !!this.properties["groupState_" + def.parent] : true;
                    const open = !!this.properties[stateProp(def)];
                    if (def.parent) {
                        toggleWidget(this.widgets.find((w) => w.originalName === def.btnName), parentOpen);
                    }
                    for (const wName of def.widgets) {
                        toggleWidget(this.widgets.find((w) => w.name === wName), parentOpen && open);
                    }
                }

                const newKey = stateKey(this);
                this.__h3Key = newKey;
                const newMin = this.computeSize();
                const saved = this.properties.h3_sizes && this.properties.h3_sizes[newKey];
                // Width comes from the node, never from the per-state table: a node saved with every
                // group collapsed used to restore the narrow width it had while collapsed.
                const targetW = Math.max(this.size[0], newMin[0]);
                const targetH = saved
                    ? Math.max(saved[1], newMin[1])
                    : Math.max(newMin[1], this.size[1] + (newMin[1] - oldMinY));
                this.__h3Busy = true;
                this.setSize([targetW, targetH]);
                this.__h3Busy = false;
                if (!saved) remember(this, newKey, this.size);
                app.graph.setDirtyCanvas(true, true);
            };

            setTimeout(() => {
                const tempCtx = document.createElement("canvas").getContext("2d");
                tempCtx.font = LiteGraph.WIDGET_TEXT_FONT || "12px Arial";
                let longestLabelWidth = 0;
                for (const d of GROUPS) {
                    const w = tempCtx.measureText("\u25BC " + d.label + "   \u25CF ON").width;
                    if (w > longestLabelWidth) longestLabelWidth = w;
                }

                for (const def of GROUPS) {
                    const dummyIndex = this.widgets.findIndex((w) => w.name === def.btnName);
                    if (dummyIndex === -1) continue;
                    toggleWidget(this.widgets[dummyIndex], false);   // hide the dummy, keep its saved-value slot

                    const propKey = stateProp(def);
                    if (this.properties[propKey] === undefined) this.properties[propKey] = false;   // collapsed by default

                    const btn = this.addWidget("button", def.label, null, () => {
                        this.properties[propKey] = !this.properties[propKey];
                        refreshGroupVisibility();
                    });
                    btn.isCustomGrouperBtn = true;
                    btn.originalName = def.btnName;

                    btn.draw = (ctx, node, widget_width, y, H) => {
                        const open = !!node.properties[propKey];
                        const isSub = def.parent !== undefined;
                        const indent = isSub ? 20 : 15;                  // subgroups are indented
                        const fillWidth = widget_width - indent - 15;
                        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#222";
                        ctx.fillRect(indent, y, fillWidth, H);
                        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#AFAFAF";
                        ctx.font = LiteGraph.WIDGET_TEXT_FONT || "12px Arial";
                        ctx.textAlign = "left";
                        let suffix = "";
                        if (def.enable) {
                            const en = node.widgets.find((w) => w.name === def.enable);
                            if (en) {
                                // h3_upscale_mode is a mode combo: only "off" means bypassed
                                const on = typeof en.value === "string" ? en.value !== "off" : !!en.value;
                                suffix = on ? "   \u25CF ON" : "   \u25CB OFF";
                            }
                        }
                        const x_start = (widget_width / 2) - (longestLabelWidth / 2) + (isSub ? 10 : 0);
                        ctx.fillText((open ? "\u25BC " : "\u25B6 ") + def.label + suffix, x_start, y + H * 0.7);
                    };

                    this.widgets.pop();
                    this.widgets.splice(dummyIndex, 0, btn);
                }

                // a brand-new node (not loaded from a workflow) starts with the seed control on "fixed"
                if (!this._h3Loaded) {
                    const ctl = this.widgets.find((w) => w.name === "control_after_generate");
                    if (ctl) ctl.value = "fixed";
                }

                refreshGroupVisibility();

                // Final size on load: the size saved with the workflow, else the remembered width and the
                // compact height of this state. The width is restored even when every group is collapsed.
                const savedSize = this._true_saved_size;
                const storedWidth = Number(this.properties.h3_width);
                const finalWidth = savedSize
                    ? savedSize[0]
                    : (Number.isFinite(storedWidth) && storedWidth > 0 ? storedWidth : this.size[0]);
                const finalHeight = savedSize ? savedSize[1] : this.computeSize()[1];
                delete this._true_saved_size;
                this.__h3Busy = true;
                this.setSize([Math.max(finalWidth, 340), Math.max(finalHeight, this.computeSize()[1])]);
                this.__h3Busy = false;
                remember(this, stateKey(this), this.size);
                this.__h3Ready = true;
                applyCollapseState(this);
                app.graph.setDirtyCanvas(true, true);
            }, 250);

            return r;
        };
    },
});
