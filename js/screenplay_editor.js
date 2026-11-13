import { app } from "../../scripts/app.js";

// Helper to toggle connection slot visibility and update node dimensions
function applyCollapseState(node) {
    const isCollapsed = !!(node.properties && node.properties.collapse_connections);

    if (node.inputs) {
        for (let i = 0; i < node.inputs.length; i++) {
            node.inputs[i].hidden = (i > 0 && isCollapsed);
        }
    }
    if (node.outputs) {
        for (let i = 0; i < node.outputs.length; i++) {
            node.outputs[i].hidden = (i > 0 && isCollapsed);
        }
    }
}

function toggleConnections(node) {
    node.properties = node.properties || {};
    node.properties.collapse_connections = !node.properties.collapse_connections;

    applyCollapseState(node);

    if (typeof node.__screenplaySyncConnectionLayout === "function") {
        node.__screenplaySyncConnectionLayout();
    } else {
        const minSize = node.computeSize();
        node.setSize([node.size[0], minSize[1]]);
        app.graph.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "Comfy.ScreenplayFormatter",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "triXope ScreenplayFormatterNode") {

            // --- 1. CONTEXT MENU HOOK ---
            const oldGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
                if (oldGetExtraMenuOptions) oldGetExtraMenuOptions.apply(this, arguments);

                if (!options) return;
                const connectionOption = options.find(o => o && typeof o.content === "string" && /connections/i.test(o.content));
                if (connectionOption) {
                    if (!connectionOption.__screenplayResizeWrapped && typeof connectionOption.callback === "function") {
                        const originalCallback = connectionOption.callback;
                        const targetNode = this;
                        connectionOption.callback = function (...args) {
                            const result = originalCallback.apply(this, args);
                            requestAnimationFrame(() => {
                                targetNode.__screenplaySyncConnectionLayout?.();
                                app.graph?.setDirtyCanvas(true, true);
                            });
                            return result;
                        };
                        connectionOption.__screenplayResizeWrapped = true;
                    }
                    return;
                }

                options.push(null); // Separator
                options.push({
                    content: this.properties?.collapse_connections ? "🟢 Expand Connections" : "▶ Collapse Connections",
                    callback: () => {
                        toggleConnections(this);
                    }
                });
            };

            // --- 2. CONNECTION WIRE POSITION HOOKS ---
            const oldGetInputPos = nodeType.prototype.getInputPos;
            nodeType.prototype.getInputPos = function(slotNumber) {
                const fn = oldGetInputPos || this.constructor?.prototype?.getInputPos || LiteGraph.LGraphNode.prototype.getInputPos;
                const isCollapsed = this.properties && this.properties.collapse_connections;
                return fn.call(this, isCollapsed ? 0 : slotNumber);
            };

            const oldGetOutputPos = nodeType.prototype.getOutputPos;
            nodeType.prototype.getOutputPos = function(slotNumber) {
                const fn = oldGetOutputPos || this.constructor?.prototype?.getOutputPos || LiteGraph.LGraphNode.prototype.getOutputPos;
                const isCollapsed = this.properties && this.properties.collapse_connections;
                return fn.call(this, isCollapsed ? 0 : slotNumber);
            };

            const oldGetConnectionPos = nodeType.prototype.getConnectionPos;
            nodeType.prototype.getConnectionPos = function(isInput, slotNumber, out) {
                const fn = oldGetConnectionPos || this.constructor?.prototype?.getConnectionPos || LiteGraph.LGraphNode.prototype.getConnectionPos;
                const isCollapsed = this.properties && this.properties.collapse_connections;
                return fn.call(this, isInput, isCollapsed ? 0 : slotNumber, out);
            };
        }
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "triXope ScreenplayFormatterNode") return;

        node.properties = node.properties || {};
        console.log("[ScreenplayFormatter] Initializing Screenplay Formatter custom node extensions...");

        // Keep LiteGraph's slot arrays and indices untouched. Its size calculation
        // still counts hidden connection rows, so subtract only the rows hidden
        // by this node's Connections toggle from the reported minimum height.
        if (!node.__screenplayVisibleSlotSizing && typeof node.computeSize === "function") {
            const originalComputeSize = node.computeSize.bind(node);
            node.__screenplayVisibleSlotSizing = true;
            node.computeSize = function (...args) {
                const size = originalComputeSize(...args);
                if (!this.properties?.collapse_connections || !Array.isArray(size)) return size;

                const inputSlots = this.inputs || [];
                const outputSlots = this.outputs || [];
                const totalRows = Math.max(inputSlots.length, outputSlots.length);
                const visibleInputRows = inputSlots.filter(slot => !slot.hidden).length;
                const visibleOutputRows = outputSlots.filter(slot => !slot.hidden).length;
                const visibleRows = Math.max(visibleInputRows, visibleOutputRows);
                const hiddenRows = Math.max(0, totalRows - visibleRows);
                const configuredSlotHeight = typeof LiteGraph !== "undefined"
                    ? Number(LiteGraph.NODE_SLOT_HEIGHT)
                    : 0;
                const slotHeight = Number.isFinite(configuredSlotHeight) && configuredSlotHeight > 0
                    ? configuredSlotHeight
                    : 20;

                if (!hiddenRows) return size;
                const adjustedHeight = Math.max(0, size[1] - hiddenRows * slotHeight);
                const contentMinimum = typeof this.__screenplayContentMinHeight === "function"
                    ? this.__screenplayContentMinHeight()
                    : 0;
                return [size[0], Math.max(adjustedHeight, contentMinimum)];
            };
        }

        const getWidget = (name) => node.widgets?.find(w => w.name === name);

        // The style_preset combo carries a one-line description per entry in its widget
        // options (built by the Python node from each preset's opening sentence), so the
        // chosen look is explained right under the dropdown instead of needing a prompt read.
        const stylePresetWidget = getWidget("style_preset");
        const stylePresetDescriptions = stylePresetWidget?.options?.preset_descriptions || {};
        const stylePresetInfo = document.createElement("div");
        stylePresetInfo.style.fontSize = "10px";
        stylePresetInfo.style.fontStyle = "italic";
        stylePresetInfo.style.lineHeight = "1.35";
        stylePresetInfo.style.color = "#9aa0a6";
        stylePresetInfo.style.whiteSpace = "normal";
        stylePresetInfo.style.overflowWrap = "anywhere";
        stylePresetInfo.style.padding = "2px 2px 4px";

        const updateStylePresetInfo = () => {
            const text = stylePresetDescriptions[stylePresetWidget?.value] || "";
            stylePresetInfo.textContent = text;
            stylePresetInfo.style.display = text ? "" : "none";
            if (stylePresetWidget?.inputEl) stylePresetWidget.inputEl.title = text;
        };

        if (stylePresetWidget) {
            const previousStyleCallback = stylePresetWidget.callback;
            stylePresetWidget.callback = function (...args) {
                const result = previousStyleCallback?.apply(this, args);
                updateStylePresetInfo();
                return result;
            };
            stylePresetWidget.inputEl?.addEventListener("change", updateStylePresetInfo);
            updateStylePresetInfo();
        }

        // Hide JSON fields visually while preserving LiteGraph index mapping
        const hideWidget = (name) => {
            const w = getWidget(name);
            if (w) {
                w.type = "hidden";
                w.hidden = true;
                w.computeSize = () => [0, -4];
                w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
                w.options = w.options || {};
                w.options.getMinHeight = () => 0;
                w.options.getMaxHeight = () => 0;
                if (w.inputEl) {
                    if (w.inputEl.parentElement) w.inputEl.parentElement.style.display = "none";
                    w.inputEl.style.display = "none";
                }
            }
            return w;
        };

        const charJsonWidget = hideWidget("characters_json");
        const locJsonWidget = hideWidget("locations_json");
        const propJsonWidget = hideWidget("props_json");

        const settingLabels = {
            use_ollama_ai: "Enable Ollama AI",
            ollama_model: "Model (Ollama / Local)",
        };

        const settingNames = [
            "auto_capitalize_headers",
            "output_format",
            "default_aspect_ratio",
            "use_ollama_ai",
            "ollama_model",
            "ollama_url",
            "script_override_mode",
            "trigger_words",
            "lock_last_generation"
        ];
        settingNames.forEach(name => hideWidget(name));
        hideWidget("stored_generation");

        let characters = [];
        let props = [];
        let locations = [];
        let isRestoring = false;

        const syncData = () => {
            if (isRestoring) return;
            if (charJsonWidget) charJsonWidget.value = JSON.stringify(characters);
            if (locJsonWidget) locJsonWidget.value = JSON.stringify(locations);
            if (propJsonWidget) propJsonWidget.value = JSON.stringify(props);
            renderQuickBars();
            console.log(`[ScreenplayFormatter] Synchronized JSON state (Characters: ${characters.length}, Props: ${props.length}, Locations: ${locations.length})`);
        };

        // Must match _LOCAL_LLM_MODELS in nodes_writing.py. Selecting one downloads it
        // on first run and runs it through ComfyUI's own text encoder, with no Ollama.
        const localModelNames = [
            "Local: Gemma 4 E4B (fp8, 9.1 GB)",
            "Local: Gemma 4 12B (int8, 12.1 GB)",
            "Local: Gemma 4 E2B (int8, 5.2 GB)",
            "Local: Qwen3.5 4B (bf16, 9.3 GB)",
            "Local: Qwen3.5 9B (int8, 9.8 GB)",
        ];

        // Query installed local Ollama models dynamically
        const populateOllamaModels = async () => {
            const modelWidget = getWidget("ollama_model");
            const urlWidget = getWidget("ollama_url");
            if (!modelWidget) return;

            const url = (urlWidget?.value || "http://127.0.0.1:11434").replace(/\/$/, "");
            console.log(`[ScreenplayFormatter] Fetching local Ollama models from ${url}/api/tags...`);
            try {
                const response = await fetch(`${url}/api/tags`);
                if (!response.ok) throw new Error("Offline");
                const data = await response.json();
                const modelNames = data.models.map(m => m.name);

                if (modelNames.length > 0) {
                    console.log(`[ScreenplayFormatter] Ollama models found: ${modelNames.join(", ")}`);
                    const choices = [...modelNames, ...localModelNames];
                    modelWidget.options.values = choices;
                    if (!choices.includes(modelWidget.value)) {
                        modelWidget.value = modelNames[0];
                    }
                } else {
                    console.warn("[ScreenplayFormatter] Ollama endpoint connected, but zero models were reported.");
                    modelWidget.options.values = ["None / Ollama Offline", ...localModelNames];
                    if (!localModelNames.includes(modelWidget.value)) {
                        modelWidget.value = "None / Ollama Offline";
                    }
                }
            } catch (e) {
                console.warn("[ScreenplayFormatter] Ollama is offline or unreachable:", e.message);
                modelWidget.options.values = ["None / Ollama Offline", ...localModelNames];
                if (!localModelNames.includes(modelWidget.value)) {
                    modelWidget.value = "None / Ollama Offline";
                }
            }
            renderSettings();
        };

        // Collapsible UI Factory
        const createCollapsible = (titleText, defaultOpen = false, stateKey = null) => {
            const outer = document.createElement("div");
            outer.style.border = "1px solid #333";
            outer.style.borderRadius = "4px";
            outer.style.background = "#181818";
            outer.style.overflow = "hidden";
            outer.style.marginTop = "4px";
            outer.style.boxSizing = "border-box";
            outer.style.width = "100%";
            outer.style.minWidth = "0";

            const header = document.createElement("div");
            header.style.padding = "5px 8px";
            header.style.cursor = "pointer";
            header.style.fontWeight = "bold";
            header.style.fontSize = "11px";
            header.style.color = "#ccc";

            const savedExpanded = stateKey
                ? node.properties?.screenplay_ui_state?.groups?.[stateKey]
                : undefined;
            const initiallyExpanded = typeof savedExpanded === "boolean" ? savedExpanded : defaultOpen;
            const titleSpan = document.createElement("span");
            const body = document.createElement("div");
            let isExpanded = initiallyExpanded;

            const updateTitle = () => {
                const countText = titleText.includes("Character") ? ` (${characters.length})` :
                                  titleText.includes("Prop") ? ` (${props.length})` :
                                  titleText.includes("Location") ? ` (${locations.length})` : "";
                titleSpan.innerText = `${isExpanded ? "▼" : "▶"} ${titleText.split(' (')[0]}${countText}`;
            };
            const setExpanded = (expanded) => {
                isExpanded = !!expanded;
                body.style.display = isExpanded ? "flex" : "none";
                updateTitle();
            };

            header.appendChild(titleSpan);
            body.style.padding = "6px";
            body.style.flexDirection = "column";
            body.style.gap = "5px";
            body.style.background = "#111";
            body.style.boxSizing = "border-box";
            body.style.minWidth = "0";
            setExpanded(initiallyExpanded);

            header.addEventListener("click", () => {
                setExpanded(!isExpanded);
                if (stateKey) {
                    const savedState = node.properties.screenplay_ui_state || {};
                    const groups = savedState.groups || {};
                    node.properties.screenplay_ui_state = {
                        ...savedState,
                        groups: { ...groups, [stateKey]: isExpanded }
                    };
                }
                outer.dispatchEvent(new Event("screenplay-group-toggle"));
            });

            outer.appendChild(header);
            outer.appendChild(body);
            return { outer, body, titleSpan, header, setExpanded, isExpanded: () => isExpanded };
        };

        // --- SETTINGS MANAGER ---
        const settingsGroup = createCollapsible("Settings", false, "settings");
        const settingsContainer = document.createElement("div");
        settingsContainer.style.display = "flex";
        settingsContainer.style.flexDirection = "column";
        settingsContainer.style.gap = "4px";

        const renderSettings = () => {
            settingsContainer.innerHTML = "";
            settingNames.forEach(name => {
                const w = getWidget(name);
                if (!w) return;

                const row = document.createElement("div");
                row.style.display = "flex";
                row.style.alignItems = "center";
                row.style.justifyContent = "space-between";
                row.style.gap = "6px";

                const label = document.createElement("span");
                label.style.fontSize = "10px";
                label.style.color = "#ccc";
                label.style.flex = "1 1 auto";
                label.innerText = settingLabels[name] || name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

                let input;
                if (typeof w.value === "boolean") {
                    input = document.createElement("input");
                    input.type = "checkbox";
                    input.checked = !!w.value;
                    input.style.cursor = "pointer";
                    input.addEventListener("change", (e) => {
                        w.value = e.target.checked;
                        if (w.callback) w.callback(w.value);
                    });
                } else if (w.options?.values && Array.isArray(w.options.values)) {
                    input = document.createElement("select");
                    input.style.fontSize = "10px";
                    input.style.background = "#222";
                    input.style.color = "#fff";
                    input.style.border = "1px solid #444";
                    input.style.borderRadius = "3px";
                    input.style.padding = "2px 4px";
                    input.style.flex = "0 0 55%";
                    input.style.minWidth = "0";
                    w.options.values.forEach(opt => {
                        const o = document.createElement("option");
                        o.value = opt;
                        o.innerText = opt;
                        if (w.value === opt) o.selected = true;
                        input.appendChild(o);
                    });
                    input.addEventListener("change", (e) => {
                        w.value = e.target.value;
                        if (w.callback) w.callback(w.value);
                    });
                } else {
                    input = document.createElement("input");
                    input.type = "text";
                    input.value = w.value || "";
                    input.style.fontSize = "10px";
                    input.style.background = "#222";
                    input.style.color = "#fff";
                    input.style.border = "1px solid #444";
                    input.style.borderRadius = "3px";
                    input.style.padding = "2px 4px";
                    input.style.flex = "0 0 55%";
                    input.style.minWidth = "0";
                    input.style.boxSizing = "border-box";
                    input.addEventListener("input", (e) => {
                        w.value = e.target.value;
                        if (w.callback) w.callback(w.value);
                    });
                }

                row.appendChild(label);
                row.appendChild(input);
                settingsContainer.appendChild(row);
            });
        };

        settingsGroup.body.appendChild(settingsContainer);

        // --- CHARACTERS MANAGER ---
        const charGroup = createCollapsible("Characters (0)", false, "characters");
        const charContainer = document.createElement("div");
        charContainer.style.display = "flex";
        charContainer.style.flexDirection = "column";
        charContainer.style.gap = "4px";

        const addCharBtn = document.createElement("button");
        addCharBtn.innerText = "+ Add Character";
        addCharBtn.style.padding = "3px 6px";
        addCharBtn.style.fontSize = "10px";
        addCharBtn.style.cursor = "pointer";
        addCharBtn.style.background = "#2d5a27";
        addCharBtn.style.color = "#fff";
        addCharBtn.style.border = "none";
        addCharBtn.style.borderRadius = "3px";

        const charQuickBar = document.createElement("div");
        charQuickBar.style.display = "flex";
        charQuickBar.style.flexWrap = "wrap";
        charQuickBar.style.gap = "6px";
        charQuickBar.style.marginTop = "4px";

        const renderCharacters = () => {
            charContainer.innerHTML = "";
            characters.forEach((char, index) => {
                const row = document.createElement("div");
                row.style.display = "flex";
                row.style.gap = "4px";

                const nameInput = document.createElement("input");
                nameInput.placeholder = "Character Name";
                nameInput.value = char.name || "";
                nameInput.style.flex = "0 0 30%";
                nameInput.style.minWidth = "0";
                nameInput.style.boxSizing = "border-box";
                nameInput.style.fontSize = "10px";
                nameInput.style.background = "#222";
                nameInput.style.color = "#fff";
                nameInput.style.border = "1px solid #444";
                nameInput.addEventListener("input", (e) => {
                    characters[index].name = e.target.value;
                    syncData();
                });

                const descInput = document.createElement("input");
                descInput.placeholder = "Description";
                descInput.value = char.description || "";
                descInput.style.flex = "1 1 0";
                descInput.style.minWidth = "0";
                descInput.style.boxSizing = "border-box";
                descInput.style.fontSize = "10px";
                descInput.style.background = "#222";
                descInput.style.color = "#fff";
                descInput.style.border = "1px solid #444";
                descInput.addEventListener("input", (e) => {
                    characters[index].description = e.target.value;
                    syncData();
                });

                const removeBtn = document.createElement("button");
                removeBtn.innerText = "✕";
                removeBtn.style.flex = "0 0 auto";
                removeBtn.style.background = "#6c757d";
                removeBtn.style.color = "#fff";
                removeBtn.style.border = "none";
                removeBtn.style.cursor = "pointer";
                removeBtn.addEventListener("click", () => {
                    characters.splice(index, 1);
                    renderCharacters();
                    syncData();
                });

                row.appendChild(nameInput);
                row.appendChild(descInput);
                row.appendChild(removeBtn);
                charContainer.appendChild(row);
            });
            charGroup.titleSpan.innerText = `${charGroup.body.style.display === "none" ? "▶" : "▼"} Characters (${characters.length})`;
        };

        addCharBtn.addEventListener("click", () => {
            characters.push({ name: "", description: "" });
            renderCharacters();
            syncData();
        });

        charGroup.body.appendChild(charContainer);
        charGroup.body.appendChild(addCharBtn);
        charGroup.body.appendChild(charQuickBar);

        // --- PROPS MANAGER ---
        const propGroup = createCollapsible("Props (0)", false, "props");
        const propContainer = document.createElement("div");
        propContainer.style.display = "flex";
        propContainer.style.flexDirection = "column";
        propContainer.style.gap = "4px";

        const addPropBtn = document.createElement("button");
        addPropBtn.innerText = "+ Add Prop";
        addPropBtn.style.padding = "3px 6px";
        addPropBtn.style.fontSize = "10px";
        addPropBtn.style.cursor = "pointer";
        addPropBtn.style.background = "#2d5a27";
        addPropBtn.style.color = "#fff";
        addPropBtn.style.border = "none";
        addPropBtn.style.borderRadius = "3px";

        const propQuickBar = document.createElement("div");
        propQuickBar.style.display = "flex";
        propQuickBar.style.flexWrap = "wrap";
        propQuickBar.style.gap = "6px";
        propQuickBar.style.marginTop = "4px";

        const renderProps = () => {
            propContainer.innerHTML = "";
            props.forEach((prop, index) => {
                const row = document.createElement("div");
                row.style.display = "flex";
                row.style.gap = "4px";

                const nameInput = document.createElement("input");
                nameInput.placeholder = "Prop Name";
                nameInput.value = prop.name || "";
                nameInput.style.flex = "0 0 30%";
                nameInput.style.minWidth = "0";
                nameInput.style.boxSizing = "border-box";
                nameInput.style.fontSize = "10px";
                nameInput.style.background = "#222";
                nameInput.style.color = "#fff";
                nameInput.style.border = "1px solid #444";
                nameInput.addEventListener("input", (e) => {
                    props[index].name = e.target.value;
                    syncData();
                });

                const descInput = document.createElement("input");
                descInput.placeholder = "Description";
                descInput.value = prop.description || "";
                descInput.style.flex = "1 1 0";
                descInput.style.minWidth = "0";
                descInput.style.boxSizing = "border-box";
                descInput.style.fontSize = "10px";
                descInput.style.background = "#222";
                descInput.style.color = "#fff";
                descInput.style.border = "1px solid #444";
                descInput.addEventListener("input", (e) => {
                    props[index].description = e.target.value;
                    syncData();
                });

                const removeBtn = document.createElement("button");
                removeBtn.innerText = "✕";
                removeBtn.style.flex = "0 0 auto";
                removeBtn.style.background = "#6c757d";
                removeBtn.style.color = "#fff";
                removeBtn.style.border = "none";
                removeBtn.style.cursor = "pointer";
                removeBtn.addEventListener("click", () => {
                    props.splice(index, 1);
                    renderProps();
                    syncData();
                });

                row.appendChild(nameInput);
                row.appendChild(descInput);
                row.appendChild(removeBtn);
                propContainer.appendChild(row);
            });
            propGroup.titleSpan.innerText = `${propGroup.body.style.display === "none" ? "▶" : "▼"} Props (${props.length})`;
        };

        addPropBtn.addEventListener("click", () => {
            props.push({ name: "", description: "" });
            renderProps();
            syncData();
        });

        propGroup.body.appendChild(propContainer);
        propGroup.body.appendChild(addPropBtn);
        propGroup.body.appendChild(propQuickBar);

        // --- LOCATIONS MANAGER ---
        const locGroup = createCollapsible("Locations (0)", false, "locations");
        const locContainer = document.createElement("div");
        locContainer.style.display = "flex";
        locContainer.style.flexDirection = "column";
        locContainer.style.gap = "4px";

        const addLocBtn = document.createElement("button");
        addLocBtn.innerText = "+ Add Location";
        addLocBtn.style.padding = "3px 6px";
        addLocBtn.style.fontSize = "10px";
        addLocBtn.style.cursor = "pointer";
        addLocBtn.style.background = "#2d5a27";
        addLocBtn.style.color = "#fff";
        addLocBtn.style.border = "none";
        addLocBtn.style.borderRadius = "3px";

        const locQuickBar = document.createElement("div");
        locQuickBar.style.display = "flex";
        locQuickBar.style.flexWrap = "wrap";
        locQuickBar.style.gap = "6px";
        locQuickBar.style.marginTop = "4px";

        const intExtOptions = ["INT", "EXT"];
        const timeOptions = ["DAY", "NIGHT", "MORNING", "EVENING", "DUSK", "DAWN", "CONTINUOUS", "LATER"];

        const renderLocations = () => {
            locContainer.innerHTML = "";
            locations.forEach((loc, index) => {
                const row = document.createElement("div");
                row.style.display = "flex";
                row.style.gap = "4px";

                const ieSelect = document.createElement("select");
                ieSelect.style.fontSize = "10px";
                ieSelect.style.flex = "0 0 auto";
                ieSelect.style.minWidth = "0";
                ieSelect.style.background = "#222";
                ieSelect.style.color = "#fff";
                ieSelect.style.border = "1px solid #444";
                intExtOptions.forEach(opt => {
                    const o = document.createElement("option");
                    o.value = opt; o.innerText = opt;
                    if (loc.int_ext === opt) o.selected = true;
                    ieSelect.appendChild(o);
                });
                ieSelect.addEventListener("change", (e) => {
                    locations[index].int_ext = e.target.value;
                    syncData();
                });

                const nameInput = document.createElement("input");
                nameInput.placeholder = "Scene Name";
                nameInput.value = loc.name || "";
                nameInput.style.flex = "1 1 0";
                nameInput.style.minWidth = "0";
                nameInput.style.boxSizing = "border-box";
                nameInput.style.fontSize = "10px";
                nameInput.style.background = "#222";
                nameInput.style.color = "#fff";
                nameInput.style.border = "1px solid #444";
                nameInput.addEventListener("input", (e) => {
                    locations[index].name = e.target.value;
                    syncData();
                });

                const todSelect = document.createElement("select");
                todSelect.style.fontSize = "10px";
                todSelect.style.flex = "0 0 auto";
                todSelect.style.minWidth = "0";
                todSelect.style.background = "#222";
                todSelect.style.color = "#fff";
                todSelect.style.border = "1px solid #444";
                timeOptions.forEach(opt => {
                    const o = document.createElement("option");
                    o.value = opt; o.innerText = opt;
                    if (loc.time_of_day === opt) o.selected = true;
                    todSelect.appendChild(o);
                });
                todSelect.addEventListener("change", (e) => {
                    locations[index].time_of_day = e.target.value;
                    syncData();
                });

                const descInput = document.createElement("input");
                descInput.placeholder = "Description";
                descInput.value = loc.description || "";
                descInput.style.flex = "1.4 1 0";
                descInput.style.minWidth = "0";
                descInput.style.boxSizing = "border-box";
                descInput.style.fontSize = "10px";
                descInput.style.background = "#222";
                descInput.style.color = "#fff";
                descInput.style.border = "1px solid #444";
                descInput.addEventListener("input", (e) => {
                    locations[index].description = e.target.value;
                    syncData();
                });

                const removeBtn = document.createElement("button");
                removeBtn.innerText = "✕";
                removeBtn.style.flex = "0 0 auto";
                removeBtn.style.background = "#6c757d";
                removeBtn.style.color = "#fff";
                removeBtn.style.border = "none";
                removeBtn.style.cursor = "pointer";
                removeBtn.addEventListener("click", () => {
                    locations.splice(index, 1);
                    renderLocations();
                    syncData();
                });

                row.appendChild(ieSelect);
                row.appendChild(nameInput);
                row.appendChild(todSelect);
                row.appendChild(descInput);
                row.appendChild(removeBtn);
                locContainer.appendChild(row);
            });
            locGroup.titleSpan.innerText = `${locGroup.body.style.display === "none" ? "▶" : "▼"} Locations (${locations.length})`;
        };

        addLocBtn.addEventListener("click", () => {
            locations.push({ int_ext: "INT", name: "", time_of_day: "DAY", description: "" });
            renderLocations();
            syncData();
        });

        locGroup.body.appendChild(locContainer);
        locGroup.body.appendChild(addLocBtn);
        locGroup.body.appendChild(locQuickBar);

        const insertTextAtCursor = (textToInsert) => {
            const scriptWidget = getWidget("raw_script");
            if (!scriptWidget) return;

            const textarea = scriptWidget.inputEl || document.querySelector(`textarea[name="${scriptWidget.name}"]`);
            if (textarea) {
                const start = textarea.selectionStart || textarea.value.length;
                const end = textarea.selectionEnd || textarea.value.length;
                const val = textarea.value;
                textarea.value = val.substring(0, start) + textToInsert + val.substring(end);
                textarea.selectionStart = textarea.selectionEnd = start + textToInsert.length;
                textarea.focus();
                scriptWidget.value = textarea.value;
                triggerLiveFormatting(textarea.value);
            } else {
                scriptWidget.value = (scriptWidget.value || "") + "\n" + textToInsert;
                triggerLiveFormatting(scriptWidget.value);
            }
        };

        const renderQuickBars = () => {
            charQuickBar.innerHTML = "";
            propQuickBar.innerHTML = "";
            locQuickBar.innerHTML = "";

            characters.forEach(c => {
                if (!c.name) return;
                const chip = document.createElement("button");
                const formattedChar = c.name.toUpperCase();
                chip.innerText = `👤 ${formattedChar}`;
                chip.style.display = "inline-flex";
                chip.style.alignItems = "center";
                chip.style.height = "26px";
                chip.style.fontSize = "11px";
                chip.style.background = "#38b000";
                chip.style.color = "#111";
                chip.style.fontWeight = "bold";
                chip.style.padding = "0 8px";
                chip.style.borderRadius = "4px";
                chip.style.cursor = "pointer";
                chip.style.border = "none";
                chip.style.whiteSpace = "nowrap";
                chip.addEventListener("click", () => insertTextAtCursor(`\n${formattedChar}\n`));
                charQuickBar.appendChild(chip);
            });

            props.forEach(prop => {
                if (!prop.name) return;
                const chip = document.createElement("button");
                const formattedProp = prop.name.toUpperCase();
                chip.innerText = `📦 ${formattedProp}`;
                chip.style.display = "inline-flex";
                chip.style.alignItems = "center";
                chip.style.height = "26px";
                chip.style.fontSize = "11px";
                chip.style.background = "#6b4f1d";
                chip.style.color = "#fff";
                chip.style.padding = "0 8px";
                chip.style.borderRadius = "4px";
                chip.style.cursor = "pointer";
                chip.style.border = "1px solid #9a7b3f";
                chip.style.whiteSpace = "nowrap";
                chip.addEventListener("click", () => insertTextAtCursor(`\n${formattedProp}\n`));
                propQuickBar.appendChild(chip);
            });

            locations.forEach(loc => {
                if (!loc.name) return;
                const chip = document.createElement("button");
                const formattedLoc = `${loc.int_ext}. ${loc.name.toUpperCase()} - ${loc.time_of_day}`;
                chip.innerText = `📍 ${formattedLoc}`;
                chip.style.display = "inline-flex";
                chip.style.alignItems = "center";
                chip.style.height = "26px";
                chip.style.fontSize = "11px";
                chip.style.background = "#1d3557";
                chip.style.color = "#f1faee";
                chip.style.padding = "0 8px";
                chip.style.borderRadius = "4px";
                chip.style.cursor = "pointer";
                chip.style.border = "1px solid #457b9d";
                chip.style.whiteSpace = "nowrap";
                chip.addEventListener("click", () => insertTextAtCursor(`\n${formattedLoc}\n`));
                locQuickBar.appendChild(chip);
            });
        };

        const managerWrapper = document.createElement("div");
        managerWrapper.style.display = "flex";
        managerWrapper.style.flexDirection = "column";
        managerWrapper.style.gap = "6px";
        managerWrapper.style.margin = "0";
        managerWrapper.style.boxSizing = "border-box";
        managerWrapper.style.width = "100%";
        managerWrapper.style.minWidth = "0";

        managerWrapper.appendChild(stylePresetInfo);
        managerWrapper.appendChild(settingsGroup.outer);
        managerWrapper.appendChild(charGroup.outer);
        managerWrapper.appendChild(propGroup.outer);
        managerWrapper.appendChild(locGroup.outer);

        const writerGroup = createCollapsible("Writer", true, "writer");
        writerGroup.outer.style.display = "flex";
        writerGroup.outer.style.flexDirection = "column";
        writerGroup.outer.style.flex = "1 1 auto";
        writerGroup.outer.style.minHeight = "0";
        writerGroup.body.style.padding = "0";
        writerGroup.body.style.flex = "1 1 auto";
        writerGroup.body.style.minHeight = "0";

        // --- COLLAPSIBLE LIVE PREVIEW GROUP ---
        const previewGroup = createCollapsible("Live Script Preview", false, "preview");
        previewGroup.outer.style.marginTop = "0";
        previewGroup.outer.style.flex = "0 0 auto";
        previewGroup.outer.style.minHeight = "26px";

        const previewWidget = document.createElement("div");
        previewWidget.style.fontFamily = "Courier New, Courier, monospace";
        previewWidget.style.fontSize = "11px";
        previewWidget.style.background = "#141414";
        previewWidget.style.color = "#e0e0e0";
        previewWidget.style.padding = "8px";
        previewWidget.style.minHeight = "80px";
        previewWidget.style.maxHeight = "400px";
        previewWidget.style.overflowY = "auto";
        previewWidget.style.border = "1px solid #333";
        previewWidget.style.borderRadius = "4px";
        previewWidget.style.boxSizing = "border-box";
        previewWidget.style.width = "100%";

        previewGroup.body.appendChild(previewWidget);

        // Caret position calculator
        const getCaretCoordinates = (textarea, position) => {
            const mirror = document.createElement("div");
            const style = window.getComputedStyle(textarea);
            const properties = [
                'direction', 'boxSizing', 'width', 'height', 'overflowX', 'overflowY',
                'borderTopWidth', 'borderRightWidth', 'borderBottomWidth', 'borderLeftWidth', 'borderStyle',
                'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
                'fontStyle', 'fontVariant', 'fontWeight', 'fontStretch', 'fontSize',
                'fontSizeAdjust', 'lineHeight', 'fontFamily', 'textAlign', 'textTransform',
                'textIndent', 'textDecoration', 'letterSpacing', 'wordSpacing',
                'tabSize', 'MozTabSize'
            ];
            properties.forEach(prop => mirror.style[prop] = style[prop]);
            mirror.style.position = "absolute";
            mirror.style.visibility = "hidden";
            mirror.style.whiteSpace = "pre-wrap";
            mirror.style.wordWrap = "break-word";

            mirror.textContent = textarea.value.substring(0, position);
            const span = document.createElement("span");
            span.textContent = textarea.value.substring(position) || ".";
            mirror.appendChild(span);

            document.body.appendChild(mirror);
            const coordinates = {
                top: span.offsetTop + parseInt(style.borderTopWidth || '0', 10),
                left: span.offsetLeft + parseInt(style.borderLeftWidth || '0', 10),
                lineHeight: parseInt(style.lineHeight || style.fontSize || '16', 10) || 18
            };
            document.body.removeChild(mirror);
            return coordinates;
        };

        // Auto-completion Popup
        const autoSuggestBox = document.createElement("div");
        autoSuggestBox.style.position = "absolute";
        autoSuggestBox.style.display = "none";
        autoSuggestBox.style.zIndex = "10000";
        autoSuggestBox.style.background = "#222222";
        autoSuggestBox.style.border = "1px solid #666";
        autoSuggestBox.style.borderRadius = "6px";
        autoSuggestBox.style.maxHeight = "220px";
        autoSuggestBox.style.minWidth = "220px";
        autoSuggestBox.style.overflowY = "auto";
        autoSuggestBox.style.boxShadow = "0 6px 16px rgba(0,0,0,0.85)";
        autoSuggestBox.style.fontSize = "13px";
        autoSuggestBox.style.fontFamily = "Courier New, Courier, monospace";

        let currentMatches = [];
        let selectedIndex = 0;

        const applySuggestion = (suggestionText) => {
            const scriptWidget = getWidget("raw_script");
            if (!scriptWidget) return;
            const textarea = scriptWidget.inputEl || document.querySelector(`textarea[name="${scriptWidget.name}"]`);
            if (!textarea) return;

            const start = textarea.selectionStart;
            const text = textarea.value;

            const lastNewline = text.lastIndexOf('\n', start - 1);
            const lineStart = lastNewline === -1 ? 0 : lastNewline + 1;

            let nextNewline = text.indexOf('\n', start);
            if (nextNewline === -1) nextNewline = text.length;

            const before = text.substring(0, lineStart);
            const after = text.substring(nextNewline);

            textarea.value = before + suggestionText + after;
            textarea.selectionStart = textarea.selectionEnd = lineStart + suggestionText.length;
            textarea.focus();

            scriptWidget.value = textarea.value;
            triggerLiveFormatting(textarea.value);
            autoSuggestBox.style.display = "none";
            currentMatches = [];
        };

        const renderMatches = () => {
            autoSuggestBox.innerHTML = "";
            currentMatches.forEach((m, idx) => {
                const item = document.createElement("div");
                item.style.padding = "8px 12px";
                item.style.cursor = "pointer";
                item.style.color = "#fff";
                item.style.fontWeight = "bold";
                item.style.borderBottom = "1px solid #333";
                item.style.background = idx === selectedIndex ? "#2b4c7e" : "transparent";
                item.innerText = m.label;

                item.addEventListener("mouseenter", () => {
                    selectedIndex = idx;
                    renderMatches();
                });
                item.addEventListener("mousedown", (e) => {
                    e.preventDefault();
                    applySuggestion(m.text);
                });

                autoSuggestBox.appendChild(item);
            });

            const selectedElem = autoSuggestBox.children[selectedIndex];
            if (selectedElem) {
                selectedElem.scrollIntoView({ block: "nearest" });
            }
        };

        const updateSuggestions = () => {
            const scriptWidget = getWidget("raw_script");
            if (!scriptWidget) return;
            const textarea = scriptWidget.inputEl || document.querySelector(`textarea[name="${scriptWidget.name}"]`);
            if (!textarea) return;

            const caretPos = textarea.selectionStart;
            const text = textarea.value;

            const lastNewline = text.lastIndexOf('\n', caretPos - 1);
            const lineStart = lastNewline === -1 ? 0 : lastNewline + 1;
            const currentLine = text.substring(lineStart, caretPos).trim().toUpperCase();

            if (!currentLine) {
                autoSuggestBox.style.display = "none";
                currentMatches = [];
                return;
            }

            const matches = [];

            if ("INT.".startsWith(currentLine) && currentLine !== "INT.") matches.push({ label: "📍 INT.", text: "INT. " });
            if ("EXT.".startsWith(currentLine) && currentLine !== "EXT.") matches.push({ label: "📍 EXT.", text: "EXT. " });

            props.forEach(prop => {
                if (!prop.name) return;
                const formatted = prop.name.toUpperCase();
                if (formatted.startsWith(currentLine) && formatted !== currentLine) {
                    matches.push({ label: `📦 ${formatted}`, text: formatted });
                }
            });

            locations.forEach(loc => {
                if (!loc.name) return;
                const formatted = `${loc.int_ext}. ${loc.name.toUpperCase()} - ${loc.time_of_day}`;
                if (formatted.toUpperCase().includes(currentLine) && formatted.toUpperCase() !== currentLine) {
                    matches.push({ label: `📍 ${formatted}`, text: formatted });
                }
            });

            characters.forEach(c => {
                if (!c.name) return;
                const nameUpper = c.name.toUpperCase();
                if (nameUpper.startsWith(currentLine) && nameUpper !== currentLine) {
                    matches.push({ label: `👤 ${nameUpper}`, text: nameUpper });
                }
            });

            if (matches.length === 0) {
                autoSuggestBox.style.display = "none";
                currentMatches = [];
                return;
            }

            currentMatches = matches;
            selectedIndex = 0;
            renderMatches();

            const coords = getCaretCoordinates(textarea, caretPos);
            const lineH = coords.lineHeight || 18;
            
            autoSuggestBox.style.left = `${textarea.offsetLeft + coords.left - textarea.scrollLeft}px`;
            autoSuggestBox.style.top = `${textarea.offsetTop + coords.top + lineH - textarea.scrollTop + 2}px`;
            autoSuggestBox.style.display = "block";
        };

        const triggerLiveFormatting = (rawText) => {
            if (!rawText || !rawText.trim()) {
                previewWidget.innerHTML = '<span style="color: #666;">Type into raw_script above to render formatted screenplay live...</span>';
                return;
            }

            const lines = rawText.split('\n');
            const pages = [];
            let pageLines = [];
            let pageLineCount = 0;
            let inCharacterContext = false;
            const sceneHeadingRe = /^(INT\.|EXT\.|INT\/EXT\.|EXT\/INT\.|INT\.\/EXT\.|EST\.)/i;
            const transitionRe = /^(FADE IN:|FADE OUT\.|CUT TO:|DISSOLVE TO:|SMASH CUT TO:|FADE TO BLACK\.)/i;
            const parentheticalRe = /^\(.*\)$/;
            const flushPage = () => {
                if (pageLines.length) pages.push(pageLines);
                pageLines = [];
                pageLineCount = 0;
            };
            const appendPreviewLine = (markup, visualLines = 1) => {
                const count = Math.max(1, visualLines);
                if (pageLineCount && pageLineCount + count > 55) flushPage();
                pageLines.push(markup);
                pageLineCount += count;
                if (pageLineCount >= 55) flushPage();
            };
            const wrappedLineCount = (line, width, indent = 0) =>
                Math.max(1, Math.ceil((line.length + indent) / width));

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line) {
                    inCharacterContext = false;
                    appendPreviewLine('<div style="height:11.75pt;"></div>');
                    continue;
                }

                const escapedLine = line.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                if (sceneHeadingRe.test(line)) {
                    inCharacterContext = false;
                    appendPreviewLine(`<div style="font-weight:bold;text-transform:uppercase;margin-bottom:11.75pt;">${escapedLine.toUpperCase()}</div>`, wrappedLineCount(line, 60) + 1);
                } else if (transitionRe.test(line) || (line === line.toUpperCase() && line.endsWith(':'))) {
                    inCharacterContext = false;
                    appendPreviewLine(`<div style="text-align:right;margin:11.75pt 0;">${escapedLine.toUpperCase()}</div>`, wrappedLineCount(line, 60, 52) + 1);
                } else if (line === line.toUpperCase() && line.length < 40 && !line.endsWith('.')) {
                    inCharacterContext = true;
                    appendPreviewLine(`<div style="margin-left:38%;margin-top:11.75pt;">${escapedLine}</div>`, wrappedLineCount(line, 60, 28) + 1);
                } else if (parentheticalRe.test(line)) {
                    appendPreviewLine(`<div style="margin-left:28%;max-width:45%;">${escapedLine}</div>`, wrappedLineCount(line, 30, 20));
                } else if (inCharacterContext) {
                    appendPreviewLine(`<div style="margin-left:20%;max-width:55%;margin-bottom:11.75pt;">${escapedLine}</div>`, wrappedLineCount(line, 36, 14) + 1);
                } else {
                    inCharacterContext = false;
                    appendPreviewLine(`<div>${escapedLine}</div>`, wrappedLineCount(line, 60));
                }
            }
            flushPage();
            const pageMarkup = pages.map((page, index) =>
                `<section class="screenplay-page"><div class="screenplay-page-number">${index + 1}.</div><div class="screenplay-page-content">${page.join("")}</div></section>`
            ).join("");
            previewWidget.innerHTML = `<style>
                .screenplay-pages{display:flex;flex-direction:column;align-items:center;gap:16px;background:#303030;padding:12px 0;}
                .screenplay-page{position:relative;box-sizing:border-box;width:8.5in;height:11in;min-height:11in;padding:1in 1in 1in 1.5in;background:#fff;color:#111;overflow:hidden;box-shadow:0 2px 12px #0008;font:12pt/11.75pt 'Courier Prime','Courier New',Courier,monospace;}
                .screenplay-page-number{position:absolute;top:.5in;right:1in;font:12pt/12pt 'Courier Prime','Courier New',Courier,monospace;}
                .screenplay-page-content{font:inherit;line-height:11.75pt;white-space:pre-wrap;overflow-wrap:anywhere;}
                @media print{.screenplay-pages{display:block;padding:0;background:#fff}.screenplay-page{margin:0;box-shadow:none;page-break-after:always;break-after:page}.screenplay-page:last-child{page-break-after:auto;break-after:auto}}
                @page{size:letter;margin:0}
            </style><div class="screenplay-pages">${pageMarkup}</div>`;
        };

        const restoreFromWidgets = () => {
            isRestoring = true;
            console.log("[ScreenplayFormatter] Restoring widget state from workflow storage...");
            const charW = getWidget("characters_json");
            const locW = getWidget("locations_json");
            const propW = getWidget("props_json");
            const scriptW = getWidget("raw_script");

            if (charW && charW.value) {
                try {
                    characters = JSON.parse(charW.value);
                    renderCharacters();
                } catch (e) {
                    characters = [];
                }
            }
            if (locW && locW.value) {
                try {
                    locations = JSON.parse(locW.value);
                    renderLocations();
                } catch (e) {
                    locations = [];
                }
            }

            if (propW && propW.value) {
                try {
                    props = JSON.parse(propW.value);
                    renderProps();
                } catch (e) {
                    props = [];
                }
            }

            if (scriptW && scriptW.value) {
                const textarea = scriptW.inputEl || document.querySelector(`textarea[name="${scriptW.name}"]`);
                if (textarea) textarea.value = scriptW.value;
                triggerLiveFormatting(scriptW.value);
            }

            renderSettings();
            renderQuickBars();
            updateStylePresetInfo();
            isRestoring = false;
        };

        // ------------------------------------------------------------------
        // LAYOUT: one flex column ("workspace") that holds
        //   [ Settings / Characters / Props / Locations ] [ Writer ] [ Live Preview ]
        // The raw_script widget reports the TOTAL height of that column to the
        // node's layout, and a ResizeObserver grows/shrinks the node whenever a
        // group is expanded/collapsed or rows/chips are added.
        // ------------------------------------------------------------------
        const TEXTAREA_MIN_HEIGHT = 140;
        const WORKSPACE_GAP = 6;
        const MIN_NODE_WIDTH = 400;

        const workspace = document.createElement("div");
        workspace.style.position = "relative";
        workspace.style.display = "flex";
        workspace.style.flexDirection = "column";
        workspace.style.gap = `${WORKSPACE_GAP}px`;
        workspace.style.width = "100%";
        workspace.style.height = "100%";
        workspace.style.flex = "1 1 auto";
        workspace.style.minHeight = "0";
        workspace.style.minWidth = "0";
        workspace.style.boxSizing = "border-box";

        let lastRequiredHeight = 0;
        let lastAutoHeight = null;
        let writerWasExpanded = writerGroup.body.style.display !== "none";
        const savedWriterHeights = node.properties?.screenplay_ui_state?.writerHeights || {};
        const readSavedHeight = (value) => value != null && Number.isFinite(Number(value)) && Number(value) > 0
            ? Number(value)
            : null;
        let writerStateHeights = {
            expanded: readSavedHeight(savedWriterHeights.expanded),
            collapsed: readSavedHeight(savedWriterHeights.collapsed)
        };
        const collapsibleGroups = {
            settings: settingsGroup,
            characters: charGroup,
            props: propGroup,
            locations: locGroup,
            writer: writerGroup,
            preview: previewGroup
        };
        let restoringSavedNodeHeight = true;
        const persistWriterStateHeights = () => {
            if (restoringSavedNodeHeight) return;
            const savedState = node.properties.screenplay_ui_state || {};
            node.properties.screenplay_ui_state = {
                ...savedState,
                writerHeights: { ...writerStateHeights }
            };
        };
        const rememberWriterStateHeight = (expanded, height = node.size[1]) => {
            if (restoringSavedNodeHeight) return;
            writerStateHeights[expanded ? "expanded" : "collapsed"] = height;
            persistWriterStateHeights();
        };
        const restoreUiStateFromProperties = () => {
            restoringSavedNodeHeight = true;
            const savedState = node.properties?.screenplay_ui_state;
            Object.entries(collapsibleGroups).forEach(([key, group]) => {
                const expanded = savedState?.groups?.[key];
                if (typeof expanded === "boolean") group.setExpanded(expanded);
            });
            writerWasExpanded = writerGroup.isExpanded();
            writerGroup.outer.style.flex = writerWasExpanded ? "1 1 auto" : "0 0 auto";
            const heights = savedState?.writerHeights || {};
            writerStateHeights.expanded = readSavedHeight(heights.expanded);
            writerStateHeights.collapsed = readSavedHeight(heights.collapsed);
        };
        let applyingWriterStateHeight = false;
        let writerStateTransitioning = false;
        let lastConnectionCollapseState = !!node.properties?.collapse_connections;
        let resizeObserver = null;
        let listenersBound = false;

        // Minimum content height uses the Writer header and textarea minimum, not
        // the Writer's current flex-grown height (which follows manual node resize).
        const getAutoHeight = () => {
            const writerHeaderHeight = writerGroup.header.offsetHeight || 24;
            const writerMinimum = writerGroup.body.style.display === "none" ? 0 : TEXTAREA_MIN_HEIGHT;
            const writerTopMargin = parseFloat(window.getComputedStyle(writerGroup.outer).marginTop) || 0;
            // The preview header still needs space when its body is collapsed.
            // During workflow restoration outer.offsetHeight can briefly be zero
            // before the nested workspace has completed its first layout pass.
            const previewHeight = previewGroup.outer.offsetHeight || previewGroup.header.offsetHeight || 24;
            return managerWrapper.offsetHeight + writerTopMargin + writerHeaderHeight + writerMinimum + previewHeight;
        };

        // Total raw_script widget height required to fit all fixed sections and
        // the Writer's minimum textarea height without double-counting its body.
        const getRequiredHeight = () => {
            const scriptWidget = getWidget("raw_script");
            const margin = (scriptWidget?.options?.margin ?? 10) * 2;
            const auto = getAutoHeight();
            if (auto > 0) {
                lastRequiredHeight = auto + WORKSPACE_GAP * 2 + margin;
            }
            return lastRequiredHeight || (TEXTAREA_MIN_HEIGHT + 120);
        };

        node.__screenplayContentMinHeight = () => {
            const scriptWidget = getWidget("raw_script");
            const measuredWidgetY = Math.max(
                Number(scriptWidget?.last_y) || 0,
                Number(scriptWidget?.y) || 0
            );
            const visibleSlotRows = Math.max(
                (node.inputs || []).filter(slot => !slot.hidden).length,
                (node.outputs || []).filter(slot => !slot.hidden).length
            );
            const titleHeight = Number(LiteGraph?.NODE_TITLE_HEIGHT) || 30;
            const slotHeight = Number(LiteGraph?.NODE_SLOT_HEIGHT) || 20;
            const estimatedWidgetY = titleHeight + visibleSlotRows * slotHeight + 10;
            const widgetY = Math.max(measuredWidgetY, estimatedWidgetY);
            // Use an estimated widget offset until LiteGraph finishes its first
            // layout pass; otherwise a collapsed workflow can temporarily get a
            // zero content floor and clip the final group.
            return Math.ceil(widgetY + getRequiredHeight() + 10);
        };

        const installLayoutHooks = (scriptWidget) => {
            if (scriptWidget.__screenplayLayoutHooked) return;
            scriptWidget.__screenplayLayoutHooked = true;
            scriptWidget.options = scriptWidget.options || {};
            scriptWidget.options.getMinHeight = getRequiredHeight;
            // legacy LiteGraph layout
            scriptWidget.computeSize = (width) => [width ?? node.size[0], getRequiredHeight()];
            // newer frontend layout
            scriptWidget.computeLayoutSize = () => ({ minHeight: getRequiredHeight(), minWidth: 0 });
        };

        // Resize the node so it always wraps its content. Keeps any extra
        // height the user dragged in for the textarea, and only applies the
        // *change* in the groups' height.
        const fitNodeToContent = () => {
            // During workflow load, resizing from partially-mounted DOM measurements
            // can overwrite the canvas size before its saved height is restored.
            if (restoringSavedNodeHeight) return;
            const auto = getAutoHeight();
            if (auto === 0) return; // detached or hidden (e.g. zoomed out)
            const delta = lastAutoHeight === null ? 0 : auto - lastAutoHeight;
            lastAutoHeight = auto;

            const min = node.computeSize();
            node.setSize([
                Math.max(node.size[0], min[0], MIN_NODE_WIDTH),
                Math.max(min[1], node.size[1] + delta)
            ]);
            if (!applyingWriterStateHeight && !writerStateTransitioning) {
                rememberWriterStateHeight(writerWasExpanded, node.size[1]);
            }
            app.graph?.setDirtyCanvas(true, true);
            syncWorkspaceHeight();
        };

        node.__screenplaySyncConnectionLayout = () => {
            // Let LiteGraph apply the hidden-slot state before measuring; doing
            // this synchronously can use the old connection-row height.
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    writerStateTransitioning = true;
                    try {
                        const minSize = node.computeSize();
                        node.setSize([
                            Math.max(node.size[0], minSize[0], MIN_NODE_WIDTH),
                            minSize[1]
                        ]);
                        lastAutoHeight = getAutoHeight();
                        rememberWriterStateHeight(writerWasExpanded, node.size[1]);
                        lastConnectionCollapseState = !!node.properties?.collapse_connections;
                    } finally {
                        writerStateTransitioning = false;
                    }
                    syncWorkspaceHeight();
                    app.graph?.setDirtyCanvas(true, true);
                });
            });
        };

        let nodeHeightRestoreToken = 0;
        let configuredWorkflowNodeHeight = null;
        const restoreSavedNodeHeight = (fallbackHeight = configuredWorkflowNodeHeight) => {
            const stateKey = writerWasExpanded ? "expanded" : "collapsed";
            // The workflow's node size is the authoritative height for the state
            // that was active when saved. Per-state values are for in-session
            // toggles and may be stale after an earlier layout pass.
            const savedHeight = readSavedHeight(fallbackHeight) ?? writerStateHeights[stateKey];
            if (savedHeight === null) {
                restoringSavedNodeHeight = false;
                requestAnimationFrame(() => fitNodeToContent());
                return;
            }

            const restoreToken = ++nodeHeightRestoreToken;
            let attempts = 0;
            const applyRestoredHeight = () => {
                if (restoreToken !== nodeHeightRestoreToken) return;

                const scriptWidget = getWidget("raw_script");
                const widgetY = scriptWidget?.last_y ?? scriptWidget?.y ?? 0;
                const contentMinimum = node.__screenplayContentMinHeight();
                if (!(widgetY > 0 && contentMinimum > 0) && attempts < 12) {
                    attempts++;
                    requestAnimationFrame(applyRestoredHeight);
                    return;
                }

                writerStateTransitioning = true;
                applyingWriterStateHeight = true;
                try {
                    const min = node.computeSize();
                    node.setSize([
                        Math.max(node.size[0], min[0], MIN_NODE_WIDTH),
                        Math.max(savedHeight, min[1], contentMinimum)
                    ]);
                } finally {
                    applyingWriterStateHeight = false;
                    writerStateTransitioning = false;
                    restoringSavedNodeHeight = false;
                }
                rememberWriterStateHeight(writerWasExpanded, node.size[1]);
                lastAutoHeight = getAutoHeight();
                syncWorkspaceHeight();
                app.graph?.setDirtyCanvas(true, true);
            };

            requestAnimationFrame(() => requestAnimationFrame(applyRestoredHeight));
        };

        node.__screenplayRestoreLayout = () => {
            // afterConfigureGraph runs after ComfyUI applies the serialized size;
            // capture that authoritative value before any DOM measurement resizes it.
            configuredWorkflowNodeHeight = configuredWorkflowNodeHeight
                ?? readSavedHeight(node.size?.[1]);
            restoreSavedNodeHeight(configuredWorkflowNodeHeight);
            requestAnimationFrame(() => requestAnimationFrame(() => {
                syncWorkspaceHeight();
                app.graph?.setDirtyCanvas(true, true);
            }));
        };

        const handleWriterToggle = () => {
            const wasExpanded = writerWasExpanded;
            const isExpanded = writerGroup.body.style.display !== "none";
            if (wasExpanded === isExpanded) return;

            rememberWriterStateHeight(wasExpanded, node.size[1]);
            writerWasExpanded = isExpanded;
            writerStateTransitioning = true;
            writerGroup.outer.style.flex = isExpanded ? "1 1 auto" : "0 0 auto";

            requestAnimationFrame(() => {
                const min = node.computeSize();
                const stateKey = isExpanded ? "expanded" : "collapsed";
                const rememberedHeight = writerStateHeights[stateKey];
                const targetHeight = rememberedHeight === null
                    ? (isExpanded ? Math.max(node.size[1], min[1]) : min[1])
                    : Math.max(min[1], rememberedHeight);
                applyingWriterStateHeight = true;
                try {
                    node.setSize([Math.max(node.size[0], min[0], MIN_NODE_WIDTH), targetHeight]);
                } finally {
                    applyingWriterStateHeight = false;
                }
                rememberWriterStateHeight(isExpanded, node.size[1]);
                lastAutoHeight = getAutoHeight();
                writerStateTransitioning = false;
                app.graph?.setDirtyCanvas(true, true);
                syncWorkspaceHeight();
            });
        };

        // Make the workspace exactly as tall as the node's remaining space, so
        // the textarea absorbs all spare height and the preview group sits
        // flush with the bottom edge at any node size.
        let lastWorkspaceHeight = -1;
        const syncWorkspaceHeight = () => {
            const w = getWidget("raw_script");
            if (!w || workspace.parentElement === null) return;
            const margin = (w.options?.margin ?? 10) * 2;
            const y = w.last_y ?? w.y ?? 0;
            if (!(y > 0)) return; // layout not computed yet; CSS 100% applies meanwhile
            const available = node.size[1] - y - margin - 6;
            const required = getRequiredHeight() - margin;
            const h = Math.round(Math.max(required, available));
            if (h !== lastWorkspaceHeight) {
                lastWorkspaceHeight = h;
                workspace.style.height = `${h}px`;
            }
        };

        const origOnResize = node.onResize;
        node.onResize = function () {
            if (origOnResize) origOnResize.apply(this, arguments);
            if (!applyingWriterStateHeight && !writerStateTransitioning) {
                rememberWriterStateHeight(writerWasExpanded, node.size[1]);
            }
            syncWorkspaceHeight();
        };

        const origOnDrawBackground = node.onDrawBackground;
        node.onDrawBackground = function () {
            if (origOnDrawBackground) origOnDrawBackground.apply(this, arguments);
            const connectionCollapseState = !!this.properties?.collapse_connections;
            if (connectionCollapseState !== lastConnectionCollapseState) {
                lastConnectionCollapseState = connectionCollapseState;
                applyCollapseState(this);
                this.__screenplaySyncConnectionLayout?.();
            }
            syncWorkspaceHeight(); // cheap: only touches the DOM when the value changes
        };

        const mountDOMContainers = () => {
            const scriptWidget = getWidget("raw_script");
            if (!scriptWidget || !scriptWidget.inputEl) return false;

            const textarea = scriptWidget.inputEl;
            const parent = textarea.parentElement || textarea.parentNode;
            if (!parent) return false;

            installLayoutHooks(scriptWidget);

            // Build the flex column: managers -> collapsible writer -> preview (+ popup)
            if (workspace.parentElement !== parent) parent.appendChild(workspace);
            if (managerWrapper.parentElement !== workspace) workspace.appendChild(managerWrapper);
            if (writerGroup.outer.parentElement !== workspace) workspace.appendChild(writerGroup.outer);
            if (textarea.parentElement !== writerGroup.body) writerGroup.body.appendChild(textarea);
            if (previewGroup.outer.parentElement !== workspace) workspace.appendChild(previewGroup.outer);
            if (autoSuggestBox.parentElement !== workspace) workspace.appendChild(autoSuggestBox);

            // The textarea takes whatever height is left over
            textarea.style.position = "relative";
            textarea.style.left = "auto";
            textarea.style.top = "auto";
            textarea.style.flex = "1 1 0px";
            textarea.style.width = "100%";
            textarea.style.height = "0px";
            textarea.style.minHeight = `${TEXTAREA_MIN_HEIGHT}px`;
            textarea.style.boxSizing = "border-box";
            textarea.style.resize = "none";

            if (!listenersBound) {
                listenersBound = true;

                textarea.addEventListener("keydown", (e) => {
                    if (autoSuggestBox.style.display !== "none" && currentMatches.length > 0) {
                        if (e.key === "ArrowDown") {
                            e.preventDefault();
                            selectedIndex = (selectedIndex + 1) % currentMatches.length;
                            renderMatches();
                        } else if (e.key === "ArrowUp") {
                            e.preventDefault();
                            selectedIndex = (selectedIndex - 1 + currentMatches.length) % currentMatches.length;
                            renderMatches();
                        } else if (e.key === "Enter" || e.key === "Tab") {
                            e.preventDefault();
                            if (currentMatches[selectedIndex]) {
                                applySuggestion(currentMatches[selectedIndex].text);
                            }
                        } else if (e.key === "Escape") {
                            autoSuggestBox.style.display = "none";
                            currentMatches = [];
                        }
                    }
                });

                textarea.addEventListener("input", (e) => {
                    scriptWidget.value = e.target.value;
                    triggerLiveFormatting(e.target.value);
                    updateSuggestions();
                });

                textarea.addEventListener("click", updateSuggestions);
                textarea.addEventListener("blur", () => {
                    setTimeout(() => autoSuggestBox.style.display = "none", 200);
                });
            }

            if (!resizeObserver && typeof ResizeObserver !== "undefined") {
                resizeObserver = new ResizeObserver(() => fitNodeToContent());
                resizeObserver.observe(managerWrapper);
                resizeObserver.observe(previewGroup.outer);
                writerGroup.outer.addEventListener("screenplay-group-toggle", handleWriterToggle);
            }

            fitNodeToContent();
            syncWorkspaceHeight();
            return true;
        };

        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            if (resizeObserver) {
                resizeObserver.disconnect();
                resizeObserver = null;
            }
            delete node.__screenplaySyncConnectionLayout;
            delete node.__screenplayContentMinHeight;
            delete node.__screenplayRestoreLayout;
            if (origOnRemoved) origOnRemoved.apply(this, arguments);
        };

        const origOnSerialize = node.onSerialize;
        node.onSerialize = function (serializedNode) {
            if (origOnSerialize) origOnSerialize.apply(this, arguments);
            if (!serializedNode || typeof serializedNode !== "object") return;

            const currentSize = Array.isArray(this.size) ? [...this.size] : null;
            if (currentSize) serializedNode.size = currentSize;
            const savedState = serializedNode.properties?.screenplay_ui_state
                || this.properties?.screenplay_ui_state
                || {};
            const groups = { ...(savedState.groups || {}) };
            Object.entries(collapsibleGroups).forEach(([key, group]) => {
                groups[key] = group.isExpanded();
            });
            const writerHeights = { ...(savedState.writerHeights || {}) };
            const actualHeight = Number(currentSize?.[1]) || node.size[1];
            writerHeights[writerWasExpanded ? "expanded" : "collapsed"] = actualHeight;

            serializedNode.properties = serializedNode.properties || {};
            serializedNode.properties.screenplay_ui_state = {
                ...savedState,
                groups,
                writerHeights
            };
        };

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (nodeData) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            configuredWorkflowNodeHeight = readSavedHeight(nodeData?.size?.[1])
                ?? readSavedHeight(this.size?.[1]);
            restoreUiStateFromProperties();
            applyCollapseState(this);
            hideWidget("characters_json");
            hideWidget("locations_json");
            hideWidget("props_json");
            hideWidget("stored_generation");
            settingNames.forEach(name => hideWidget(name));
            mountDOMContainers();
            restoreFromWidgets();
            restoreSavedNodeHeight(configuredWorkflowNodeHeight);
        };

        setTimeout(async () => {
            restoreUiStateFromProperties();
            applyCollapseState(node);
            hideWidget("characters_json");
            hideWidget("locations_json");
            hideWidget("props_json");
            hideWidget("stored_generation");
            settingNames.forEach(name => hideWidget(name));
            mountDOMContainers();
            await populateOllamaModels();
            restoreFromWidgets();
            restoreSavedNodeHeight(configuredWorkflowNodeHeight);
        }, 100);

        node.onExecuted = function (message) {
            if (message && message.stored_generation) {
                const cacheWidget = getWidget("stored_generation");
                if (cacheWidget) cacheWidget.value = message.stored_generation[0] || "";
            }
            if (message && message.script) {
                const outputFormat = getWidget("output_format")?.value;
                if (outputFormat === "HTML Visual Script" || outputFormat === "Print-Ready (US Letter)") {
                    previewWidget.innerHTML = message.script[0];
                }
            }
        };
    },

    async afterConfigureGraph() {
        const nodes = app.graph?._nodes || [];
        nodes.forEach(node => {
            if (node.comfyClass === "triXope ScreenplayFormatterNode") {
                node.__screenplayRestoreLayout?.();
            }
        });
    }
});

// Register Buffer Extension to automatically sync backend state to the widget
app.registerExtension({
    name: "Comfy.ShotLengthBuffer",
    async nodeCreated(node) {
        if (node.comfyClass === "ShotLengthBufferNode") {
            node.onExecuted = function(message) {
                if (message && message.stored_array) {
                    const widget = node.widgets?.find(w => w.name === "stored_array");
                    if (widget) {
                        widget.value = message.stored_array[0];
                    }
                }
            };
        }
    }
});