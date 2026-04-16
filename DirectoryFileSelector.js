import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "Gemini.DirectoryFileSelector",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DirectoryFileSelectorGemini") {
            
            // This is the fix:
            // We intercept the node definition before it is registered and tell the
            // frontend to render a combo box (dropdown) for the 'file' input,
            // even though the backend type is a wildcard '*'.
            if (nodeData.input?.required?.file?.[0] === '*') {
                nodeData.input.required.file[0] = [""]; 
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const dirWidget = this.widgets.find((w) => w.name === "directory");
                const fileWidget = this.widgets.find((w) => w.name === "file");
                
                let debounceTimeout;

                const updateFileList = async (directoryPath) => {
                    try {
                        const resp = await api.fetchApi(`/gemini/get_files?directory=${encodeURIComponent(directoryPath)}`);
                        
                        if (resp.status !== 200) {
                            throw new Error(`Failed to fetch: ${resp.status}`);
                        }

                        const data = await resp.json();
                        const files = data.files || [];
                        
                        const currentFile = fileWidget.value;
                        
                        fileWidget.options.values = files.length > 0 ? files : [""];
                        
                        if (files.includes(currentFile)) {
                            fileWidget.value = currentFile;
                        } else {
                            fileWidget.value = fileWidget.options.values[0];
                        }

                    } catch (error) {
                        console.error("DirectoryFileSelector: Failed to fetch files:", error);
                        fileWidget.options.values = ["Error loading files"];
                        fileWidget.value = "Error loading files";
                    }
                };

                const originalCallback = dirWidget.callback;

                dirWidget.callback = (value) => {
                    originalCallback?.(value);
                    
                    clearTimeout(debounceTimeout);
                    debounceTimeout = setTimeout(() => {
                        updateFileList(value);
                    }, 500); 
                };
            };
        }
    },
});
