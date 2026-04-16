import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.GroupMonitor",
    
    /**
     * Called once when the app is setting up.
     * This is where we patch the websocket message handler.
     */
    setup() {
        // Ensure app.socket is available
        if (app.socket) {
            const oldOnMessage = app.socket.onmessage;
            
            // Override the onmessage function
            app.socket.onmessage = function(event) {
                // Call the original message handler
                if (oldOnMessage) {
                    oldOnMessage.apply(this, arguments);
                }

                try {
                    const msg = JSON.parse(event.data);
                    let groupName = null;
                    let isExecuting = false;
                    let isReset = false;

                    if (msg.type === 'executing' && msg.data.node) {
                        // A node is starting to execute
                        const nodeId = msg.data.node;
                        const runningNode = app.graph.getNodeById(nodeId);
                        
                        if (runningNode && runningNode.group) {
                            // Node is in a group
                            groupName = runningNode.group.title || "Untitled Group";
                        } else if (runningNode) {
                            // Node is running but not in a group
                            groupName = "No Group";
                        }
                        isExecuting = true;

                    } else if (msg.type === 'execution_start' || msg.type === 'execution_cached') {
                        // A new prompt is starting or finishing, reset the display
                        isReset = true;
                    }

                    // Now, find all GroupMonitor nodes and update them
                    for (const node of app.graph._nodes) {
                        // Identify our node by its Python class name
                        if (node.comfyClass === "GroupMonitor") { 
                            if (isExecuting && node.updateDisplay) {
                                node.updateDisplay(groupName);
                            } else if (isReset && node.resetDisplay) {
                                // Reset when a new prompt starts
                                node.resetDisplay();
                            }
                        }
                    }

                } catch (e) {
                    // console.error("Error in GroupMonitor patch:", e);
                }
            }
        } else {
            console.error("GroupMonitor: app.socket not ready. Cannot patch websocket.");
        }
    },

    /**
     * Called every time a node is created in the graph.
     * @param {LGraphNode} node The node that was just created
     * @param {object} app The main app object
     */
    nodeCreated(node, app) {
        // Check if the created node is our GroupMonitor
        if (node.comfyClass === "GroupMonitor") {
            
            // Add the text widget
            const widget = node.addWidget("text", "status", "Waiting...", (value) => {}, {
                "serialize": false // Don't save this value in the workflow
            });
            
            // Make the widget read-only in the node's properties panel
            if(widget.inputEl) {
                widget.inputEl.readOnly = true;
                widget.inputEl.style.opacity = 0.6; // Visually indicate it's read-only
            }
            
            // Add a custom method to the node instance to update the widget
            node.updateDisplay = function(groupName) {
                widget.value = "Currently running: " + groupName;
            }
            
            // Add a custom method to reset the display
            node.resetDisplay = function() {
                widget.value = "Waiting...";
            }
            
            // Set an initial size
            node.size = node.computeSize();
            node.size[0] = Math.max(300, node.size[0]); // Make it a bit wider
        }
    }
});