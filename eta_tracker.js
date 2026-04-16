import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI.RealtimeLoopTracker",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RealtimeLoopTracker") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                
                this.displayTime = "00:00:00";
                this.elapsedStr = "00:00:00";
                this.statusText = "READY";
                
                this.isCalculating = true;
                this.etaMs = 0;
                this.loopDurations = [];
                this.lastUpdateTime = 0;
                
                this.serialize_widgets = true; 
                this.size = [300, 180]; 
            };

            nodeType.prototype.updateDisplayTime = function() {
                if (this.etaMs <= 0 && !this.isCalculating) {
                    this.displayTime = "00:00:00";
                } else {
                    const seconds = Math.floor(this.etaMs / 1000);
                    const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
                    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
                    const s = (seconds % 60).toString().padStart(2, '0');
                    this.displayTime = `${h}:${m}:${s}`;
                }
            };

            nodeType.prototype.onDrawForeground = function (ctx) {
                if (this.flags.collapsed) return;

                const margin = 10;
                const y = 30; 
                const x = margin;
                const w = this.size[0] - margin * 2;
                const h = this.size[1] - y - margin; 

                ctx.fillStyle = "#000000";
                ctx.beginPath();
                ctx.roundRect(x, y, w, h, 5);
                ctx.fill();
                ctx.strokeStyle = "#444";
                ctx.stroke();

                const etaFontSize = Math.max(16, Math.min(w / 6.5, h * 0.35));
                const smallFontSize = Math.max(10, etaFontSize * 0.35);
                const centerY = y + (h / 2);
                ctx.textBaseline = "middle"; 

                ctx.fillStyle = "#FFFFFF";
                ctx.font = `${smallFontSize}px Arial`;
                ctx.textAlign = "center";
                ctx.fillText(`elapsed time: ${this.elapsedStr}`, x + w/2, centerY - etaFontSize * 0.9);

                ctx.fillStyle = "#00FF00"; 
                if (this.isCalculating) {
                    ctx.font = `bold ${etaFontSize * 0.6}px monospace`;
                    ctx.fillText("WAITING FOR ETA...", x + w/2, centerY);
                } else {
                    ctx.font = `bold ${etaFontSize}px monospace`;
                    ctx.fillText(`${this.displayTime} ETA`, x + w/2, centerY);
                }

                ctx.fillStyle = "#888";
                ctx.font = `${smallFontSize}px Arial`;
                ctx.fillText(this.statusText, x + w/2, centerY + etaFontSize * 0.9);
            };

            let globalStartTime = 0;
            let isExecuting = false;
            let elapsedInterval = null;

            api.addEventListener("executing", (event) => {
                if (!event.detail) isExecuting = false; 
            });

            api.addEventListener("execution_start", () => {
                globalStartTime = performance.now();
                isExecuting = true;

                app.graph._nodes.filter(n => n.type === "RealtimeLoopTracker").forEach(n => {
                    n.isCalculating = true;
                    n.etaMs = 0;
                    n.loopDurations = [];
                    n.lastUpdateTime = 0;
                    n.statusText = "WAITING FOR LOOP TO BEGIN...";
                    n.setDirtyCanvas(true, true);
                });

                if (!elapsedInterval) {
                    elapsedInterval = setInterval(() => {
                        if (isExecuting) {
                            const now = performance.now();
                            const elapsedSecs = Math.floor((now - globalStartTime) / 1000);
                            const h = Math.floor(elapsedSecs / 3600).toString().padStart(2, '0');
                            const m = Math.floor((elapsedSecs % 3600) / 60).toString().padStart(2, '0');
                            const s = (elapsedSecs % 60).toString().padStart(2, '0');
                            const timeStr = `${h}:${m}:${s}`;

                            app.graph._nodes.filter(n => n.type === "RealtimeLoopTracker").forEach(n => {
                                n.elapsedStr = timeStr;
                                
                                if (!n.isCalculating && n.etaMs > 0) {
                                    n.etaMs -= 1000;
                                    if (n.etaMs < 0) n.etaMs = 0;
                                    n.updateDisplayTime();
                                }
                                
                                n.setDirtyCanvas(true, true); 
                            });
                        }
                    }, 1000);
                }
            });

            api.addEventListener("loop_tracker_update", (event) => {
                const data = event.detail;
                const nodeId = Array.isArray(data.node_id) ? data.node_id[0] : data.node_id;
                
                const now = performance.now();
                const index = parseInt(data.index, 10);
                const total = parseInt(data.total, 10);

                // Offset the index by 1 so the UI reads 1, 2, 3 instead of 0, 1, 2
                const displayIndex = index + 1;

                const trackers = app.graph._nodes.filter(n => n.type === "RealtimeLoopTracker");
                if (trackers.length === 0) return;

                trackers.forEach(node => {
                    
                    // Cap the text at the total when finished (e.g., Loop 10 / 10)
                    if (index >= total) {
                        node.isCalculating = false;
                        node.etaMs = 0;
                        node.statusText = `Generating ${total} / ${total} | FINISHED`;
                        node.updateDisplayTime();
                        node.setDirtyCanvas(true, true);
                        return;
                    }

                    if (node.lastUpdateTime === 0) {
                        node.lastUpdateTime = now;
                        node.isCalculating = true;
                        node.statusText = `Generating ${displayIndex} / ${total} | CALCULATING...`;
                        node.setDirtyCanvas(true, true);
                        return;
                    }

                    const currentLoopDuration = now - node.lastUpdateTime;
                    node.lastUpdateTime = now;

                    if (currentLoopDuration > 1000) {
                        node.loopDurations.push(currentLoopDuration);
                        if (node.loopDurations.length > 5) node.loopDurations.shift();
                    }

                    const avgLoopTimeMs = node.loopDurations.length > 0 
                        ? node.loopDurations.reduce((a, b) => a + b, 0) / node.loopDurations.length 
                        : currentLoopDuration;

                    // Perfectly calibrated for 0-indexed loops
                    let remainingLoops = total - index;
                    if (remainingLoops < 0) remainingLoops = 0;

                    node.etaMs = remainingLoops * avgLoopTimeMs;
                    node.isCalculating = false;
                    
                    node.statusText = `Generating ${displayIndex} / ${total} | ${(avgLoopTimeMs/1000).toFixed(2)}s/loop`;
                    node.updateDisplayTime();
                    node.setDirtyCanvas(true, true);
                });
            });
        }
    }
});