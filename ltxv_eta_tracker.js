import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "triXope.LTXV_ETA_Display",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LTXV_ETA_Display") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                
                this.displayTime = "00:00:00";
                this.elapsedStr = "00:00:00";
                this.statusText = "Awaiting Connection...";
                this.passText = "";
                
                this.overallProgress = 0.0;
                this.etaMs = 0;
                this.target_ltxv_id = null;
                
                this.serialize_widgets = true; 
                this.size = [350, 160]; 
            };

            nodeType.prototype.onDrawForeground = function (ctx) {
                if (this.flags.collapsed) return;

                const margin = 10;
                const y = 30; 
                const x = margin;
                const w = this.size[0] - margin * 2;
                const h = this.size[1] - y - margin; 

                // Draw Background
                ctx.fillStyle = "#1a1a1a";
                ctx.beginPath();
                ctx.roundRect(x, y, w, h, 5);
                ctx.fill();

                // Draw Progress Bar
                if (this.overallProgress > 0) {
                    const progressWidth = w * this.overallProgress;
                    ctx.fillStyle = "#204a20"; // Deep Green
                    ctx.beginPath();
                    ctx.roundRect(x, y, progressWidth, h, 5);
                    ctx.fill();
                }

                ctx.strokeStyle = "#444";
                ctx.strokeRect(x, y, w, h);

                const etaFontSize = Math.max(16, Math.min(w / 6.5, h * 0.30));
                const smallFontSize = Math.max(10, etaFontSize * 0.35);
                const centerY = y + (h / 2);
                ctx.textBaseline = "middle"; 
                ctx.textAlign = "center";

                // Elapsed Time
                ctx.fillStyle = "#aaaaaa";
                ctx.font = `${smallFontSize}px Arial`;
                ctx.fillText(`Elapsed: ${this.elapsedStr}`, x + w/2, centerY - etaFontSize * 1.2);

                // Main ETA Display
                ctx.fillStyle = "#00FF00"; 
                if (this.overallProgress === 0) {
                    ctx.font = `bold ${etaFontSize * 0.6}px monospace`;
                    ctx.fillText("WAITING...", x + w/2, centerY - 5);
                } else if (this.overallProgress >= 1) {
                    ctx.font = `bold ${etaFontSize * 0.8}px monospace`;
                    ctx.fillText("FINISHED", x + w/2, centerY - 5);
                } else {
                    const seconds = Math.floor(this.etaMs / 1000);
                    
                    if (seconds <= 0) {
                        ctx.font = `bold ${etaFontSize * 0.55}px monospace`;
                        ctx.fillText("CALCULATING...", x + w/2, centerY - 5);
                    } else {
                        ctx.font = `bold ${etaFontSize}px monospace`;
                        const h_str = Math.floor(seconds / 3600).toString().padStart(2, '0');
                        const m_str = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
                        const s_str = (seconds % 60).toString().padStart(2, '0');
                        
                        ctx.fillText(`ETA ${h_str}:${m_str}:${s_str}`, x + w/2, centerY - 5);
                    }
                }

                // Subtext / Current Pass
                ctx.fillStyle = "#FFFFFF";
                ctx.font = `bold ${smallFontSize * 1.1}px Arial`;
                ctx.fillText(this.passText, x + w/2, centerY + etaFontSize * 0.7);

                ctx.fillStyle = "#888888";
                ctx.font = `${smallFontSize}px Arial`;
                ctx.fillText(this.statusText, x + w/2, centerY + etaFontSize * 1.3);
            };

            let globalStartTime = 0;
            let isExecuting = false;
            let elapsedInterval = null;

            api.addEventListener("execution_start", () => {
                globalStartTime = performance.now();
                isExecuting = true;

                app.graph._nodes.filter(n => n.type === "LTXV_ETA_Display").forEach(node => {
                    node.overallProgress = 0;
                    node.etaMs = 0;
                    node.passText = "Initializing...";
                    node.statusText = "";

                    node.stepTimes = []; 
                    node.lastStepTimestamp = 0; 
                    node.currentPassName = ""; 

                    node.target_ltxv_id = null;
                    if (node.inputs[0] && node.inputs[0].link) {
                        const link = app.graph.links[node.inputs[0].link];
                        if (link) {
                            const originNode = app.graph.getNodeById(link.origin_id);
                            if (originNode) {
                                node.target_ltxv_id = originNode.id.toString();
                            }
                        }
                    }
                    node.setDirtyCanvas(true, true);
                });

                if (!elapsedInterval) {
                    elapsedInterval = setInterval(() => {
                        if (isExecuting) {
                            const now = performance.now();
                            const elapsedSecs = Math.floor((now - globalStartTime) / 1000);
                            const h = Math.floor(elapsedSecs / 3600).toString().padStart(2, '0');
                            const m = Math.floor((elapsedSecs % 3600) / 60).toString().padStart(2, '0');
                            const s = (elapsedSecs % 60).toString().padStart(2, '0');
                            
                            app.graph._nodes.filter(n => n.type === "LTXV_ETA_Display").forEach(n => {
                                n.elapsedStr = `${h}:${m}:${s}`;

                                if (n.etaMs > 0 && n.overallProgress > 0 && n.overallProgress < 1.0) {
                                    n.etaMs = Math.max(0, n.etaMs - 1000);
                                }

                                n.setDirtyCanvas(true, true); 
                            });
                        }
                    }, 1000);
                }
            });

            api.addEventListener("executing", (event) => {
                if (!event.detail) isExecuting = false; 
            });

            api.addEventListener("ltxv_eta_update", (event) => {
                const data = event.detail;
                const incomingId = Array.isArray(data.node_id) ? data.node_id[0] : data.node_id;
                const now = performance.now();

                app.graph._nodes.filter(n => n.type === "LTXV_ETA_Display").forEach(node => {
                    if (node.target_ltxv_id !== incomingId.toString()) return;

                    const rawProgress = data.global_step / data.total_global_steps;
                    node.overallProgress = Math.min(rawProgress, 0.99);

                    if (node.currentPassName !== data.pass_name) {
                        node.stepTimes = []; 
                        node.currentPassName = data.pass_name;
                    } else if (node.lastStepTimestamp > 0) {
                        const stepDuration = now - node.lastStepTimestamp;

                        if (stepDuration < 60000) { 
                            node.stepTimes.push(stepDuration);
                        }

                        if (node.stepTimes.length > 4) {
                            node.stepTimes.shift();
                        }
                    }
                    node.lastStepTimestamp = now;

                    const remainingSteps = data.total_global_steps - data.global_step;

                    if (node.stepTimes.length > 0) {
                        const avgStepTime = node.stepTimes.reduce((a, b) => a + b, 0) / node.stepTimes.length;
                        node.etaMs = avgStepTime * remainingSteps;
                    } else if (node.etaMs === 0 && node.overallProgress > 0) {
                        const elapsed = now - globalStartTime;
                        const totalEstimated = elapsed / node.overallProgress;
                        node.etaMs = totalEstimated - elapsed;
                    }

                    const displayGlobal = Math.min(data.global_step, data.total_global_steps);
                    
                    if (data.chunk === data.total_chunks && data.step >= data.total_steps) {
                        node.passText = "UNet Diffusion Complete";
                        if (data.is_face_restore) {
                            node.statusText = "Decoding VAE & Restoring Faces...";
                        } else {
                            node.statusText = "Decoding VAE...";
                        }
                    } else {
                        node.passText = `${data.pass_name} (Chunk ${data.chunk} of ${data.total_chunks})`;
                        node.statusText = `Step ${data.step} of ${data.total_steps} (Global ${displayGlobal}/${data.total_global_steps})`;
                    }
                    
                    node.setDirtyCanvas(true, true);
                });
            });

            api.addEventListener("executing", (event) => {
                if (!event.detail) {
                    isExecuting = false; 
                    app.graph._nodes.filter(n => n.type === "LTXV_ETA_Display").forEach(node => {
                        if (node.overallProgress > 0) {
                            node.overallProgress = 1.0;
                            node.etaMs = 0;
                            node.passText = "Execution Complete";
                            node.statusText = "Finished";
                            node.setDirtyCanvas(true, true);
                        }
                    });
                }
            });
        }
    }
});
