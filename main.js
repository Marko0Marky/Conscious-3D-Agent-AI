// --- START OF FILE main.js ---
import { logger, showLoading, hideLoading, clamp, isFiniteVector } from './utils.js';
import { OntologicalWorldModel } from './owm.js';
import { LearningAI, StrategicAI } from './ai-agents.js';
import { ThreeDeeGame } from './three-dee-game.js';
import { NeuralNetworkVisualizer } from './nn-visualizer.js';

// Global for positioning vertex divs, not part of sheaf logic itself
const sheafVertexPositions = {
    0: {x:0.1, y:0.5}, 1: {x:0.3, y:0.2}, 2: {x:0.3, y:0.8},
    3: {x:0.9, y:0.5}, 4: {x:0.7, y:0.2}, 5: {x:0.7, y:0.8},
    6: {x:0.5, y:0.35}, 7: {x:0.5, y:0.65}
};

/**
 * The main application class, orchestrating the game, AI, and UI updates.
 */
class MainApp {
    constructor() {
        logger.info('MainApp Constructor started.');
        this.gameCanvas = document.getElementById('gameCanvas');
        this.sheafGraphCanvas = document.getElementById('sheafGraphCanvas');

        if (!this.gameCanvas || !this.sheafGraphCanvas) {
            document.getElementById('status').textContent = 'Error: Canvas not found. Check HTML IDs.';
            logger.error('Canvas not found');
            throw new Error('Canvas not found');
        }
        
        this.sheafGraphCtx = this.sheafGraphCanvas.getContext('2d');

        if (!this.sheafGraphCtx) {
            document.getElementById('status').textContent = 'Error: Failed to get 2D rendering context for sheaf canvas.';
            logger.error('Failed to get 2D context for sheaf');
            throw new Error('Failed to get 2D context for sheaf');
        }

        this.chartData = {
            qValue: [],
            predError: [],
            epsilon: [],
            score: []
        };
        this.MAX_CHART_POINTS = 100;

        this.applyCanvasDPR(this.sheafGraphCanvas, this.sheafGraphCtx);
        const gameCanvasRO = new ResizeObserver(entries => {
            for (let entry of entries) {
                const { width, height } = entry.contentRect;
                if (this.game) {
                    this.game.resize(width, height);
                }
            }
        });
        gameCanvasRO.observe(this.gameCanvas);
        
        this.isRunning = false;
        this.isFastForward = false;
        this.FAST_FORWARD_MULTIPLIER = 3;
        this.frameCount = 0;
        this.game = null; 

        this.boundGameLoop = this.gameLoop.bind(this);
        this.setupTooltips();
        this.bindEvents();
        this.setupQualiaAttentionPanel();
        document.getElementById('status').textContent = 'Initializing... Please wait.';

        this.setupGameAndAIs();
    }
    
    applyCanvasDPR(canvas, ctx) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
    }

    async setupGameAndAIs() {
        logger.info('setupGameAndAIs() started.');
        showLoading('game', 'Initializing 3D Environment...');
        showLoading('mainBrain', 'Building Main AI World Model...');
        showLoading('opponentBrain', 'Building Opponent AI World Model...');
        showLoading('metrics', 'Initializing OFTCC Sheaf...');

        try {
            this.game = new ThreeDeeGame(this.gameCanvas);
            logger.info('ThreeDeeGame initialized.');
            
            const STATE_DIM = 13;
            const ACTION_DIM = 4;

            this.mainAI_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, 7, [64, 64], true);
            this.opponent_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, 7, [64, 64], false);

            logger.info('Calling Promise.all for OWM initialization...');
            await Promise.all([
                this.mainAI_worldModel.initialize(),
                this.opponent_worldModel.initialize()
            ]);
            logger.info('OWM initialization completed.');

            this.mainAI = new LearningAI(this.mainAI_worldModel, this.game, true);
            this.opponentAI = new LearningAI(this.opponent_worldModel, this.game, false);
            
            this.mainStrategicAI = new StrategicAI(this.mainAI);
            this.opponentStrategicAI = new StrategicAI(this.opponentAI);

            this.mainViz = new NeuralNetworkVisualizer('nn-visualization-container', this.mainAI_worldModel, 'main');
            this.opponentViz = new NeuralNetworkVisualizer('nn-visualization-container-opponent', this.opponent_worldModel, 'opponent');

            this.game.render(); 
            this.updateVisualization();
            logger.info('Game and AIs initialized successfully. Awaiting user interaction.');
            document.getElementById('status').textContent = 'Ready. Click "Toggle Simulation" to start.';

        } catch (e) {
            logger.error('Failed to set up game and AIs:', e);
            document.getElementById('status').textContent = `Initialization Failed: ${e.message}`;
        } finally {
            hideLoading('game');
            hideLoading('mainBrain');
            hideLoading('opponentBrain');
            hideLoading('metrics');
        }
    }

    bindEvents() {
        document.getElementById('toggleSimButton').onclick = () => this.toggleGame();
        document.getElementById('resetSimButton').onclick = () => this.resetAI();
        document.getElementById('tuneButton').onclick = () => this.tuneParameters();
        document.getElementById('pauseButton').onclick = () => this.stop();
        document.getElementById('stepButton').onclick = () => this.gameLoop(null, true);
        document.getElementById('fastForwardButton').onclick = () => this.toggleFastForward();
        
        window.addEventListener('keydown', (e) => this.handleKeyDown(e));

        ['alphaSlider', 'betaSlider', 'gammaSlider'].forEach(id => {
            const slider = document.getElementById(id);
            const valueDisplay = document.getElementById(`${id.replace('Slider', '')}-param`);
            slider.addEventListener('input', () => {
                const paramName = id.replace('Slider', '');
                const value = parseFloat(slider.value);
                if (this.mainAI_worldModel?.qualiaSheaf) this.mainAI_worldModel.qualiaSheaf[paramName] = value;
                if (this.opponent_worldModel?.qualiaSheaf) this.opponent_worldModel.qualiaSheaf[paramName] = value;
                valueDisplay.textContent = value.toFixed(3);
                slider.setAttribute('aria-valuetext', value.toFixed(3));
            });
        });
    }

    setupTooltips() {
        tippy('#phi-display', { content: 'Φ (Phi) measures integrated information, indicating the system\'s level of consciousness or experience.' });
        tippy('#free-energy', { content: 'Free Energy (F) quantifies the system’s predictive divergence from its world model; the AI strives to minimize this.' });
        tippy('#prediction-error', { content: 'Prediction Error measures the discrepancy between the AI\'s predicted next state and the actual observed next state. High error can trigger "curiosity".' });
        tippy('#gestalt-unity', { content: 'Gestalt Unity quantifies the holistic coherence and interconnectedness across the sheaf structure, indicating how well its internal model is integrated.' });
        tippy('#h1-dimension', { content: 'dim H¹ represents the first cohomology dimension, indicating structural complexity and the presence of "holes" or non-trivial loops in the information flow within the sheaf.' });
        tippy('#inconsistency', { content: 'Gluing Inconsistency measures the misalignment in qualia projections between connected vertices (concepts) in the sheaf, indicating internal contradictions.' });
        tippy('#learning-rate', { content: 'The current learning rate (alpha) of the main AI\'s reinforcement learning algorithm.' });
        tippy('#epsilon-value', { content: 'The current exploration rate (epsilon) of the main AI\'s reinforcement learning algorithm. Higher values mean more random actions.' });
        tippy('#qualia-being-fill', { content: 'Being: Reflects the system’s fundamental existence and persistent state, representing its self-awareness.' });
        tippy('#qualia-intent-fill', { content: 'Intent: Captures directed action potential and goal-oriented focus, signifying its purpose and goals.' });
        tippy('#qualia-existence-fill', { content: 'Existence: Models state persistence and resilience to change, representing its current state and environmental robustness.' });
        tippy('#qualia-emergence-fill', { content: 'Emergence: Represents non-linear state synthesis and novel property formation, indicating its capacity for novelty.' });
        tippy('#qualia-gestalt-fill', { content: 'Gestalt: Quantifies holistic coherence and pattern recognition, indicating its ability to form meaningful wholes from parts.' });
        tippy('#context-fill', { content: 'Context: Incorporates environmental modulation and background information, reflecting its understanding of the surrounding world.' });
        tippy('#qualia-rel-emergence-fill', { content: 'Relational Emergence: Captures dynamic entity coupling and interaction effects, indicating its comprehension of relationships.' });
        tippy('#alphaSlider', { content: 'α (Alpha) controls the influence of external sensory input on qualia diffusion; higher values mean more responsiveness to environment.' });
        tippy('#betaSlider', { content: 'β (Beta) adjusts the overall diffusion strength and speed across the sheaf; higher values lead to faster qualia propagation and integration.' });
        tippy('#gammaSlider', { content: 'γ (Gamma) sets the inertia for qualia updates and acts as an effective learning rate for diffusion; higher values imply quicker adaptation.' });
        tippy('#toggleSimButton', { content: 'Toggles the simulation run/pause state. (Spacebar)' });
        tippy('#resetSimButton', { content: 'Resets the game, AI states, and world models to their initial configurations. (R key)' });
        tippy('#tuneButton', { content: 'Adaptively adjusts AI parameters (α, β, γ) based on real-time system stability and consciousness metrics, aiming for optimal performance. (T key)' });
        tippy('#pauseButton', { content: 'Pauses the simulation if it is currently running. (P key or Spacebar)' });
        tippy('#stepButton', { content: 'Advances the simulation by a single frame/step. (S key)' });
        tippy('#fastForwardButton', { content: 'Toggles fast forward mode, running multiple simulation steps per frame. (F key)' });

        tippy('#vertex-0', { content: 'Agent-X: The agent\'s X-axis position in the 3D world.' });
        tippy('#vertex-1', { content: 'Agent-Z: The agent\'s Z-axis position in the 3D world.' });
        tippy('#vertex-2', { content: 'Agent-Rot: The agent\'s current rotation (heading) on the Y-axis.' });
        tippy('#vertex-3', { content: 'Target-X: The target\'s X-axis position.' });
        tippy('#vertex-4', { content: 'Target-Z: The target\'s Z-axis position.' });
        tippy('#vertex-5', { content: 'Vec-DX: The X-component of the vector from the agent to the target.' });
        tippy('#vertex-6', { content: 'Vec-DZ: The Z-component of the vector from the agent to the target.' });
        tippy('#vertex-7', { content: 'Dist-Target: The direct distance from the agent to the target.' });
    }
    
    setupQualiaAttentionPanel() {
        const panel = document.getElementById('qualiaAttentionPanel');
        if (!panel) return;
        
        const qualiaNames = ['Being', 'Intent', 'Existence', 'Emergence', 'Gestalt', 'Context', 'Rel. Emergence'];
        let html = `<h4 id="qualia-attention-heading">Qualia Attention (Main AI)</h4>`;
        qualiaNames.forEach((name, i) => {
            html += `
                <div class="attention-bar-container">
                    <span class="attention-label">${name}</span>
                    <div class="attention-bar-wrapper">
                        <div class="attention-bar" id="attention-bar-${i}"></div>
                    </div>
                </div>
            `;
        });
        panel.innerHTML = html;
    }

    updateQualiaAttentionVisuals() {
        const softmaxScores = this.mainAI_worldModel?.lastSoftmaxScores;
        if (!softmaxScores || !isFiniteVector(softmaxScores)) return;

        for (let i = 0; i < softmaxScores.length; i++) {
            const bar = document.getElementById(`attention-bar-${i}`);
            if (bar) {
                const normalizedWidth = (softmaxScores[i] || 0) * 100;
                bar.style.width = `${clamp(normalizedWidth, 0, 100)}%`;
            }
        }
    }

    drawSheafGraph() {
        if (!this.sheafGraphCtx) return;
        this.sheafGraphCtx.clearRect(0, 0, this.sheafGraphCanvas.width, this.sheafGraphCanvas.height);

        const sheaf = this.mainAI_worldModel?.qualiaSheaf;
        if (!sheaf || !sheaf.graph || !sheaf.graph.vertices || !sheaf.graph.edges || !sheaf.adjacencyMatrix) {
            return;
        }

        const adj = sheaf.adjacencyMatrix;
        const { width, height } = this.sheafGraphCanvas;

        sheaf.graph.edges.forEach(([u, v]) => {
            const uIdx = sheaf.graph.vertices.indexOf(u);
            const vIdx = sheaf.graph.vertices.indexOf(v);
            if (!sheafVertexPositions[uIdx] || !sheafVertexPositions[vIdx] || uIdx === -1 || vIdx === -1) return;

            const weight = adj[uIdx]?.[vIdx] || 0.1;
            this.sheafGraphCtx.strokeStyle = `rgba(68, 170, 255, ${clamp(weight, 0.1, 1.0)})`;
            this.sheafGraphCtx.lineWidth = clamp(weight * 2, 0.5, 3.0);
            
            const p1 = sheafVertexPositions[uIdx], p2 = sheafVertexPositions[vIdx];
            const vertexEl = document.getElementById('vertex-0');
            const vWidth = vertexEl?.offsetWidth || 40;
            const vHeight = vertexEl?.offsetHeight || 24;

            this.sheafGraphCtx.beginPath();
            this.sheafGraphCtx.moveTo(p1.x * (width - vWidth) + vWidth / 2, p1.y * (height - vHeight) + vHeight / 2);
            this.sheafGraphCtx.lineTo(p2.x * (width - vWidth) + vHeight / 2, p2.y * (height - vHeight) + vHeight / 2);
            this.sheafGraphCtx.stroke();
        });
    }

    drawChart(svgId, data, color, yMin, yMax) {
        const svg = document.getElementById(svgId);
        if (!svg) return;
        svg.innerHTML = '';

        const width = svg.clientWidth;
        const height = svg.clientHeight;
        const padding = 10;

        if (data.length < 2) return;

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        let d = `M ${padding},${height - padding} `;
        
        for (let i = 0; i < data.length; i++) {
            const x = padding + i * (width - 2 * padding) / (data.length - 1);
            const y_val = Number.isFinite(data[i]) ? clamp(data[i], yMin, yMax) : (yMin + yMax) / 2; 
            const y = (height - 2 * padding) * (1 - (y_val - yMin) / (yMax - yMin)) + padding;
            d += `L ${x.toFixed(2)},${y.toFixed(2)} `;
        }

        path.setAttribute('d', d);
        path.setAttribute('stroke', color);
        path.setAttribute('stroke-width', '2');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke-linejoin', 'round');
        svg.appendChild(path);
    }

    updatePerformanceCharts() {
        if(this.mainAI && this.game) {
            this.chartData.qValue.push(this.mainAI.avgQValue);
            this.chartData.epsilon.push(this.mainAI.epsilon);
            this.chartData.predError.push(this.mainAI_worldModel.predictionError);
            this.chartData.score.push(this.game.score.ai - this.game.score.player);
        }

        for(const key in this.chartData) {
            if (this.chartData[key].length > this.MAX_CHART_POINTS) {
                this.chartData[key].shift();
            }
        }
        
        this.drawChart('qValueChart', this.chartData.qValue, 'var(--primary-blue)', -2, 2);
        this.drawChart('predErrorChart', this.chartData.predError, 'var(--warn-orange)', 0, 5);
        this.drawChart('epsilonChart', this.chartData.epsilon, 'var(--info-green)', 0, 1);
        this.drawChart('scoreChart', this.chartData.score, 'var(--error-red)', -10, 10);
    }


    updateVisualization() {
        const qualia = this.mainAI_worldModel?.qualiaSheaf;
        const model = this.mainAI_worldModel;
        const mainAI = this.mainAI;

        if (!qualia || !model || !mainAI) return;

        const avgQualia = new Float32Array(qualia.qDim).fill(0);
        let count = 0;
        qualia.stalks.forEach((stalk) => {
            if (isFiniteVector(stalk)) {
                stalk.forEach((v, i) => { if (Number.isFinite(v)) avgQualia[i] += v; });
                count++;
            }
        });
        if (count > 0) avgQualia.forEach((_, i) => avgQualia[i] /= count);
        const qualiaValues = avgQualia.map(v => Number.isFinite(v) ? clamp(v, 0, 1) : 0);

        requestAnimationFrame(() => {
            document.getElementById('being-value').textContent = qualiaValues[0].toFixed(3);
            document.getElementById('qualia-being-fill').style.width = `${qualiaValues[0] * 100}%`;
            document.getElementById('intent-value').textContent = qualiaValues[1].toFixed(3);
            document.getElementById('qualia-intent-fill').style.width = `${qualiaValues[1] * 100}%`;
            document.getElementById('existence-value').textContent = qualiaValues[2].toFixed(3);
            document.getElementById('qualia-existence-fill').style.width = `${qualiaValues[2] * 100}%`;
            document.getElementById('emergence-value').textContent = qualiaValues[3].toFixed(3);
            document.getElementById('qualia-emergence-fill').style.width = `${qualiaValues[3] * 100}%`;
            document.getElementById('gestalt-value').textContent = qualiaValues[4].toFixed(3);
            document.getElementById('qualia-gestalt-fill').style.width = `${qualiaValues[4] * 100}%`;
            document.getElementById('context-value').textContent = qualiaValues[5].toFixed(3);
            document.getElementById('qualia-context-fill').style.width = `${qualiaValues[5] * 100}%`;
            document.getElementById('rel-emergence-value').textContent = qualiaValues[6].toFixed(3);
            document.getElementById('qualia-rel-emergence-fill').style.width = `${qualiaValues[6] * 100}%`;

            document.getElementById('phi-display').textContent = `Φ: ${clamp(qualia.phi, 0, 5).toFixed(5)}`;
            document.getElementById('free-energy').textContent = (model.freeEnergy || 0).toFixed(5);
            document.getElementById('prediction-error').textContent = (model.predictionError || 0).toFixed(5);
            document.getElementById('h1-dimension').textContent = clamp(qualia.h1Dimension, 0, 3).toFixed(2);
            document.getElementById('gestalt-unity').textContent = clamp(qualia.gestaltUnity, 0, 1).toFixed(5);
            document.getElementById('inconsistency').textContent = (qualia.inconsistency || 0).toFixed(5);
            document.getElementById('learning-rate').textContent = (mainAI.learningRate || 0).toFixed(4);
            document.getElementById('epsilon-value').textContent = (mainAI.epsilon || 0).toFixed(3);
            document.getElementById('stability-fill').style.width = `${clamp(qualia.stability, 0, 1) * 100}%`;
            document.getElementById('alpha-param').textContent = qualia.alpha.toFixed(3);
            document.getElementById('alphaSlider').value = qualia.alpha;
            document.getElementById('beta-param').textContent = qualia.beta.toFixed(3);
            document.getElementById('betaSlider').value = qualia.beta;
            document.getElementById('gamma-param').textContent = qualia.gamma.toFixed(3);
            document.getElementById('gammaSlider').value = qualia.gamma;

            qualia.visualizeActivity();
            this.drawSheafGraph();
            this.updateQualiaAttentionVisuals();
        });
    }

    async gameLoop(timestamp, isManualStep = false) {
        if (!this.isRunning && !isManualStep) return;
        
        if (!this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready || !this.game) {
            document.getElementById('status').textContent = 'Waiting for AI/Game initialization...';
            if (!isManualStep) requestAnimationFrame(this.boundGameLoop);
            return; 
        }

        this.game.scene.updateMatrixWorld(true);

        try {
            const stepsPerFrame = isManualStep ? 1 : (this.isFastForward ? this.FAST_FORWARD_MULTIPLIER : 1);
            let mainDecision, opponentDecision, gameUpdateResult, postGameState;

            for (let step = 0; step < stepsPerFrame; step++) {
                this.frameCount++;
                const preGameState = this.game.getState();
                
                [mainDecision, opponentDecision] = await Promise.all([
                    this.mainAI.makeDecision(preGameState),
                    this.opponentAI.makeDecision(preGameState)
                ]);

                if (mainDecision.corrupted) this.mainAI.reset();
                if (opponentDecision.corrupted) this.opponentAI.reset();

                this.game.setAIAction(mainDecision.action);
                this.game.setPlayerAction(opponentDecision.action);
                
                gameUpdateResult = this.game.update();
                postGameState = this.game.getState();

                await Promise.all([
                    this.mainAI.learn(gameUpdateResult.aReward, postGameState, gameUpdateResult.isDone),
                    this.opponentAI.learn(gameUpdateResult.pReward, postGameState, gameUpdateResult.isDone)
                ]);
                
                this.mainStrategicAI.observe(gameUpdateResult.aReward);
                this.opponentStrategicAI.observe(gameUpdateResult.pReward);

                if(step === stepsPerFrame - 1 && this.frameCount % 50 === 0) {
                    this.mainStrategicAI.modulateParameters();
                    this.opponentStrategicAI.modulateParameters();
                }
            }

            document.getElementById('player-score').textContent = this.game.score.player;
            document.getElementById('ai-score').textContent = this.game.score.ai;

            if (this.frameCount % 5 === 0 || isManualStep) {
                this.mainViz.update(mainDecision.activations, mainDecision.chosenActionIndex);
                this.opponentViz.update(opponentDecision.activations, opponentDecision.chosenActionIndex);
                this.updateVisualization();
            }
            if (this.frameCount % 20 === 0 || isManualStep) {
                this.updatePerformanceCharts();
            }

            this.game.render();

        } catch (error) {
            logger.error("Error in game loop, stopping simulation:", error.message || error.toString() || error);
            console.error("Critical error in gameLoop, stopping:", error);
            this.stop(); 
        } finally {
            if (this.isRunning && !isManualStep) {
                requestAnimationFrame(this.boundGameLoop);
            }
        }
    }

    toggleGame() {
        if (!this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready) {
            logger.warn('AIs are not fully initialized yet. Please wait.');
            document.getElementById('status').textContent = 'AIs Not Ready - Please Wait...';
            return;
        }
        this.isRunning = !this.isRunning;
        const btn = document.getElementById('toggleSimButton');
        btn.textContent = this.isRunning ? '⏸️ Pause Simulation' : '🚀 Toggle Simulation';

        if (this.isRunning) {
            document.getElementById('status').textContent = this.isFastForward ? 'Fast Forward Active' : 'Conscious AI Active';
            if (this.game && this.game.audioContext.state === 'suspended') {
                this.game.resumeAudioContext().then(() => this.gameLoop(null, false));
            } else {
                this.gameLoop(null, false);
            }
            logger.info('Simulation started.');
        } else {
            if (this.isFastForward) {
                this.isFastForward = false;
                const ffBtn = document.getElementById('fastForwardButton');
                ffBtn.classList.remove('active');
                ffBtn.innerHTML = 'Fast ⏩';
            }
            document.getElementById('status').textContent = 'Paused';
            logger.info('Simulation paused.');
        }
    }

    toggleFastForward() {
        if (!this.mainAI_worldModel?.ready) {
            logger.warn('AIs not ready, cannot toggle fast forward.');
            return;
        }
        this.isFastForward = !this.isFastForward;
        const btn = document.getElementById('fastForwardButton');
        btn.classList.toggle('active', this.isFastForward);
        btn.innerHTML = this.isFastForward ? 'Normal Speed 🐢' : 'Fast ⏩';

        if (this.isFastForward && !this.isRunning) {
            this.toggleGame();
        } else {
            if(this.isRunning) {
               document.getElementById('status').textContent = this.isFastForward ? 'Fast Forward Active' : 'Conscious AI Active';
            }
        }
        logger.info(`Fast forward toggled ${this.isFastForward ? 'ON' : 'OFF'}.`);
    }

    stop() {
        this.isRunning = false;
        document.getElementById('toggleSimButton').textContent = '🚀 Toggle Simulation';
        if (this.isFastForward) {
            this.isFastForward = false;
            const ffBtn = document.getElementById('fastForwardButton');
            ffBtn.classList.remove('active');
            ffBtn.innerHTML = 'Fast ⏩';
        }
        document.getElementById('status').textContent = 'Paused';
        logger.info('Simulation stopped.');
    }

    async resetAI() {
        this.stop();
        logger.info('Resetting all game and AI states...');
        showLoading('game', 'Resetting Game...');
        showLoading('mainBrain', 'Resetting Main AI...');
        showLoading('opponentBrain', 'Resetting Opponent AI...');
        showLoading('metrics', 'Resetting Metrics...');
        
        this.isFastForward = false;
        const ffBtn = document.getElementById('fastForwardButton');
        ffBtn.classList.remove('active');
        ffBtn.innerHTML = 'Fast ⏩';

        for (const key in this.chartData) this.chartData[key] = [];

        try {
            this.game = new ThreeDeeGame(this.gameCanvas); 
            document.getElementById('player-score').textContent = 0;
            document.getElementById('ai-score').textContent = 0;
            
            const STATE_DIM = 13;
            const ACTION_DIM = 4;

            this.mainAI_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, 7, [64, 64], true);
            await this.mainAI_worldModel.initialize();
            this.mainAI = new LearningAI(this.mainAI_worldModel, this.game, true);
            this.mainStrategicAI = new StrategicAI(this.mainAI);
            
            this.opponent_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, 7, [64, 64], false);
            await this.opponent_worldModel.initialize();
            this.opponentAI = new LearningAI(this.opponent_worldModel, this.game, false);
            this.opponentStrategicAI = new StrategicAI(this.opponentAI);
            
            this.mainViz = new NeuralNetworkVisualizer('nn-visualization-container', this.mainAI_worldModel, 'main');
            this.opponentViz = new NeuralNetworkVisualizer('nn-visualization-container-opponent', this.opponent_worldModel, 'opponent');

            this.frameCount = 0;
            document.getElementById('status').textContent = 'Reset Complete. Click "Toggle Simulation" to start.'; 
            this.updateVisualization();
            this.game.render();
            logger.info('All states reset successfully. Awaiting user interaction.');

        } catch (e) {
            logger.error('Error during reset:', e);
            document.getElementById('status').textContent = `Reset Failed: ${e.message}`;
        } finally {
            hideLoading('game');
            hideLoading('mainBrain');
            hideLoading('opponentBrain');
            hideLoading('metrics');
        }
    }

    tuneParameters() {
        if (!this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready) return;
        this.mainAI_worldModel.qualiaSheaf.tuneParameters();
        this.opponent_worldModel.qualiaSheaf.tuneParameters();

        const paramIds = ['alpha-param', 'beta-param', 'gamma-param'];
        paramIds.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.classList.remove('param-flash-active');
                void el.offsetWidth;
                el.classList.add('param-flash-active');
            }
        });

        this.updateVisualization();
    }
    
    handleKeyDown(e) {
        switch (e.key.toLowerCase()) {
            case ' ': e.preventDefault(); this.toggleGame(); break;
            case 'r': e.preventDefault(); this.resetAI(); break;
            case 't': e.preventDefault(); this.tuneParameters(); break;
            case 'p': e.preventDefault(); this.stop(); break;
            case 's': e.preventDefault(); if (!this.isRunning) this.gameLoop(null, true); break;
            case 'f': e.preventDefault(); this.toggleFastForward(); break;
        }
    }
}

window.onload = async () => {
    function positionVertices() {
        const graph = document.getElementById('sheafGraph');
        if (!graph) return;
        const rect = graph.getBoundingClientRect();
        const v_count = 8;
        for (let i = 0; i < v_count; i++) {
            const v = document.getElementById('vertex-' + i);
            if (!v) continue;
            const p = sheafVertexPositions[i] || { x: 0.5, y: 0.5 };
            v.style.left = Math.round(p.x * (rect.width - v.offsetWidth)) + 'px';
            v.style.top = Math.round(p.y * (rect.height - v.offsetHeight)) + 'px';
        }
    }

    window.addEventListener('resize', positionVertices);

    try {
        const app = new MainApp();
        positionVertices();
        logger.info('UI initialized — ready. Awaiting user interaction.');
    } catch (e) {
        document.getElementById('status').textContent = `Initialization Error: ${e.message}`;
        logger.error('Initialization failed', e.message);
    }
};
// --- END OF FILE main.js ---
