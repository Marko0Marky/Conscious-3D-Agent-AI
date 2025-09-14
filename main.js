// --- START OF FILE main.js ---
import { logger, showLoading, hideLoading, clamp, isFiniteVector, norm2, vecZeros, vecAdd } from './utils.js';
import { OntologicalWorldModel } from './owm.js';
import { LearningAIAgent, StrategicAI } from './ai-agents.js';
import { ThreeDeeGame } from './three-dee-game.js';
import { NeuralNetworkVisualizer } from './nn-visualizer.js';
import { initLive2D, updateLive2DEmotions, cleanupLive2D, isLive2DReady, updateLive2D } from './viz-live2d.js';
import { initConceptVisualization, cleanupConceptVisualization, renderConceptVisualization, animateConceptNodes, isConceptVisualizationReady, updateAgentSimulationVisuals } from './viz-concepts.js';
import { EnhancedQualiaSheaf } from './qualia-sheaf.js'; // Import EnhancedQualiaSheaf

import * as THREE from 'three';

const tf = window.tf;

const sheafVertexPositions = {
    0: {x:0.1, y:0.5}, 1: {x:0.3, y:0.2}, 2: {x:0.3, y:0.8},
    3: {x:0.9, y:0.5}, 4: {x:0.7, y:0.2}, 5: {x:0.5, y:0.35},
    6: {x:0.7, y:0.8}, 7: {x:0.5, y:0.65}
};

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
        this.clock = new THREE.Clock();
        this.chartData = { stateValue: [], predError: [], epsilon: [], score: [] };
        this.chartEMA = { stateValue: 0, predError: 0, epsilon: 0, score: 0 };
        this.EMA_ALPHA = 0.1;
        this.MAX_CHART_POINTS = 100;
        this.applyCanvasDPR(this.sheafGraphCanvas, this.sheafGraphCtx);
        new ResizeObserver(() => this.applyCanvasDPR(this.sheafGraphCanvas, this.sheafGraphCtx)).observe(document.getElementById('sheafGraph'));

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
        this.sheafStepCount = 0;
        this.sheafAdaptFrequency = 100;
        this.consecutiveZeroLosses = 0;
        this.MAX_ZERO_LOSSES = 50;
        this.game = null;
        this.lastPhi = 0;
        this.boundGameLoop = this.gameLoop.bind(this);
        this.bindEvents();
        this.setupTooltips();
        this.setupQualiaAttentionPanel();
        document.getElementById('status').textContent = 'Initializing... Please wait.';
        this.setupGameAndAIs();
    }

    applyCanvasDPR(canvas, ctx) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement ? canvas.parentElement.getBoundingClientRect() : canvas.getBoundingClientRect();
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
        showLoading('metrics', 'Initializing Qualia Sheaf...');

        try {
            this.game = new ThreeDeeGame(this.gameCanvas);
            logger.info('ThreeDeeGame initialized.');
            const STATE_DIM = 13;
            const ACTION_DIM = 4;
            const Q_DIM = 7;

            // 1. Create the EnhancedQualiaSheaf instances FIRST
            this.mainAI_qualiaSheaf = new EnhancedQualiaSheaf(null, STATE_DIM, Q_DIM, 0.1, 0.1, 0.05);
            this.opponent_qualiaSheaf = new EnhancedQualiaSheaf(null, STATE_DIM, Q_DIM, 0.1, 0.1, 0.05);

            // 2. Initialize the Sheaf instances
            logger.info("Calling Promise.all for QualiaSheaf initialization...");
            await Promise.all([
                this.mainAI_qualiaSheaf.initialize(),
                this.opponent_qualiaSheaf.initialize()
            ]);
            logger.info("Core QualiaSheafs initialized successfully.");


            // 3. Create OntologicalWorldModel instances, passing the initialized Sheafs
            this.mainAI_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], false, this.mainAI_qualiaSheaf);
            this.opponent_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], true, this.opponent_qualiaSheaf);

            // 4. Set the OWM reference back in each Sheaf instance (now that OWMs exist)
            this.mainAI_qualiaSheaf.setOWM(this.mainAI_worldModel);
            this.opponent_qualiaSheaf.setOWM(this.opponent_worldModel);


            let retries = 3;
            while (retries > 0) {
                try {
                    logger.info("Calling Promise.all for OWM initialization...");
                    await Promise.all([
                        this.mainAI_worldModel.initialize(),
                        this.opponent_worldModel.initialize()
                    ]);
                    if (this.mainAI_worldModel.ready && this.opponent_worldModel.ready) {
                        logger.info("Core OWMs are ready.");
                        break;
                    }
                    throw new Error('OWM not ready after initialization');
                } catch (e) {
                    retries--;
                    logger.warn(`OWM initialization failed, retries left: ${retries}`, e);
                    if (retries === 0) throw e;
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            }


            logger.info("Visualizations initializing...");
            await Promise.all([
                initLive2D(this.clock, this.mainAI_qualiaSheaf),
                initConceptVisualization(this.clock, this.mainAI_qualiaSheaf)
            ]);
            logger.info("Visualizations initialized successfully.");

            // 5. Instantiate LearningAIAgent agents, passing the initialized OWM instances
            this.mainAI = new LearningAIAgent(STATE_DIM, ACTION_DIM, this.game, 3, this.mainAI_worldModel);
            this.opponentAI = new LearningAIAgent(STATE_DIM, ACTION_DIM, this.game, 3, this.opponent_worldModel);
            
            this.mainStrategicAI = new StrategicAI(this.mainAI);
            this.opponentStrategicAI = new StrategicAI(this.opponentAI);
            this.mainViz = new NeuralNetworkVisualizer('nn-visualization-container', this.mainAI_worldModel, 'main');
            this.opponentViz = new NeuralNetworkVisualizer('nn-visualization-container-opponent', this.opponent_worldModel, 'opponent');

            this.game.render();
            this.updateVisualization();
            logger.info('Game and AIs initialized successfully.');
            logger.info(`OWM Readiness Check (End of setupGameAndAIs): MainAI_OWM.ready = ${this.mainAI_worldModel.ready}, OpponentAI_OWM.ready = ${this.opponent_worldModel.ready}`);
            document.getElementById('status').textContent = 'Ready. Click "Toggle Sim" to start.';

        } catch (e) {
            logger.error('Failed to set up game and AIs:', e);
            document.getElementById('status').textContent = `Initialization Failed: ${e.message}`;
            if (e.message.includes('OWM')) {
                document.getElementById('status').textContent = 'Initialization Failed: Critical failure: OWMs not initialized';
            } else if (e.message.includes('ThreeDeeGame')) {
                document.getElementById('status').textContent = 'Initialization Failed: Game engine not initialized';
            } else if (e.message.includes('QualiaSheaf')) {
                 document.getElementById('status').textContent = 'Initialization Failed: Critical failure: QualiaSheaf not initialized';
            }
        } finally {
            hideLoading('game');
            hideLoading('mainBrain');
            hideLoading('opponentBrain');
            hideLoading('metrics');
        }
    }

    mapEmotionsForAvatar(worldModel) {
        const qualia = worldModel.qualiaSheaf;
        const h1Influence = clamp(qualia.h1Dimension / (qualia.graph.edges.length || 1), 0, 1);
        const joy = clamp(qualia.gestaltUnity * qualia.stability, 0, 1);
        const fear = clamp(1 - qualia.stability, 0, 1);
        const curiosity = clamp(worldModel.predictionError / 5.0 + h1Influence * 0.5, 0, 1);
        const frustration = clamp(qualia.inconsistency + (worldModel.actorLoss || 0), 0, 1);
        const calm = clamp(qualia.stability * (1 - worldModel.freeEnergy / 2.0), 0, 1);
        const surprise = clamp(Math.abs(qualia.phi - this.lastPhi) / 0.1, 0, 1);
        this.lastPhi = qualia.phi;
        const emotionVec = [joy, fear, curiosity, frustration, calm, surprise];

        let emotionWrapper = { arraySync: () => [emotionVec], isDisposed: false, tensor: null };

        if (tf) {
            const tensor = tf.tensor([emotionVec]);
            emotionWrapper.tensor = tensor;
            emotionWrapper.arraySync = () => tensor.arraySync();
            emotionWrapper.dispose = () => {
                if (emotionWrapper.tensor && !emotionWrapper.isDisposed) {
                    tensor.dispose();
                    emotionWrapper.isDisposed = true;
                }
            };
        }
        return emotionWrapper;
    }

    updateQualiaBars(qualiaSheaf) {
        if (!qualiaSheaf) return;
        const entityNames = qualiaSheaf.entityNames.map(name => name.replace('_', '-'));
        const qDim = qualiaSheaf.qDim;
        const numVertices = qualiaSheaf.graph.vertices.length;
        if (numVertices === 0) return;

        const aggregateStalk = vecZeros(qDim);
        for (const stalk of qualiaSheaf.stalks.values()) {
            if (isFiniteVector(stalk)) {
                for (let i = 0; i < qDim; i++) {
                    aggregateStalk[i] += stalk[i];
                }
            }
        }

        for (let i = 0; i < qDim; i++) {
            const avgValue = aggregateStalk[i] / numVertices;
            const clampedValue = clamp(avgValue, 0, 1);
            const fillElement = document.getElementById(`qualia-${entityNames[i]}-fill`);
            const valueElement = document.getElementById(`${entityNames[i]}-value`);
            if (fillElement) fillElement.style.width = `${clampedValue * 100}%`;
            if (valueElement) valueElement.textContent = clampedValue.toFixed(3);
        }
    }

    updateVisualization() {
        const qualia = this.mainAI_worldModel?.qualiaSheaf;
        if (!qualia || !this.mainAI_worldModel || !this.mainAI) return;

        this.updateQualiaBars(qualia);

        const avgQualia = vecZeros(qualia.qDim);
        let count = 0;
        qualia.stalks.forEach((stalk) => {
            if (isFiniteVector(stalk)) {
                stalk.forEach((v, i) => { if (Number.isFinite(v)) avgQualia[i] += v; });
                count++;
            }
        });
        if (count > 0) avgQualia.forEach((_, i) => avgQualia[i] /= count);
        const qualiaValues = avgQualia.map(v => Number.isFinite(v) ? clamp(v, 0, 1) : 0);

        document.getElementById('being-value').textContent = qualiaValues[0].toFixed(3);
        document.getElementById('intent-value').textContent = qualiaValues[1].toFixed(3);
        document.getElementById('existence-value').textContent = qualiaValues[2].toFixed(3);
        document.getElementById('emergence-value').textContent = qualiaValues[3].toFixed(3);
        document.getElementById('gestalt-value').textContent = qualiaValues[4].toFixed(3);
        document.getElementById('context-value').textContent = qualiaValues[5].toFixed(3);
        document.getElementById('rel-emergence-value').textContent = qualiaValues[6].toFixed(3);

        requestAnimationFrame(() => {
            document.getElementById('phi-display').textContent = `Φ: ${clamp(qualia.phi, 0, 5).toFixed(5)}`;
            document.getElementById('free-energy').textContent = (this.mainAI_worldModel.freeEnergy || 0).toFixed(5);
            document.getElementById('prediction-error').textContent = (this.mainAI_worldModel.predictionError || 0).toFixed(5);
            document.getElementById('h1-dimension').textContent = clamp(qualia.h1Dimension, 0, qualia.graph.edges.length).toFixed(2);
            document.getElementById('gestalt-unity').textContent = clamp(qualia.gestaltUnity, 0, 1).toFixed(5);
            document.getElementById('inconsistency').textContent = (qualia.inconsistency || 0).toFixed(5);
            document.getElementById('learning-rate').textContent = (this.mainAI.lr || 0).toFixed(4);
            document.getElementById('epsilon-value').textContent = (this.mainAI.epsilon || 0).toFixed(3);

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

            if (isLive2DReady()) {
                const emotionWrapper = this.mapEmotionsForAvatar(this.mainAI_worldModel);
                updateLive2DEmotions(emotionWrapper, this.game.score.ai > this.game.score.player ? 'nod' : 'idle');
                updateLive2D(this.clock.getDelta());
                if (emotionWrapper.dispose) emotionWrapper.dispose();
            }
            if (isConceptVisualizationReady()) {
                const gameState = this.game.getState() || {};
                const augmentedGameState = {
                    ...gameState,
                    dist: gameState.ai && gameState.aiTarget ? Math.sqrt(Math.pow(gameState.aiTarget.x - gameState.ai.x, 2) + Math.pow(gameState.aiTarget.z - gameState.ai.z, 2)) : 0
                };
                updateAgentSimulationVisuals(this.mainAI_qualiaSheaf, augmentedGameState, qualia.phi);
                animateConceptNodes(this.clock.getDelta());
                renderConceptVisualization();
            }
        });
    }

    bindEvents() {
        const toggleSimButton = document.getElementById('toggleSimButton');
        if (toggleSimButton) toggleSimButton.onclick = () => this.toggleGame();
        const resetSimButton = document.getElementById('resetSimButton');
        if (resetSimButton) resetSimButton.onclick = () => this.resetAI();
        const tuneButton = document.getElementById('tuneButton');
        if (tuneButton) tuneButton.onclick = () => this.tuneParameters();
        const pauseButton = document.getElementById('pauseButton');
        if (pauseButton) pauseButton.onclick = () => this.stop();
        const stepButton = document.getElementById('stepButton');
        if (stepButton) stepButton.onclick = () => this.gameLoop(null, true);
        const fastForwardButton = document.getElementById('fastForwardButton');
        if (fastForwardButton) fastForwardButton.onclick = () => this.toggleFastForward();

        window.addEventListener('keydown', (e) => this.handleKeyDown(e));

        ['alphaSlider', 'betaSlider', 'gammaSlider'].forEach(id => {
            const slider = document.getElementById(id);
            const valueDisplay = document.getElementById(`${id.replace('Slider', '')}-param`);
            if (slider && valueDisplay) {
                slider.addEventListener('input', () => {
                    const paramName = id.replace('Slider', '');
                    const value = parseFloat(slider.value);
                    if (this.mainAI_qualiaSheaf) this.mainAI_qualiaSheaf[paramName] = value;
                    if (this.opponent_qualiaSheaf) this.opponent_qualiaSheaf[paramName] = value;
                    valueDisplay.textContent = value.toFixed(3);
                    slider.setAttribute('aria-valuetext', value.toFixed(3));
                });
            }
        });
    }

    setupTooltips() {
        if (!window.tippy) {
            logger.warn('Tippy.js not loaded; tooltips will not be initialized.');
            return;
        }
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
        tippy('#resetSimButton', { content: 'R1esets the game, AI states, and world models to their initial configurations. (R key)' });
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
                const normalizedWidth = clamp(softmaxScores[i] || 0, 0, 1) * 100;
                bar.style.width = `${normalizedWidth}%`;
            }
        }
    }

    drawSheafGraph() {
        if (!this.sheafGraphCtx || !this.mainAI_qualiaSheaf?.adjacencyMatrix) return;
        const { width, height } = this.sheafGraphCanvas;
        this.sheafGraphCtx.clearRect(0, 0, width, height);
        const sheaf = this.mainAI_qualiaSheaf;
        const adj = sheaf.adjacencyMatrix;
        const canvasRect = this.sheafGraphCanvas.getBoundingClientRect();
        const canvasWidth = canvasRect.width;
        const canvasHeight = canvasRect.height;
        sheaf.graph.edges.forEach(([u, v]) => {
            const uIdx = sheaf.graph.vertices.indexOf(u);
            const vIdx = sheaf.graph.vertices.indexOf(v);
            if (!sheafVertexPositions[uIdx] || !sheafVertexPositions[vIdx] || uIdx === -1 || vIdx === -1) return;
            const weight = adj[uIdx]?.[vIdx] || 0.1;
            this.sheafGraphCtx.strokeStyle = `rgba(68, 170, 255, ${clamp(weight, 0.1, 1.0)})`;
            this.sheafGraphCtx.lineWidth = clamp(weight * 2, 0.5, 3.0);
            const p1 = sheafVertexPositions[uIdx];
            const p2 = sheafVertexPositions[vIdx];
            const vertexEl = document.getElementById('vertex-0');
            const vWidth = vertexEl?.offsetWidth || 40;
            const vHeight = vertexEl?.offsetHeight || 24;
            this.sheafGraphCtx.beginPath();
            this.sheafGraphCtx.moveTo(p1.x * (canvasWidth - vWidth) + vWidth / 2, p1.y * (canvasHeight - vHeight) + vHeight / 2);
            this.sheafGraphCtx.lineTo(p2.x * (canvasWidth - vWidth) + vWidth / 2, p2.y * (canvasHeight - vHeight) + vHeight / 2);
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
        let d = `M ${padding},${(height - 2 * padding) * (1 - (clamp(data[0], yMin, yMax) - yMin) / (yMax - yMin)) + padding} `;
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
        if (!this.mainAI || !this.game) return;
        const stateValue = this.mainAI_worldModel.lastStateValue || this.mainAI.avgStateValue || 0;
        const predError = this.mainAI_worldModel.predictionError || 0;
        const epsilon = this.mainAI.epsilon || 0.1;
        const score = this.game.score.ai - this.game.score.player;

        this.chartEMA.stateValue = this.EMA_ALPHA * stateValue + (1 - this.EMA_ALPHA) * this.chartEMA.stateValue;
        this.chartEMA.predError = this.EMA_ALPHA * predError + (1 - this.EMA_ALPHA) * this.chartEMA.predError;
        this.chartEMA.epsilon = this.EMA_ALPHA * epsilon + (1 - this.EMA_ALPHA) * this.chartEMA.epsilon;
        this.chartEMA.score = this.EMA_ALPHA * score + (1 - this.EMA_ALPHA) * this.chartEMA.score;

        this.chartData.stateValue.push(this.chartEMA.stateValue);
        this.chartData.predError.push(this.chartEMA.predError);
        this.chartData.epsilon.push(this.chartEMA.epsilon);
        this.chartData.score.push(this.chartEMA.score);

        if (this.chartData.stateValue.length > this.MAX_CHART_POINTS) {
            for (const key in this.chartData) this.chartData[key].shift();
        }

        this.drawChart('qValueChart', this.chartData.stateValue, 'var(--primary-blue)', -2, 2);
        this.drawChart('predErrorChart', this.chartData.predError, 'var(--warn-orange)', 0, 10);
        this.drawChart('epsilonChart', this.chartData.epsilon, 'var(--info-green)', 0, 1);
        this.drawChart('scoreChart', this.chartData.score, 'var(--error-red)', -10, 10);
    }

    async gameLoop(timestamp, isManualStep = false) {
        if (!this.isRunning && !isManualStep) return;

        if (!this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready) {
            logger.warn(`OWM Readiness Check (Start of gameLoop): MainAI_OWM.ready = ${this.mainAI_worldModel?.ready}, OpponentAI_OWM.ready = ${this.opponent_worldModel?.ready}. Attempting re-initialization.`);
            document.getElementById('status').textContent = 'Error: Simulation not ready, re-initializing...';
            
            try {
                await Promise.all([
                    this.mainAI_qualiaSheaf.initialize(),
                    this.opponent_qualiaSheaf.initialize(),
                ]);
                await Promise.all([
                    this.mainAI_worldModel.initialize(),
                    this.opponent_worldModel.initialize(),
                ]);
                this.mainAI.reset();
                this.opponentAI.reset();
            } catch (e) {
                logger.error('Failed to re-initialize OWMs in gameLoop:', e);
                this.stop();
                document.getElementById('status').textContent = 'Error: Critical re-initialization failed. Simulation stopped.';
                return;
            }

            if (!this.mainAI_worldModel.ready || !this.opponent_worldModel.ready) {
                this.stop();
                document.getElementById('status').textContent = 'Error: Critical re-initialization failed. Simulation stopped.';
                return;
            }
            if (!isManualStep) requestAnimationFrame(this.boundGameLoop);
            return;
        }

        this.game.scene.updateMatrixWorld(true);
        try {
            const stepsPerFrame = isManualStep ? 1 : (this.isFastForward ? this.FAST_FORWARD_MULTIPLIER : 1);
            let mainDecision, opponentDecision;
            let corruptedCount = 0;
            const MAX_CORRUPTED = 5;

            for (let step = 0; step < stepsPerFrame; step++) {
                this.frameCount++;
                this.sheafStepCount++;

                const preGameState = this.game.getState() || {};

                [mainDecision, opponentDecision] = await Promise.all([
                    this.mainAI.makeDecision(preGameState).catch(err => {
                        logger.error('Error in mainAI.makeDecision', err.message, err.stack);
                        return { action: [0,0,0,1], chosenActionIndex: 3, corrupted: true, activations: [] };
                    }),
                    this.opponentAI.makeDecision(preGameState).catch(err => {
                        logger.error('Error in opponentAI.makeDecision', err.message, err.stack);
                        return { action: [0,0,0,1], chosenActionIndex: 3, corrupted: true, activations: [] };
                    })
                ]);

                if (mainDecision.corrupted || opponentDecision.corrupted) {
                    corruptedCount++;
                    logger.warn(`Corrupted decision detected (main: ${mainDecision.corrupted}, opp: ${opponentDecision.corrupted}). Consecutive: ${corruptedCount}/${MAX_CORRUPTED}`);
                    if (corruptedCount >= MAX_CORRUPTED) {
                        logger.warn('Too many consecutive corrupted decisions. Resetting AIs and OWMs.');
                        await this.resetAI();
                        corruptedCount = 0;
                    }
                    continue;
                } else {
                    corruptedCount = 0;
                }

                this.game.setAIAction(mainDecision.action);
                this.game.setPlayerAction(opponentDecision.action);
                const gameUpdateResult = this.game.update();

                if (!gameUpdateResult || !Number.isFinite(gameUpdateResult.aReward) || !Number.isFinite(gameUpdateResult.pReward)) {
                    logger.warn('Invalid game update result, resetting game and continuing.');
                    this.game.reset();
                    continue;
                }

                const postGameState = this.game.getState() || {};

                if (this.mainAI_worldModel.ready && this.opponent_worldModel.ready) {
                    await Promise.all([
                        this.mainAI.learn(preGameState, mainDecision.chosenActionIndex, gameUpdateResult.aReward, postGameState, gameUpdateResult.isDone).catch(err => {
                            logger.error('Error in mainAI.learn', err.message, err.stack);
                        }),
                        this.opponentAI.learn(preGameState, opponentDecision.chosenActionIndex, gameUpdateResult.pReward, postGameState, gameUpdateResult.isDone).catch(err => {
                            logger.error('Error in opponentAI.learn', err.message, err.stack);
                        })
                    ]);

                    if (this.mainAI_worldModel.actorLoss === 0 && this.mainAI_worldModel.criticLoss === 0 && this.mainAI_worldModel.predictionLoss === 0) {
                        this.consecutiveZeroLosses++;
                        if (this.consecutiveZeroLosses > this.MAX_ZERO_LOSSES) {
                            logger.warn('Detected stalled learning (zero losses). Reinitializing OWMs.');
                            await Promise.all([
                                this.mainAI_qualiaSheaf.initialize(),
                                this.opponent_qualiaSheaf.initialize(),
                                this.mainAI_worldModel.initialize(),
                                this.opponent_worldModel.initialize(),
                                this.mainAI.reset(),
                                this.opponentAI.reset()
                            ]);
                            this.consecutiveZeroLosses = 0;
                        }
                    } else {
                        this.consecutiveZeroLosses = 0;
                    }
                } else {
                    logger.warn('OWMs not ready for learning. Attempting re-initialization and skipping learning step.');
                    await Promise.all([
                        this.mainAI_qualiaSheaf.initialize(),
                        this.opponent_qualiaSheaf.initialize(),
                        this.mainAI_worldModel.initialize(),
                        this.opponent_worldModel.initialize(),
                        this.mainAI.reset(),
                        this.opponentAI.reset()
                    ]);
                    continue;
                }

                this.mainStrategicAI.observe(gameUpdateResult.aReward);
                this.opponentStrategicAI.observe(gameUpdateResult.pReward);

                if (this.sheafStepCount % this.sheafAdaptFrequency === 0) {
                    await this.mainAI_qualiaSheaf.adaptSheafTopology(this.sheafAdaptFrequency, this.sheafStepCount);
                    await this.opponent_qualiaSheaf.adaptSheafTopology(this.sheafAdaptFrequency, this.sheafStepCount);
                    logger.info(`Sheaf adaptation triggered at step ${this.sheafStepCount}.`);
                }

                if (this.frameCount % 50 === 0) {
                    this.mainStrategicAI.modulateParameters();
                    this.opponentStrategicAI.modulateParameters();
                }
            }
            document.getElementById('player-score').textContent = this.game.score.player;
            document.getElementById('ai-score').textContent = this.game.score.ai;
            // Increased visualization update frequency to every 5 frames
            if (this.frameCount % 5 === 0 || isManualStep) {
                this.mainViz.update(mainDecision?.activations || [], mainDecision?.chosenActionIndex || 0);
                this.opponentViz.update(opponentDecision?.activations || [], opponentDecision?.chosenActionIndex || 0);
                this.updateVisualization();
                this.updatePerformanceCharts();
            }
            this.game.render();
            if (isConceptVisualizationReady()) renderConceptVisualization();
        } catch (error) {
            logger.error('Error in game loop:', error.message, error.stack);
            this.stop();
            document.getElementById('status').textContent = `Error: Game loop stopped due to ${error.message || 'unknown error'}`;
            await this.resetAI();
        } finally {
            if (this.isRunning && !isManualStep) requestAnimationFrame(this.boundGameLoop);
        }
    }

    toggleGame() {
        if (!this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready) {
            logger.warn(`toggleGame rejected: MainAI_OWM.ready = ${this.mainAI_worldModel?.ready}, OpponentAI_OWM.ready = ${this.opponent_worldModel?.ready}`);
            document.getElementById('status').textContent = 'Error: AI not ready';
            return;
        }
        this.isRunning = !this.isRunning;
        const btn = document.getElementById('toggleSimButton');
        if (btn) {
            btn.textContent = this.isRunning ? '⏸️ Pause' : '🚀 Toggle Sim';
            btn.classList.toggle('active', this.isRunning);
        }
        if (this.isRunning) {
            this.game.resumeAudioContext();
            document.getElementById('status').textContent = this.isFastForward ? 'Fast Forward Active' : 'Conscious AI Active';
            this.gameLoop(null, false);
        } else {
            document.getElementById('status').textContent = 'Paused';
        }
    }

    toggleFastForward() {
        if (!this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready) {
            logger.warn('AIs not ready, cannot toggle fast forward.');
            document.getElementById('status').textContent = 'Error: AI not ready';
            return;
        }
        this.isFastForward = !this.isFastForward;
        const btn = document.getElementById('fastForwardButton');
        if (btn) {
            btn.classList.toggle('active', this.isFastForward);
            btn.innerHTML = this.isFastForward ? 'Normal Speed 🐢' : 'Fast ⏩';
        }
        if (this.isFastForward && !this.isRunning) this.toggleGame();
        else if (this.isRunning) {
            document.getElementById('status').textContent = this.isFastForward ? 'Fast Forward Active' : 'Conscious AI Active';
        }
    }

    stop() {
        this.isRunning = false;
        const toggleSimButton = document.getElementById('toggleSimButton');
        if (toggleSimButton) {
            toggleSimButton.textContent = '🚀 Toggle Sim';
            toggleSimButton.classList.remove('active');
        }
        if (this.isFastForward) {
            this.isFastForward = false;
            const ffBtn = document.getElementById('fastForwardButton');
            if (ffBtn) {
                ffBtn.classList.remove('active');
                ffBtn.innerHTML = 'Fast ⏩';
            }
        }
        document.getElementById('status').textContent = 'Paused';
    }

    async resetAI() {
        this.stop();
        logger.info('Resetting all game and AI states...');
        showLoading('game', 'Resetting Game...');
        showLoading('mainBrain', 'Resetting Main AI...');
        showLoading('opponentBrain', 'Resetting Opponent AI...');
        showLoading('metrics', 'Resetting Metrics...');
        try {
            this.game.reset();
            document.getElementById('player-score').textContent = 0;
            document.getElementById('ai-score').textContent = 0;
            
            this.mainAI_worldModel?.dispose();
            this.opponent_worldModel?.dispose();
            
            if (tf) tf.disposeVariables();

            const STATE_DIM = 13;
            const ACTION_DIM = 4;
            const Q_DIM = 7;

            // Re-create Sheaf instances first
            this.mainAI_qualiaSheaf = new EnhancedQualiaSheaf(null, STATE_DIM, Q_DIM, 0.1, 0.1, 0.05);
            this.opponent_qualiaSheaf = new EnhancedQualiaSheaf(null, STATE_DIM, Q_DIM, 0.1, 0.1, 0.05);

            // Re-create OWM instances, passing the new Sheafs
            this.mainAI_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], false, this.mainAI_qualiaSheaf);
            this.opponent_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], true, this.opponent_qualiaSheaf);

            // Set OWM reference in sheaves
            this.mainAI_qualiaSheaf.setOWM(this.mainAI_worldModel);
            this.opponent_qualiaSheaf.setOWM(this.opponent_worldModel);


            let retries = 3;
            while (retries > 0) {
                try {
                    await Promise.all([
                        this.mainAI_qualiaSheaf.initialize(),
                        this.opponent_qualiaSheaf.initialize(),
                    ]);
                    await Promise.all([
                        this.mainAI_worldModel.initialize(),
                        this.opponent_worldModel.initialize()
                    ]);
                    if (this.mainAI_worldModel.ready && this.opponent_worldModel.ready) break;
                    throw new Error('OWM not ready after initialization');
                } catch (e) {
                    retries--;
                    logger.warn(`OWM reset initialization failed, retries left: ${retries}`, e);
                    if (retries === 0) throw e;
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            }
            cleanupLive2D();
            cleanupConceptVisualization();
            await Promise.all([
                initLive2D(this.clock, this.mainAI_qualiaSheaf),
                initConceptVisualization(this.clock, this.mainAI_qualiaSheaf)
            ]);

            // Pass the newly initialized OWMs to the AI agents
            this.mainAI = new LearningAIAgent(STATE_DIM, ACTION_DIM, this.game, 3, this.mainAI_worldModel);
            this.opponentAI = new LearningAIAgent(STATE_DIM, ACTION_DIM, this.game, 3, this.opponent_worldModel);
            
            this.mainStrategicAI = new StrategicAI(this.mainAI);
            this.opponentStrategicAI = new StrategicAI(this.opponentAI);

            this.mainViz.worldModel = this.mainAI_worldModel;
            this.opponentViz.worldModel = this.opponent_worldModel;

            this.frameCount = 0;
            this.sheafStepCount = 0;
            this.consecutiveZeroLosses = 0;
            this.chartData = { stateValue: [], predError: [], epsilon: [], score: [] };
            this.chartEMA = { stateValue: 0, predError: 0, epsilon: 0, score: 0 };
            this.updatePerformanceCharts();
            this.updateVisualization();
            this.game.render();
            document.getElementById('status').textContent = 'Reset Complete.';
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

    async tuneParameters() {
        if (!this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready) {
            document.getElementById('status').textContent = 'Error: AI not ready';
            return;
        }
        await Promise.all([
            this.mainAI_qualiaSheaf.tuneParameters(),
            this.opponent_qualiaSheaf.tuneParameters()
        ]);
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
        document.getElementById('status').textContent = 'Parameters Tuned';
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

async function bootstrapApp() {
    if (tf) {
        try {
            await tf.setBackend('webgl');
            logger.info('TensorFlow.js WebGL backend initialized successfully at bootstrap.');
            await tf.ready();
            if (tf.getBackend() !== 'webgl') {
                logger.warn('WebGL backend not active, falling back to CPU');
                await tf.setBackend('cpu');
            }
        } catch (e) {
            logger.error('Failed to set TensorFlow.js WebGL backend during bootstrap, falling back to CPU:', e);
            await tf.setBackend('cpu');
        }
    } else {
        logger.error('TensorFlow.js not loaded globally. Please ensure tf.min.js is loaded as a global script before this module.');
    }

    try {
        const app = new MainApp();
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
        const graphPanel = document.getElementById('sheafGraph');
        if (graphPanel) new ResizeObserver(positionVertices).observe(graphPanel);
        window.addEventListener('load', positionVertices);
        window.addEventListener('resize', positionVertices);
        logger.info('UI initialized — ready. Awaiting user interaction.');
    } catch (e) {
        document.getElementById('status').textContent = `Initialization Error: ${e.message}`;
        logger.error('Initialization failed', e.message);
    }
}

window.addEventListener('load', bootstrapApp);

window.addEventListener('beforeunload', () => {
    cleanupLive2D();
    cleanupConceptVisualization();
    if (tf) {
        tf.disposeVariables();
        logger.info('TensorFlow.js resources disposed during unload.');
    }
});
