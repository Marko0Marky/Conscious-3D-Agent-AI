// --- START OF FILE main.js ---
// --- MERGED VERSION: Prioritizing robust features from first file, incorporating UI/logic from second ---
import { logger, showLoading, hideLoading, clamp, isFiniteVector, norm2, vecZeros, vecAdd } from './utils.js';
import { OntologicalWorldModel } from './owm.js';
import { LearningAI, StrategicAI } from './ai-agents.js';
import { ThreeDeeGame } from './three-dee-game.js';
import { NeuralNetworkVisualizer } from './nn-visualizer.js';
import { initLive2D, updateLive2DEmotions, cleanupLive2D, isLive2DReady, updateLive2D } from './viz-live2d.js';
import { initConceptVisualization, cleanupConceptVisualization, renderConceptVisualization, animateConceptNodes, isConceptVisualizationReady, updateAgentSimulationVisuals } from './viz-concepts.js';

// Timeout helper from first file
const withTimeout = (promise, ms, errorMsg) => Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(errorMsg)), ms))
]);

// Global for positioning vertex divs
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
        this.clock = new THREE.Clock(); // From first
        this.chartData = { stateValue: [], predError: [], epsilon: [], score: [] }; // From first
        this.MAX_CHART_POINTS = 100;
        this.applyCanvasDPR(this.sheafGraphCanvas, this.sheafGraphCtx);
        new ResizeObserver(() => this.applyCanvasDPR(this.sheafGraphCanvas, this.sheafGraphCtx)).observe(document.getElementById('sheafGraph'));
        // Additional resize for game from second
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
        this.lastPhi = 0; // From first
        this.boundGameLoop = this.gameLoop.bind(this);
        this.bindEvents();
        this.setupTooltips(); // From second
        this.setupQualiaAttentionPanel(); // From second
        document.getElementById('status').textContent = 'Initializing... Please wait.';
        this.setupGameAndAIs();
    }
    
    applyCanvasDPR(canvas, ctx) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement ? canvas.parentElement.getBoundingClientRect() : canvas.getBoundingClientRect(); // Merged
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
        showLoading('metrics', 'Initializing Qualia Sheaf...'); // From first

        try {
            this.game = new ThreeDeeGame(this.gameCanvas);
            logger.info('ThreeDeeGame initialized.');
            const STATE_DIM = 13;
            const ACTION_DIM = 4;
            const Q_DIM = 7; // From first

            this.mainAI_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], true);
            this.opponent_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], false);

            logger.info("Calling Promise.all for OWM initialization...");
            await withTimeout(Promise.all([
                this.mainAI_worldModel.initialize(),
                this.opponent_worldModel.initialize()
            ]), 15000, 'OWM initialization timed out.'); // From first
            
            logger.info("Core OWMs initialized successfully.");

            // Viz inits from first
            await withTimeout(Promise.all([
                initLive2D(this.clock),
                initConceptVisualization(this.clock, this.mainAI_worldModel.qualiaSheaf)
            ]), 10000, 'Visualization initialization timed out.');

            logger.info("Visualizations initialized successfully.");

            this.mainAI = new LearningAI(this.mainAI_worldModel, this.game, true);
            this.opponentAI = new LearningAI(this.opponent_worldModel, this.game, false);
            this.mainStrategicAI = new StrategicAI(this.mainAI);
            this.opponentStrategicAI = new StrategicAI(this.opponentAI);
            this.mainViz = new NeuralNetworkVisualizer('nn-visualization-container', this.mainAI_worldModel, 'main');
            this.opponentViz = new NeuralNetworkVisualizer('nn-visualization-container-opponent', this.opponent_worldModel, 'opponent');

            this.game.render();
            this.updateVisualization();
            logger.info('Game and AIs initialized successfully.');
            document.getElementById('status').textContent = 'Ready. Click "Toggle Sim" to start.';

        } catch (e) {
            logger.error('Failed to set up game and AIs:', e);
            document.getElementById('status').textContent = `Initialization Failed: ${e.message}`;
            if (e.message.includes('OWM')) {
                document.getElementById('status').textContent = 'Initialization Failed: Critical failure: OWMs not initialized';
            }
        } finally {
            hideLoading('game');
            hideLoading('mainBrain');
            hideLoading('opponentBrain');
            hideLoading('metrics');
        }
    }

    mapEmotionsForAvatar(worldModel) { // From first
        const qualia = worldModel.qualiaSheaf;
        const joy = clamp(qualia.gestaltUnity * qualia.stability, 0, 1);
        const fear = clamp(1 - qualia.stability, 0, 1);
        const curiosity = clamp(worldModel.predictionError / 5.0 + qualia.h1Dimension / 3.0, 0, 1);
        const frustration = clamp(qualia.inconsistency + (worldModel.actorLoss || 0), 0, 1);
        const calm = clamp(qualia.stability * (1 - worldModel.freeEnergy / 2.0), 0, 1);
        const surprise = clamp(Math.abs(qualia.phi - this.lastPhi) / 0.1, 0, 1);
        this.lastPhi = qualia.phi;
        const emotionVec = [joy, fear, curiosity, frustration, calm, surprise];
        return tf.tensor([emotionVec]);
    }

    // Merged updateQualiaBars from first, with logic from second's updateVisualization
    updateQualiaBars(qualiaSheaf) {
        if (!qualiaSheaf) return;

        const entityNames = qualiaSheaf.entityNames.map(name => name.replace('_', '-'));
        const qDim = qualiaSheaf.qDim;
        const numVertices = qualiaSheaf.graph.vertices.length;
        if (numVertices === 0) return;

        const aggregateStalk = new Float32Array(qDim).fill(0);
        
        // Sum up all stalk vectors
        for (const stalk of qualiaSheaf.stalks.values()) {
            if (isFiniteVector(stalk)) {
                for (let i = 0; i < qDim; i++) {
                    aggregateStalk[i] += stalk[i];
                }
            }
        }

        // Update each bar and value
        for (let i = 0; i < qDim; i++) {
            const avgValue = aggregateStalk[i] / numVertices;
            const intensity = Math.abs(avgValue); // Bar width represents magnitude/intensity
            const clampedValue = clamp(avgValue, 0, 1); // From second for display

            const fillElement = document.getElementById(`qualia-${entityNames[i]}-fill`);
            const valueElement = document.getElementById(`${entityNames[i]}-value`);

            if (fillElement) {
                fillElement.style.width = `${clampedValue * 100}%`; // Use clamped for consistency
            }
            if (valueElement) {
                valueElement.textContent = clampedValue.toFixed(3); // Fixed decimals from second
            }
        }
    }

    // Merged updateVisualization
    updateVisualization() {
        const qualia = this.mainAI_worldModel?.qualiaSheaf;
        if (!qualia) return;

        const model = this.mainAI_worldModel;
        const mainAI = this.mainAI;

        if (!model || !mainAI) return;

        // Qualia bars update from merged function
        this.updateQualiaBars(qualia);

        // Avg qualia logic from second for additional displays if needed
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

        // Update specific value elements from second
        document.getElementById('being-value').textContent = qualiaValues[0].toFixed(3);
        document.getElementById('intent-value').textContent = qualiaValues[1].toFixed(3);
        document.getElementById('existence-value').textContent = qualiaValues[2].toFixed(3);
        document.getElementById('emergence-value').textContent = qualiaValues[3].toFixed(3);
        document.getElementById('gestalt-value').textContent = qualiaValues[4].toFixed(3);
        document.getElementById('context-value').textContent = qualiaValues[5].toFixed(3);
        document.getElementById('rel-emergence-value').textContent = qualiaValues[6].toFixed(3);

        requestAnimationFrame(() => {
            document.getElementById('phi-display').textContent = `Φ: ${clamp(qualia.phi, 0, 5).toFixed(5)}`;
            document.getElementById('free-energy').textContent = (model.freeEnergy || 0).toFixed(5);
            document.getElementById('prediction-error').textContent = (model.predictionError || 0).toFixed(5);
            document.getElementById('h1-dimension').textContent = clamp(qualia.h1Dimension, 0, qualia.graph.edges.length).toFixed(2);
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
            this.updateQualiaAttentionVisuals(); // From second
            
            if (isLive2DReady()) {
                const emotionTensor = this.mapEmotionsForAvatar(this.mainAI_worldModel);
                updateLive2DEmotions(emotionTensor, this.game.score.ai > this.game.score.player ? 'nod' : 'idle');
                updateLive2D(this.clock.getDelta());
                tf.dispose(emotionTensor);
            }
            if (isConceptVisualizationReady()) {
                const gameState = this.game.getState();
                const augmentedGameState = { ...gameState, dist: Math.sqrt(Math.pow(gameState.aiTarget.x - gameState.ai.x, 2) + Math.pow(gameState.aiTarget.z - gameState.ai.z, 2)) };
                updateAgentSimulationVisuals(null, augmentedGameState, qualia.phi);
                animateConceptNodes(this.clock.getDelta());
                renderConceptVisualization();
            }
        });
    }
    
    bindEvents() {
        document.getElementById('toggleSimButton').onclick = () => this.toggleGame();
        document.getElementById('resetSimButton').onclick = () => this.resetAI();
        document.getElementById('tuneButton').onclick = () => this.tuneParameters();
        document.getElementById('pauseButton').onclick = () => this.stop();
        document.getElementById('stepButton').onclick = () => this.gameLoop(null, true);
        document.getElementById('fastForwardButton').onclick = () => this.toggleFastForward();
        
        // From second
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

    // From second
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
    
    // From second
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

    // From second
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

    drawSheafGraph() { // Merged from both
        if (!this.sheafGraphCtx || !this.mainAI_worldModel?.qualiaSheaf?.adjacencyMatrix) return;
        const { width, height } = this.sheafGraphCanvas;
        this.sheafGraphCtx.clearRect(0, 0, width, height);
        const sheaf = this.mainAI_worldModel.qualiaSheaf;
        const adj = sheaf.adjacencyMatrix;
        const canvasWidth = this.sheafGraphCanvas.getBoundingClientRect().width;
        const canvasHeight = this.sheafGraphCanvas.getBoundingClientRect().height;
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

    // Merged drawChart
    drawChart(svgId, data, color, yMin, yMax) {
        const svg = document.getElementById(svgId);
        if (!svg) return;
        svg.innerHTML = '';
        const width = svg.clientWidth;
        const height = svg.clientHeight;
        const padding = 5; // From first, but 10 in second - use 10 for consistency
        if (data.length < 2) return;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        let d = `M ${padding},${(height - 2*padding) * (1 - (clamp(data[0], yMin, yMax) - yMin) / (yMax - yMin)) + padding} `;
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
        path.setAttribute('stroke-linejoin', 'round'); // From second
        svg.appendChild(path);
    }

    // Merged updatePerformanceCharts - use stateValue from first, but avgQValue if available
    updatePerformanceCharts() {
        if (this.mainAI && this.game) {
            this.chartData.stateValue.push(this.mainAI_worldModel.lastStateValue || this.mainAI.avgQValue || 0);
            this.chartData.predError.push(this.mainAI_worldModel.predictionError || 0);
            this.chartData.epsilon.push(this.mainAI.epsilon || 0.1);
            this.chartData.score.push(this.game.score.ai - this.game.score.player);
        }
        for (const key in this.chartData) {
            if (this.chartData[key].length > this.MAX_CHART_POINTS) this.chartData[key].shift();
        }
        this.drawChart('qValueChart', this.chartData.stateValue, 'var(--primary-blue)', -2, 2);
        this.drawChart('predErrorChart', this.chartData.predError, 'var(--warn-orange)', 0, 10); // 10 from first
        this.drawChart('epsilonChart', this.chartData.epsilon, 'var(--info-green)', 0, 1);
        this.drawChart('scoreChart', this.chartData.score, 'var(--error-red)', -10, 10);
    }

    // Merged gameLoop - prioritize first's logic with second's improvements
    async gameLoop(timestamp, isManualStep = false) {
        if (!this.isRunning && !isManualStep) return;
        if (!this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready || !this.game) {
            if (!isManualStep) requestAnimationFrame(this.boundGameLoop);
            return;
        }
        this.game.scene.updateMatrixWorld(true); // From second
        try {
            const stepsPerFrame = isManualStep ? 1 : (this.isFastForward ? this.FAST_FORWARD_MULTIPLIER : 1);
            let mainDecision, opponentDecision;
            for (let step = 0; step < stepsPerFrame; step++) {
                this.frameCount++;
                const preGameState = this.game.getState();
                [mainDecision, opponentDecision] = await Promise.all([
                    this.mainAI.makeDecision(preGameState),
                    this.opponentAI.makeDecision(preGameState)
                ]);
                if (mainDecision.corrupted) this.mainAI.reset(); // From second
                if (opponentDecision.corrupted) this.opponentAI.reset();
                this.game.setAIAction(mainDecision.action);
                this.game.setPlayerAction(opponentDecision.action);
                const gameUpdateResult = this.game.update();
                const postGameState = this.game.getState();
                await Promise.all([
                    this.mainAI.learn(preGameState, mainDecision.chosenActionIndex, gameUpdateResult.aReward, postGameState, gameUpdateResult.isDone), // Full args from first
                    this.opponentAI.learn(preGameState, opponentDecision.chosenActionIndex, gameUpdateResult.pReward, postGameState, gameUpdateResult.isDone)
                ]);
                this.mainStrategicAI.observe(gameUpdateResult.aReward); // From second
                this.opponentStrategicAI.observe(gameUpdateResult.pReward);
                if (this.frameCount % 50 === 0) {
                    this.mainStrategicAI.modulateParameters();
                    this.opponentStrategicAI.modulateParameters();
                    await this.mainAI_worldModel.qualiaSheaf.tuneParameters();
                }
            }
            document.getElementById('player-score').textContent = this.game.score.player;
            document.getElementById('ai-score').textContent = this.game.score.ai;
            if (this.frameCount % 5 === 0 || isManualStep) {
                this.mainViz.update(mainDecision.activations, mainDecision.chosenActionIndex);
                this.opponentViz.update(opponentDecision.activations, opponentDecision.chosenActionIndex);
                this.updateVisualization();
            }
            if (this.frameCount % 20 === 0 || isManualStep) this.updatePerformanceCharts();
            this.game.render();
            if (isConceptVisualizationReady()) renderConceptVisualization();
        } catch (error) {
            logger.error("Error in game loop, stopping simulation:", error);
            this.stop();
        } finally {
            if (this.isRunning && !isManualStep) requestAnimationFrame(this.boundGameLoop);
        }
    }

    // Merged toggleGame
    toggleGame() {
        if (!this.mainAI_worldModel?.ready) return;
        this.isRunning = !this.isRunning;
        const btn = document.getElementById('toggleSimButton');
        btn.textContent = this.isRunning ? '⏸️ Pause' : '🚀 Toggle Sim';
        btn.classList.toggle('active', this.isRunning);
        if (this.isRunning) {
             this.game.resumeAudioContext();
             document.getElementById('status').textContent = this.isFastForward ? 'Fast Forward Active' : 'Conscious AI Active'; // From second
             this.gameLoop(null, false);
        } else {
            document.getElementById('status').textContent = 'Paused';
        }
    }

    // Merged toggleFastForward
    toggleFastForward() {
        if (!this.mainAI_worldModel?.ready) {
            logger.warn('AIs not ready, cannot toggle fast forward.');
            return;
        }
        this.isFastForward = !this.isFastForward;
        const btn = document.getElementById('fastForwardButton');
        btn.classList.toggle('active', this.isFastForward);
        btn.innerHTML = this.isFastForward ? 'Normal Speed 🐢' : 'Fast ⏩';
        if (this.isFastForward && !this.isRunning) this.toggleGame();
        else if(this.isRunning) {
           document.getElementById('status').textContent = this.isFastForward ? 'Fast Forward Active' : 'Conscious AI Active';
        }
    }

    stop() { // Merged
        this.isRunning = false;
        document.getElementById('toggleSimButton').textContent = '🚀 Toggle Sim';
        document.getElementById('toggleSimButton').classList.remove('active');
        if (this.isFastForward) {
            this.isFastForward = false;
            const ffBtn = document.getElementById('fastForwardButton');
            ffBtn.classList.remove('active');
            ffBtn.innerHTML = 'Fast ⏩';
        }
        document.getElementById('status').textContent = 'Paused';
    }

    // Merged resetAI - use first's with second's improvements
    async resetAI() {
        this.stop();
        logger.info('Resetting all game and AI states...');
        showLoading('game', 'Resetting Game...');
        showLoading('mainBrain', 'Resetting Main AI...'); // From second
        showLoading('opponentBrain', 'Resetting Opponent AI...');
        showLoading('metrics', 'Resetting Metrics...');
        this.isFastForward = false;
        const ffBtn = document.getElementById('fastForwardButton');
        ffBtn.classList.remove('active');
        ffBtn.innerHTML = 'Fast ⏩';
        for (const key in this.chartData) this.chartData[key] = []; // From second
        try {
            this.game.reset(); // From first
            document.getElementById('player-score').textContent = 0;
            document.getElementById('ai-score').textContent = 0;
            this.mainAI_worldModel?.dispose();
            this.opponent_worldModel?.dispose();
            const STATE_DIM = 13;
            const ACTION_DIM = 4;
            const Q_DIM = 7;
            this.mainAI_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], true);
            this.opponent_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], false);
            cleanupLive2D();
            cleanupConceptVisualization();
            await withTimeout(Promise.all([
                this.mainAI_worldModel.initialize(),
                this.opponent_worldModel.initialize()
            ]), 15000, 'OWM reset timeout');
            await withTimeout(Promise.all([
                initLive2D(this.clock),
                initConceptVisualization(this.clock, this.mainAI_worldModel.qualiaSheaf)
            ]), 10000, 'Visualization reset timeout');
            this.mainAI = new LearningAI(this.mainAI_worldModel, this.game, true);
            this.opponentAI = new LearningAI(this.opponent_worldModel, this.game, false);
            this.mainStrategicAI = new StrategicAI(this.mainAI);
            this.opponentStrategicAI = new StrategicAI(this.opponentAI);
            this.mainViz.worldModel = this.mainAI_worldModel;
            this.opponentViz.worldModel = this.opponent_worldModel;
            this.frameCount = 0;
            this.chartData = { stateValue: [], predError: [], epsilon: [], score: [] };
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

    // Merged tuneParameters
    async tuneParameters() {
        if (!this.mainAI_worldModel?.ready) return;
        await this.mainAI_worldModel.qualiaSheaf.tuneParameters();
        await this.opponent_worldModel.qualiaSheaf.tuneParameters();
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

    // From second
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
    window.addEventListener('resize', positionVertices); // From second
    logger.info('UI initialized — ready. Awaiting user interaction.');
} catch (e) {
    document.getElementById('status').textContent = `Initialization Error: ${e.message}`;
    logger.error('Initialization failed', e.message);
}

window.addEventListener('beforeunload', () => {
    cleanupLive2D();
    cleanupConceptVisualization();
});
// --- END OF FILE main.js ---
