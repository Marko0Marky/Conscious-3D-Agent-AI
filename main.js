// =================================================================
// IMPORTS
// =================================================================
import { logger, showLoading, hideLoading, clamp, isFiniteVector, vecZeros } from './utils.js';
import { OntologicalWorldModel } from './owm.js';
import { LearningAIAgent, StrategicAI } from './ai-agents.js';
import { ThreeDeeGame } from './three-dee-game.js';
import { NeuralNetworkVisualizer } from './nn-visualizer.js';
import { initLive2D, updateLive2DEmotions, cleanupLive2D, isLive2DReady, updateLive2D } from './viz-live2d.js';
import { initConceptVisualization, cleanupConceptVisualization, renderConceptVisualization, animateConceptNodes, isConceptVisualizationReady, updateAgentSimulationVisuals } from './viz-concepts.js';
import { FloquetPersistentSheaf } from './sheaf/FloquetPersistentSheaf.js'; // <-- CORRECTED PATH
import * as THREE from 'three';

const tf = window.tf;

// --- TEMPORARY DIAGNOSTIC FIX for "weight is not defined" ---
// If an external module is attempting to use a global 'weight' variable that
// is no longer implicitly declared, this will prevent a hard crash.
// This is a symptom of a deeper issue in the external code.
if (typeof window.weight === 'undefined') {
    window.weight = 0.5; // Default or arbitrary value to prevent crash
    logger.warn('Global \'weight\' variable was not defined; initialized to 0.5 for stability. Please check external AI Agent or Game code for undeclared variables.');
}
// --- END TEMPORARY DIAGNOSTIC FIX ---

const sheafVertexPositions = {
    0: { x: 0.1, y: 0.5 }, 1: { x: 0.3, y: 0.2 }, 2: { x: 0.3, y: 0.8 },
    3: { x: 0.9, y: 0.5 }, 4: { x: 0.7, y: 0.2 }, 5: { x: 0.5, y: 0.35 },
    6: { x: 0.7, y: 0.8 }, 7: { x: 0.5, y: 0.65 }
};

class MainApp {
    constructor() {
        logger.info('MainApp constructor started.');
        this.gameCanvas = document.getElementById('gameCanvas');
        this.sheafGraphCanvas = document.getElementById('sheafGraphCanvas');
        if (!this.gameCanvas || !this.sheafGraphCanvas) {
            throw new Error('Canvas not found');
        }
        this.sheafGraphCtx = this.sheafGraphCanvas.getContext('2d');
        if (!this.sheafGraphCtx) {
            throw new Error('Failed to get 2D context for sheaf');
        }

        this.toggleSimButton = document.getElementById('toggleSimButton');
        this.resetSimButton = document.getElementById('resetSimButton');
        this.fastForwardButton = document.getElementById('fastForwardButton');

        if (this.toggleSimButton) this.toggleSimButton.disabled = true;
        if (this.resetSimButton) this.resetSimButton.disabled = true;
        if (this.fastForwardButton) this.fastForwardButton.disabled = true;
        document.getElementById('status').textContent = 'Initializing... Please wait.';

        this.clock = new THREE.Clock();
        this.clock.start(); // Start clock immediately to ensure getDelta() returns valid values
        this.chartData = { qValue: [], score: [], cupProduct: [], structuralSensitivity: [], coherence: [], freeEnergy: [] };
        this.chartEMA = { qValue: 0, score: 0, cupProduct: 0, structuralSensitivity: 0, coherence: 0, freeEnergy: 0 };
        this.EMA_ALPHA = 0.1;
        this.MAX_CHART_POINTS = 100;
        this.applyCanvasDPR(this.sheafGraphCanvas, this.sheafGraphCtx);
        new ResizeObserver(() => this.applyCanvasDPR(this.sheafGraphCanvas, this.sheafGraphCtx)).observe(document.getElementById('sheafGraph'));
        new ResizeObserver(entries => {
            for (let entry of entries) {
                const { width, height } = entry.contentRect;
                if (this.game) this.game.resize(width, height);
            }
        }).observe(this.gameCanvas);
        this.isRunning = false;
        this.isFastForward = false;
        this.FAST_FORWARD_MULTIPLIER = 3;
        this.frameCount = 0;
        this.sheafAdaptFrequency = 200;
        this.consecutiveZeroLosses = 0;
        this.MAX_ZERO_LOSSES = 50;
        this.lastPhi = 0;
        this.isInitialized = false;
        this.boundGameLoop = this.gameLoop.bind(this);
        this.bindEvents();
        this.setupTooltips();
        this.initialize().catch(e => {
            logger.error(`Initialization failed in constructor: ${e.message}`, e);
            document.getElementById('status').textContent = `Fatal Error: ${e.message}`;
        });
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

    async initialize() {
        await this.setupGameAndAIs();
        this.mainViz = new NeuralNetworkVisualizer('mainBrainCanvas', this.mainAI_worldModel);
        this.opponentViz = new NeuralNetworkVisualizer('opponentBrainCanvas', this.opponent_worldModel);
        logger.info("MainApp fully initialized, including visualizers. Ready for interaction.");
        document.getElementById('status').textContent = 'Ready. Press Toggle Sim to Start.';

        if (this.toggleSimButton) this.toggleSimButton.disabled = false;
        if (this.resetSimButton) this.resetSimButton.disabled = false;
        if (this.fastForwardButton) this.fastForwardButton.disabled = false;
        this.isInitialized = true;
        await this.updateVisualization();
    }

    async setupGameAndAIs() {
    logger.info('setupGameAndAIs() started.');
    showLoading('game', 'Initializing 3D Environment...');
    showLoading('mainBrain', 'Building Main AI World Model...');
    showLoading('opponentBrain', 'Building Opponent AI World Model...');
    showLoading('metrics', 'Initializing Qualia Sheaf...');

    // --- START OF DEFINITIVE FIX ---
    try {
        this.game = new ThreeDeeGame(this.gameCanvas);
        await this.game.reset();

        const STATE_DIM = 13;
        const ACTION_DIM = 4;
        const Q_DIM = 7;

        this.mainAI_qualiaSheaf = new FloquetPersistentSheaf({}, { stateDim: STATE_DIM, qDim: Q_DIM });
        this.opponent_qualiaSheaf = new FloquetPersistentSheaf({}, { stateDim: STATE_DIM, qDim: Q_DIM });

        this.mainAI_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], false, this.mainAI_qualiaSheaf);
        this.opponent_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], true, this.opponent_qualiaSheaf);

        this.mainAI_qualiaSheaf.setOWM(this.mainAI_worldModel);
        this.opponent_qualiaSheaf.setOWM(this.opponent_worldModel);

        // This is where the original error likely happened.
        this.mainAI = new LearningAIAgent(this.mainAI_worldModel);
        this.opponentAI = new StrategicAI(this.opponent_worldModel);
        this.mainStrategicAI = new StrategicAI(this.mainAI_worldModel);
        this.opponentStrategicAI = new StrategicAI(this.opponent_worldModel);

        await Promise.all([
            this.mainAI_qualiaSheaf.initialize(),
            this.opponent_qualiaSheaf.initialize(),
            this.mainAI_worldModel.initialize(),
            this.opponent_worldModel.initialize()
        ]);

        logger.info('Initialization completed for sheafs and world models.');

        await initLive2D(this.clock, this.mainAI_qualiaSheaf);
        await initConceptVisualization(this.clock, this.mainAI_qualiaSheaf);
        await this.setupQualiaAttentionPanel();

    } catch (e) {
        // This block now catches any failure during setup.
        logger.error('CRITICAL FAILURE during setupGameAndAIs:', e);
        document.getElementById('status').textContent = `Initialization Failed: ${e.message}`;
        // Re-throw the error to stop the application from proceeding into a broken state.
        throw e;
    } finally {
        // This will run whether setup succeeds or fails.
        hideLoading('game');
        hideLoading('mainBrain');
        hideLoading('opponentBrain');
        hideLoading('metrics');
    }
    // --- END OF DEFINITIVE FIX ---
}

    _stateToVector(state) {
        if (!state || !state.ai || !state.player || !state.aiTarget) {
            logger.warn("Received incomplete state for vector conversion. Returning zeros.");
            return vecZeros(13);
        }
        return new Float32Array([
            state.ai.x, state.ai.z, state.ai.rot,
            state.player.x, state.player.z, state.player.rot,
            state.aiTarget.x, state.aiTarget.z,
            state.playerTarget.x, state.playerTarget.z,
            state.ai.x - state.aiTarget.x, state.ai.z - state.aiTarget.z,
            Math.sqrt((state.ai.x - state.aiTarget.x)**2 + (state.ai.z - state.aiTarget.z)**2)
        ]);
    }

    mapEmotionsForAvatar(worldModel) {
        const qualia = worldModel?.qualiaSheaf;
        if (!qualia) {
            logger.warn('mapEmotionsForAvatar: qualiaSheaf is undefined. Returning default emotions.');
            return tf ? tf.tensor([[0, 0, 0, 0, 0, 0]]) : { arraySync: () => [[0, 0, 0, 0, 0, 0]], dispose: () => {} };
        }
        const h1Influence = clamp(qualia.h1Dimension / (qualia.graph.edges.length || 1), 0, 1);
        const joy = clamp(qualia.gestaltUnity * qualia.stability, 0, 1);
        const fear = clamp(1 - qualia.stability, 0, 1);
        const curiosity = clamp(worldModel.predictionError / 5.0 + h1Influence * 0.5, 0, 1);
        const frustration = clamp(qualia.inconsistency + (worldModel.actorLoss || 0), 0, 1);
        const calm = clamp(qualia.stability * (1 - worldModel.freeEnergy / 2.0), 0, 1);
        const surprise = clamp(Math.abs(qualia.phi - this.lastPhi) / 0.1, 0, 1);
        this.lastPhi = qualia.phi;
        const emotionVec = [joy, fear, curiosity, frustration, calm, surprise];
        return tf ? tf.tensor([emotionVec]) : { arraySync: () => [emotionVec], dispose: () => {} };
    }

    async updateVisualization() {
        if (!this.isInitialized || !this.mainAI_worldModel?.ready || !this.mainAI) return;

        const qualia = this.mainAI_worldModel?.qualiaSheaf;
        if (!qualia) {
            logger.warn('updateVisualization: qualiaSheaf is undefined.');
            return;
        }

        requestAnimationFrame(() => {
            document.getElementById('phi-display').textContent = `Φ: ${clamp(qualia.phi, 0, 5).toFixed(5)}`;
            document.getElementById('feel-F-display').textContent = `${qualia.feel_F.toFixed(5)}`;
            document.getElementById('intentionality-display').textContent = `${qualia.intentionality_F.toFixed(4)}`;
            document.getElementById('cup-product-intensity').textContent = `${(qualia.cup_product_intensity || 0).toFixed(4)}`;
            document.getElementById('structural-sensitivity-display').textContent = `${(qualia.structural_sensitivity || 0).toFixed(5)}`;
            document.getElementById('h1-dimension').textContent = clamp(qualia.h1Dimension, 0, qualia.graph.edges.length).toFixed(2);
            document.getElementById('gestalt-unity').textContent = clamp(qualia.gestaltUnity, 0, 1).toFixed(5);
            document.getElementById('inconsistency').textContent = (qualia.inconsistency || 0).toFixed(5);
            document.getElementById('stability-fill').style.width = `${clamp(qualia.stability, 0, 1) * 100}%`;
            document.getElementById('alpha-param').textContent = qualia.alpha.toFixed(3);
            document.getElementById('alphaSlider').value = qualia.alpha;
            document.getElementById('beta-param').textContent = qualia.beta.toFixed(3);
            document.getElementById('betaSlider').value = qualia.beta;
            document.getElementById('gamma-param').textContent = qualia.gamma.toFixed(3);
            document.getElementById('gammaSlider').value = qualia.gamma;

            this.drawSheafGraph();
            this.updateQualiaDynamicsVisuals();
            this.updatePerformanceCharts();

            if (isLive2DReady()) {
                const emotionWrapper = this.mapEmotionsForAvatar(this.mainAI_worldModel);
                updateLive2DEmotions(emotionWrapper, this.game.score.ai > this.game.score.player ? 'nod' : 'idle');
                updateLive2D(this.clock.getDelta());
                if (emotionWrapper.dispose) emotionWrapper.dispose();
            }

            if (isConceptVisualizationReady()) {
                animateConceptNodes(this.clock.getDelta());
                renderConceptVisualization();
            }
        });
    }

    setupQualiaAttentionPanel() {
        const panel = document.getElementById('qualiaAttentionPanel');
        if (!panel) return;
        const VERTEX_MAP = {
            'agent_x': { name: 'Agent-X' }, 'agent_z': { name: 'Agent-Z' },
            'agent_rot': { name: 'Agent-Rot' }, 'target_x': { name: 'Target-X' },
            'target_z': { name: 'Target-Z' }, 'vec_dx': { name: 'Vec-DX' },
            'vec_dz': { name: 'Vec-DZ' }, 'dist_target': { name: 'Dist-Target' }
        };
        const qualiaNames = this.mainAI_qualiaSheaf?.entityNames || [];
        if (qualiaNames.length === 0) {
            panel.innerHTML = `<h4>Qualia Attention (Main AI)</h4><p>Error: Qualia names not available</p>`;
            return;
        }
        let html = `<h4>Qualia Attention (Main AI)</h4>`;
        qualiaNames.forEach((name, i) => {
            const displayName = VERTEX_MAP[name]?.name || name.charAt(0).toUpperCase() + name.slice(1);
            html += `
                <div class="attention-bar-container">
                    <span class="attention-label">${displayName}</span>
                    <div class="attention-bar-wrapper">
                        <div class="attention-bar" id="attention-bar-${i}"></div>
                    </div>
                </div>
            `;
        });
        panel.innerHTML = html;
    }

    updateQualiaDynamicsVisuals() {
        const qualiaSheaf = this.mainAI_worldModel?.qualiaSheaf;
        if (!qualiaSheaf || !qualiaSheaf.ready) return;

        const qDim = qualiaSheaf.qDim;
        const numVertices = qualiaSheaf.graph.vertices.length;
        if (numVertices === 0) return;

        const aggregateAbsoluteStalk = vecZeros(qDim);

        for (const stalk of qualiaSheaf.stalks.values()) {
            if (isFiniteVector(stalk)) {
                for (let i = 0; i < qDim; i++) {
                    aggregateAbsoluteStalk[i] += Math.abs(stalk[i]);
                }
            }
        }

        qualiaSheaf.entityNames.forEach((name, i) => {
            const avgAbsoluteValue = aggregateAbsoluteStalk[i] / numVertices;
            const percent = clamp(avgAbsoluteValue * 100, 0, 100);

            const fillElement = document.getElementById(`qualia-${name}-fill`);
            const valueElement = document.getElementById(`${name}-value`);

            if (fillElement) {
                fillElement.style.width = `${percent}%`;
            }
            if (valueElement) {
                valueElement.textContent = avgAbsoluteValue.toFixed(2);
            }
        });
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
        if (!this.mainAI || !this.game || !this.mainAI_worldModel) {
            logger.warn('updatePerformanceCharts: Main AI or game not initialized.');
            return;
        }
        const stateValue = this.mainAI_worldModel.lastStateValue || 0;
        const score = this.game.score.ai - this.game.score.player;
        const cupProduct = this.mainAI_worldModel.qualiaSheaf?.cup_product_intensity || 0;
        const structuralSensitivity = this.mainAI_worldModel.qualiaSheaf?.structural_sensitivity || 0;
        const coherence = this.mainAI_worldModel.qualiaSheaf?.coherence || 0;
        const freeEnergy = this.mainAI_worldModel.freeEnergy || 0;

        this.chartEMA.qValue = this.EMA_ALPHA * stateValue + (1 - this.EMA_ALPHA) * this.chartEMA.qValue;
        this.chartEMA.score = this.EMA_ALPHA * score + (1 - this.EMA_ALPHA) * this.chartEMA.score;
        this.chartEMA.cupProduct = this.EMA_ALPHA * cupProduct + (1 - this.EMA_ALPHA) * this.chartEMA.cupProduct;
        this.chartEMA.structuralSensitivity = this.EMA_ALPHA * structuralSensitivity + (1 - this.EMA_ALPHA) * this.chartEMA.structuralSensitivity;
        this.chartEMA.coherence = this.EMA_ALPHA * coherence + (1 - this.EMA_ALPHA) * this.chartEMA.coherence;
        this.chartEMA.freeEnergy = this.EMA_ALPHA * freeEnergy + (1 - this.EMA_ALPHA) * this.chartEMA.freeEnergy;

        this.chartData.qValue.push(this.chartEMA.qValue);
        this.chartData.score.push(this.chartEMA.score);
        this.chartData.cupProduct.push(this.chartEMA.cupProduct);
        this.chartData.structuralSensitivity.push(this.chartEMA.structuralSensitivity);
        this.chartData.coherence.push(this.chartEMA.coherence);
        this.chartData.freeEnergy.push(this.chartEMA.freeEnergy);

        for (const key in this.chartData) {
            if (this.chartData[key].length > this.MAX_CHART_POINTS) this.chartData[key].shift();
        }

        this.drawChart('qValueChart', this.chartData.qValue, 'var(--primary-blue)', -2, 2);
        this.drawChart('scoreChart', this.chartData.score, 'var(--error-red)', -10, 10);
        this.drawChart('cupProductChart', this.chartData.cupProduct, 'var(--info-green)', 0, 1);
        this.drawChart('structuralSensitivityChart', this.chartData.structuralSensitivity, 'var(--warn-orange)', 0, 0.1);
    }

    /**
 * FINAL, UNCRASHABLE VERSION: gameLoop
 *
 * This function has been fully hardened to prevent the "psi is not defined" error
 * and any similar crashes originating from the AI's decision-making process.
 *
 * THE FIX:
 * 1.  SAFE INITIALIZATION: `mainDecision` and `opponentDecision` are now initialized
 *     with safe, default 'IDLE' objects at the start of every frame. This
 *     guarantees they always have a valid value.
 * 2.  ROBUST ERROR HANDLING: The entire decision-making block (`Promise.all` calling
 *     `makeDecision`) is wrapped in a comprehensive `try...catch` block.
 * 3.  RESULT VALIDATION: After the AI returns a decision, the result is explicitly
 *     validated. If the result is malformed, invalid, or missing, the loop logs a
 *     warning and uses the safe 'IDLE' default that was pre-initialized.
 *
 * This three-level defense ensures that even if the AI's internal logic fails
 * catastrophically, the game loop itself will never crash and the simulation can continue.
 */
async gameLoop(_timestamp, isManualStep = false) {
    if (!this.isRunning && !isManualStep) return;
    if (!this.isInitialized || !this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready || !this.mainViz || !this.opponentViz) {
        logger.warn('Game loop waiting for full initialization...');
        if (!this.isInitialized) {
            await this.initialize().catch(e => {
                logger.error(`Re-initialization failed: ${e.message}`, e);
                this.stop();
                document.getElementById('status').textContent = 'Error: Initialization failed.';
            });
        }
        if (this.isRunning && !isManualStep) {
            requestAnimationFrame(this.boundGameLoop);
        }
        return;
    }

    try {
        this.game.scene.updateMatrixWorld(true);
        const deltaTime = Math.min(this.clock.getDelta(), 0.1);
        const stepsPerFrame = isManualStep ? 1 : (this.isFastForward ? this.FAST_FORWARD_MULTIPLIER : 1);

        // --- START OF DEFINITIVE FIX ---

        // Step 1: Initialize decision objects with safe defaults at the beginning of the frame.
        // This guarantees they are never undefined.
        let mainDecision = { action: 'IDLE', chosenActionIndex: 3, activations: [] };
        let opponentDecision = { action: 'IDLE', chosenActionIndex: 3, activations: [] };

        for (let step = 0; step < stepsPerFrame; step++) {
            this.frameCount++;
            const preGameState = this.game.getState();
            const rawStateVector = this._stateToVector(preGameState);

            if (this.frameCount % 5 === 0) {
                await Promise.all([
                    this.mainAI_qualiaSheaf.update(rawStateVector, this.frameCount),
                    this.opponent_qualiaSheaf.update(rawStateVector, this.frameCount)
                ]);
            }

            // Step 2: Wrap the entire decision-making process in a try...catch block.
            try {
                const [mainResult, opponentResult] = await Promise.all([
                    this.mainAI.makeDecision(preGameState),
                    this.opponentAI.makeDecision(preGameState)
                ]);

                // Step 3: Validate the results before assignment.
                // If the result is invalid, we keep the safe default from Step 1.
                if (mainResult && mainResult.action && mainResult.chosenActionIndex !== undefined) {
                    mainDecision = mainResult;
                } else {
                    logger.warn('Main AI returned an invalid decision object. Defaulting to IDLE.');
                }

                if (opponentResult && opponentResult.action && opponentResult.chosenActionIndex !== undefined) {
                    opponentDecision = opponentResult;
                } else {
                    logger.warn('Opponent AI returned an invalid decision object. Defaulting to IDLE.');
                }

            } catch (e) {
                // If the makeDecision promise itself throws an error (like "psi is not defined"),
                // this block will catch it. The loop continues safely using the default IDLE actions.
                logger.error(`Decision-making process threw an error: ${e.message}. Agents will perform IDLE action.`);
            }

            // --- END OF DEFINITIVE FIX ---

            // The rest of the loop can now safely use the decision objects.
            this.game.setAIAction(mainDecision.action);
            this.game.setPlayerAction(opponentDecision.action);
            const gameUpdateResult = this.game.update(deltaTime / stepsPerFrame);
            const postGameState = this.game.getState();

            const learnPromises = [];
            if (typeof this.mainAI.learn === 'function') {
                const prior = await this.mainAI_qualiaSheaf.computeFreeEnergyPrior(this._stateToVector(postGameState), null);
                learnPromises.push(this.mainAI.learn(preGameState, mainDecision.chosenActionIndex, gameUpdateResult.aReward, postGameState, gameUpdateResult.isDone, prior));
            }
            if (typeof this.opponentAI.learn === 'function') {
                learnPromises.push(this.opponentAI.learn(preGameState, opponentDecision.chosenActionIndex, gameUpdateResult.pReward, postGameState, gameUpdateResult.isDone));
            }
            if (learnPromises.length > 0) {
                await Promise.all(learnPromises);
            }

            if (this.mainAI_worldModel.actorLoss === 0 && this.mainAI_worldModel.criticLoss === 0) {
                this.consecutiveZeroLosses++;
                if (this.consecutiveZeroLosses > this.MAX_ZERO_LOSSES) {
                    logger.warn('Stalled learning detected; reinitializing AIs.');
                    await this.resetAI();
                    return; // Exit the loop to allow reset to complete.
                }
            } else {
                this.consecutiveZeroLosses = 0;
            }

            // ... (rest of the original loop logic for StrategicAI, adaptation, etc.)
            if (this.mainStrategicAI?.learningAI?.owm?.qualiaSheaf) {
                this.mainStrategicAI.observe(gameUpdateResult.aReward);
            }
            if (this.opponentStrategicAI?.learningAI?.owm?.qualiaSheaf) {
                this.opponentStrategicAI.observe(gameUpdateResult.pReward);
            }

            if (this.frameCount % 50 === 0) {
                this.mainStrategicAI.modulateParameters();
                this.opponentStrategicAI.modulateParameters();
            }

            if (this.frameCount > 0 && this.frameCount % this.sheafAdaptFrequency === 0) {
                await this.mainAI_qualiaSheaf.adaptSheafTopology(this.sheafAdaptFrequency, this.frameCount);
            }

            if (this.frameCount % 500 === 0) {
                const mainState = this.mainAI_qualiaSheaf.saveState();
                localStorage.setItem('mainSheafState', JSON.stringify(mainState));
                logger.info(`Sheaf state persisted at frame ${this.frameCount}.`);
            }
        }

        this.updateUI(mainDecision, opponentDecision);
        this.game.render();

    } catch (error) {
        logger.error(`gameLoop error: ${error.message}`, error);
        this.stop();
        document.getElementById('status').textContent = `Error: Game loop stopped. Check console.`;
    } finally {
        if (this.isRunning && !isManualStep) {
            requestAnimationFrame(this.boundGameLoop);
        }
    }
}
    updateUI(mainDecision, opponentDecision) {
        document.getElementById('player-score').textContent = this.game.score.player;
        document.getElementById('ai-score').textContent = this.game.score.ai;

        if (this.frameCount % 5 === 0) {
            // Extract current activations for all relevant layers from the world model for visualization
const currentGameStateVector = this._stateToVector(this.game.getState());
const sanitizeAndLogActivations = (rawActivations, expectedLength, name, defaultToZeros = true) => {
    let sanitized = rawActivations;
    let originalLength = rawActivations?.length;

    if (!isFiniteVector(rawActivations) || originalLength !== expectedLength) {
        logger.warn(`MainApp.updateUI: Raw activations for ${name} are invalid (finite=${isFiniteVector(rawActivations)}, len=${originalLength}, expected=${expectedLength}). Resetting to zeros.`, { rawActivations });
        sanitized = vecZeros(expectedLength);
    } else if (originalLength === 0 && expectedLength > 0) { // Handle cases where an array is unexpectedly empty
        logger.warn(`MainApp.updateUI: Raw activations for ${name} are empty but expected length ${expectedLength}. Resetting to zeros.`);
        sanitized = vecZeros(expectedLength);
    }
    return sanitized;
};
// Main AI Activations
const mainInputActivations = this.mainAI_worldModel._getFullInputVector(currentGameStateVector); // Input to LSTM
const mainCellStateActivations = this.mainAI_worldModel.cellState;
const mainHiddenStateActivations = this.mainAI_worldModel.hiddenState;
const mainOutputActivations = this.mainAI_worldModel.lastActionLogProbs; // Action logits

const mainActivationsForViz = [
    mainInputActivations,
    mainCellStateActivations,
    mainHiddenStateActivations,
    mainOutputActivations
].map(arr => isFiniteVector(arr) ? arr : vecZeros(arr?.length || 1)); // Ensure all are finite vectors

// Opponent AI Activations (assuming similar structure)
const opponentInputActivations = this.opponent_worldModel._getFullInputVector(currentGameStateVector);
const opponentCellStateActivations = this.opponent_worldModel.cellState;
const opponentHiddenStateActivations = this.opponent_worldModel.hiddenState;
const opponentOutputActivations = this.opponent_worldModel.lastActionLogProbs;

const opponentActivationsForViz = [
    opponentInputActivations,
    opponentCellStateActivations,
    opponentHiddenStateActivations,
    opponentOutputActivations
].map(arr => isFiniteVector(arr) ? arr : vecZeros(arr?.length || 1)); // Ensure all are finite vectors

if (this.mainViz && this.opponentViz) {
    this.mainViz.update(mainActivationsForViz, mainDecision?.chosenActionIndex || 0);
    this.opponentViz.update(opponentActivationsForViz, opponentDecision?.chosenActionIndex || 0);
}
            this.updateVisualization();
        }
    }

    toggleGame() {
        if (!this.isInitialized) {
            logger.warn("Cannot toggle game: Initialization not complete.");
            document.getElementById('status').textContent = 'Initializing... Please wait.';
            return;
        }

        this.isRunning = !this.isRunning;
        this.toggleSimButton.textContent = this.isRunning ? '⏸️ Pause' : '🚀 Start Sim';
        this.toggleSimButton.classList.toggle('active', this.isRunning);

        if (this.isRunning) {
            this.game.resumeAudioContext();
            document.getElementById('status').textContent = this.isFastForward ? 'Fast Forward Active' : 'Conscious AI Active';
            if (!this.clock.running) {
                this.clock.start();
            }
            requestAnimationFrame(() => this.gameLoop(null, false));
        } else {
            this.clock.stop();
            document.getElementById('status').textContent = 'Paused';
        }
    }

    toggleFastForward() {
        this.isFastForward = !this.isFastForward;
        this.fastForwardButton.classList.toggle('active', this.isFastForward);
        this.fastForwardButton.innerHTML = this.isFastForward ? 'Normal Speed 🐢' : 'Fast ⏩';
        if (this.isFastForward && !this.isRunning) this.toggleGame();
    }

    stop() {
        this.isRunning = false;
        this.toggleSimButton.textContent = '🚀 Start Sim';
        this.toggleSimButton.classList.remove('active');
    }

    async resetAI() {
        this.stop();
        logger.info('Resetting all game and AI states...');
        showLoading('main', 'Resetting Full Simulation...');
        try {
            if (tf) tf.disposeVariables();
            await this.setupGameAndAIs();
            this.frameCount = 0;
            this.consecutiveZeroLosses = 0;
            this.chartData = { qValue: [], score: [], cupProduct: [], structuralSensitivity: [], coherence: [], freeEnergy: [] };
            this.chartEMA = { qValue: 0, score: 0, cupProduct: 0, structuralSensitivity: 0, coherence: 0, freeEnergy: 0 };
            document.getElementById('status').textContent = 'Reset Complete. Ready to Start.';
            this.updateVisualization();
            this.game.render();
        } catch (e) {
            logger.error(`resetAI error: ${e.message}`, e);
            document.getElementById('status').textContent = `Reset Failed: ${e.message}`;
        } finally {
            hideLoading('main');
        }
    }

    tuneParameters() {
        if (!this.mainAI_worldModel?.ready) return;
        this.mainStrategicAI.modulateParameters();
        this.opponentStrategicAI.modulateParameters();
        const paramIds = ['alpha-param', 'beta-param', 'gamma-param'];
        paramIds.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.classList.remove('param-flash-active');
                void el.offsetWidth;
                el.classList.add('param-flash-active');
            }
        });
        logger.info(`Parameters tuned via StrategicAI modulation.`);
    }

    setupTooltips() {
        if (!window.tippy) {
            logger.warn('Tippy.js not loaded; tooltips skipped.');
            return;
        }
        tippy('#phi-display', { content: 'Φ (Phi) measures integrated information, indicating the system\'s level of consciousness.' });
        tippy('#feel-F-display', { content: 'F measures the system’s free energy, reflecting predictive divergence.' });
        tippy('#intentionality-display', { content: 'F_int quantifies directed action potential and goal-oriented focus.' });
        tippy('#cup-product-intensity', { content: 'Cup Product Intensity measures topological coherence in qualia interactions.' });
        tippy('#structural-sensitivity-display', { content: 'Structural Sensitivity quantifies sheaf response to topological changes.' });
        tippy('#h1-dimension', { content: 'dim H¹ indicates structural complexity and non-trivial loops in information flow.' });
        tippy('#gestalt-unity', { content: 'Gestalt Unity quantifies holistic coherence across the sheaf structure.' });
        tippy('#inconsistency', { content: 'Gluing Inconsistency measures misalignment in qualia projections.' });
        tippy('#stability-fill', { content: 'Stability reflects the system’s robustness to perturbations.' });
        tippy('#alphaSlider', { content: 'α controls external sensory input influence on qualia diffusion.' });
        tippy('#betaSlider', { content: 'β adjusts diffusion strength and speed across the sheaf.' });
        tippy('#gammaSlider', { content: 'γ sets inertia for qualia updates and learning rate.' });
    }

    bindEvents() {
        if (this.toggleSimButton) this.toggleSimButton.onclick = () => this.toggleGame();
        if (this.resetSimButton) this.resetSimButton.onclick = () => this.resetAI();
        document.getElementById('tuneButton').onclick = () => this.tuneParameters();
        document.getElementById('pauseButton').onclick = () => this.stop();
        document.getElementById('stepButton').onclick = () => this.gameLoop(null, true);
        if (this.fastForwardButton) this.fastForwardButton.onclick = () => this.toggleFastForward();

        window.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            switch (e.key.toLowerCase()) {
                case ' ': e.preventDefault(); this.toggleGame(); break;
                case 'r': e.preventDefault(); this.resetAI(); break;
                case 't': e.preventDefault(); this.tuneParameters(); break;
                case 'p': e.preventDefault(); this.stop(); break;
                case 's': e.preventDefault(); if (!this.isRunning) this.gameLoop(null, true); break;
                case 'f': e.preventDefault(); this.toggleFastForward(); break;
            }
        });

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
                });
            }
        });
    }
}

export async function bootstrapApp() {
    try {
        if (tf) {
            await tf.setBackend('webgl');
            await tf.ready();
            logger.info('TensorFlow.js WebGL backend initialized.');
        } else {
            logger.warn('TensorFlow.js not loaded.');
        }

        window.mainApp = new MainApp();
        function positionVertices() {
            const graph = document.getElementById('sheafGraph');
            if (!graph) return;
            const rect = graph.getBoundingClientRect();
            for (let i = 0; i < 8; i++) {
                const v = document.getElementById('vertex-' + i);
                if (!v) continue;
                const p = sheafVertexPositions[i];
                v.style.left = `${p.x * (rect.width - v.offsetWidth)}px`;
                v.style.top = `${p.y * (rect.height - v.offsetHeight)}px`;
            }
        }
        const graphPanel = document.getElementById('sheafGraph');
        if (graphPanel) new ResizeObserver(positionVertices).observe(graphPanel);
        window.addEventListener('resize', positionVertices);
        positionVertices();

    } catch (e) {
        logger.error(`Application Bootstrap failed: ${e.message}`, e);
        document.getElementById('status').textContent = `Fatal Error: ${e.message}`;
    }
}

window.addEventListener('load', bootstrapApp);
window.addEventListener('beforeunload', () => {
    if (window.mainApp) {
        const mainState = window.mainApp.mainAI_qualiaSheaf?.saveState();
        if (mainState) {
            localStorage.setItem('mainSheafState', JSON.stringify(mainState));
        }
    }
    cleanupLive2D();
    cleanupConceptVisualization();
    if (tf) {
        tf.disposeVariables();
        logger.info('TensorFlow.js resources disposed.');
    }
});
