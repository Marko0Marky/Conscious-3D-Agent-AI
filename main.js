import { logger, showLoading, hideLoading, clamp, isFiniteVector, norm2, vecZeros, vecAdd } from './utils.js';
import { OntologicalWorldModel } from './owm.js';
import { LearningAIAgent, StrategicAI } from './ai-agents.js';
import { ThreeDeeGame } from './three-dee-game.js';
import { NeuralNetworkVisualizer } from './nn-visualizer.js';
import { initLive2D, updateLive2DEmotions, cleanupLive2D, isLive2DReady, updateLive2D } from './viz-live2d.js';
import { initConceptVisualization, cleanupConceptVisualization, renderConceptVisualization, animateConceptNodes, isConceptVisualizationReady, updateAgentSimulationVisuals } from './viz-concepts.js';
import { EnhancedQualiaSheaf } from './qualia-sheaf.js';
import * as THREE from 'three';

const tf = window.tf;

const sheafVertexPositions = {
    0: {x:0.1, y:0.5}, 1: {x:0.3, y:0.2}, 2: {x:0.3, y:0.8},
    3: {x:0.9, y:0.5}, 4: {x:0.7, y:0.2}, 5: {x:0.5, y:0.35},
    6: {x:0.7, y:0.8}, 7: {x:0.5, y:0.65}
};

class MainApp {
    constructor() {
        logger.info('MainApp constructor started.');
        this.gameCanvas = document.getElementById('gameCanvas');
        this.sheafGraphCanvas = document.getElementById('sheafGraphCanvas');
        if (!this.gameCanvas || !this.sheafGraphCanvas) {
            logger.error('Canvas not found');
            document.getElementById('status').textContent = 'Error: Canvas not found';
            throw new Error('Canvas not found');
        }
        this.sheafGraphCtx = this.sheafGraphCanvas.getContext('2d');
        if (!this.sheafGraphCtx) {
            logger.error('Failed to get 2D context for sheaf');
            document.getElementById('status').textContent = 'Error: Failed to get 2D context for sheaf';
            throw new Error('Failed to get 2D context for sheaf');
        }
        this.clock = new THREE.Clock();
        this.chartData = { qValue: [], score: [], cupProduct: [], structuralSensitivity: [] };
        this.chartEMA = { qValue: 0, score: 0, cupProduct: 0, structuralSensitivity: 0 };
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
        this.sheafStepCount = 0;
        this.sheafAdaptFrequency = 200;
        this.consecutiveZeroLosses = 0;
        this.MAX_ZERO_LOSSES = 50;
        this.lastPhi = 0;
        this.boundGameLoop = this.gameLoop.bind(this);
        this.bindEvents();
        this.setupTooltips();
        // FIX: Moved this.setupQualiaAttentionPanel() to setupGameAndAIs to ensure sheaf is initialized first.
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
        const STATE_DIM = 13;
        const ACTION_DIM = 4;
        const Q_DIM = 7;

        // STEP 1: Create the EnhancedQualiaSheaf instances.
        this.mainAI_qualiaSheaf = new EnhancedQualiaSheaf(null, STATE_DIM, Q_DIM, 0.1, 0.1, 0.05);
        this.opponent_qualiaSheaf = new EnhancedQualiaSheaf(null, STATE_DIM, Q_DIM, 0.1, 0.1, 0.05);

        // STEP 2: Asynchronously initialize the sheaves. They must be ready first.
        await Promise.all([
            this.mainAI_qualiaSheaf.initialize(),
            this.opponent_qualiaSheaf.initialize()
        ]);
        logger.info("Core QualiaSheafs initialized successfully.");

        // STEP 3: Now that sheaves are ready, create the OWMs that depend on their structure.
        this.mainAI_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], false, this.mainAI_qualiaSheaf);
        this.opponent_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], true, this.opponent_qualiaSheaf);
        
        // STEP 4: Provide the OWM reference back to the sheaf for metrics like actorLoss.
        this.mainAI_qualiaSheaf.setOWM(this.mainAI_worldModel);
        this.opponent_qualiaSheaf.setOWM(this.opponent_worldModel);

        // STEP 5: Initialize the OWMs.
        let retries = 3;
        while (retries > 0) {
            try {
                await Promise.all([
                    this.mainAI_worldModel.initialize(),
                    this.opponent_worldModel.initialize()
                ]);
                if (this.mainAI_worldModel.ready && this.opponent_worldModel.ready) {
                    logger.info("Core OWMs are ready.");
                    break;
                }
                throw new Error('OWM not ready after initialization call');
            } catch (e) {
                retries--;
                logger.warn(`OWM initialization attempt failed, retries left: ${retries}`, e);
                if (retries === 0) throw e;
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        }

        // STEP 6: Initialize visualizations and AI agents with fully ready components.
        await Promise.all([
            initLive2D(this.clock, this.mainAI_qualiaSheaf),
            initConceptVisualization(this.clock, this.mainAI_qualiaSheaf)
        ]);

        this.mainAI = new LearningAIAgent(STATE_DIM, ACTION_DIM, this.game, 3, this.mainAI_worldModel);
        this.opponentAI = new LearningAIAgent(STATE_DIM, ACTION_DIM, this.game, 3, this.opponent_worldModel);
        
        this.mainStrategicAI = new StrategicAI(this.mainAI);
        this.opponentStrategicAI = new StrategicAI(this.opponentAI);
        this.mainViz = new NeuralNetworkVisualizer('nn-visualization-container', this.mainAI_worldModel, 'main');
        this.opponentViz = new NeuralNetworkVisualizer('nn-visualization-container-opponent', this.opponent_worldModel, 'opponent');
        
        // FIX: Call setup for UI elements that depend on the sheaf *after* the sheaf is created.
        this.setupQualiaAttentionPanel();

        this.game.render();
        this.updateVisualization();
        document.getElementById('status').textContent = 'Ready. Click "Toggle Sim" to start.';

    } catch (e) {
        logger.error('CRITICAL FAILURE during setupGameAndAIs:', e);
        document.getElementById('status').textContent = `Initialization Failed: ${e.message}`;
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

    const qualiaInfo = qualiaSheaf.entityNames.map(name => ({
        idName: name.toLowerCase().replace(/\s+/g, '-'),
        displayName: name.charAt(0).toUpperCase() + name.slice(1)
    }));

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
        if (i < qualiaInfo.length) {
            const avgValue = aggregateStalk[i] / numVertices;
            const clampedValue = clamp(avgValue, 0, 1);
            const { idName } = qualiaInfo[i];
            
            const fillElement = document.getElementById(`qualia-${idName}-fill`);
            const valueElement = document.getElementById(`${idName}-value`);

            if (fillElement) fillElement.style.width = `${clampedValue * 100}%`;
            if (valueElement) valueElement.textContent = clampedValue.toFixed(3);
        }
    }
}

    updateVisualization() {
        const qualia = this.mainAI_worldModel?.qualiaSheaf;
        if (!qualia || !this.mainAI_worldModel || !this.mainAI) return;

        this.updateQualiaBars(qualia);

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
            this.updateQualiaAttentionVisuals();
            this.updatePerformanceCharts();

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

    setupQualiaAttentionPanel() {
        const panel = document.getElementById('qualiaAttentionPanel');
        if (!panel) return;
        const qualiaNames = this.mainAI_qualiaSheaf.entityNames;
        let html = `<h4 id="qualia-attention-heading">Qualia Attention (Main AI)</h4>`;
        qualiaNames.forEach((name, i) => {
            html += `
                <div class="attention-bar-container">
                    <span class="attention-label">${name.charAt(0).toUpperCase() + name.slice(1)}</span>
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
    if (!this.mainAI || !this.game || !this.mainAI_worldModel) return;
    
    const stateValue = this.mainAI_worldModel.lastStateValue || this.mainAI.avgStateValue || 0;
    const score = this.game.score.ai - this.game.score.player;
    const cupProduct = this.mainAI_worldModel.qualiaSheaf.cup_product_intensity || 0;
    const structuralSensitivity = this.mainAI_worldModel.qualiaSheaf.structural_sensitivity || 0;

    this.chartEMA.qValue = this.EMA_ALPHA * stateValue + (1 - this.EMA_ALPHA) * this.chartEMA.qValue;
    this.chartEMA.score = this.EMA_ALPHA * score + (1 - this.EMA_ALPHA) * this.chartEMA.score;
    this.chartEMA.cupProduct = this.EMA_ALPHA * cupProduct + (1 - this.EMA_ALPHA) * this.chartEMA.cupProduct;
    this.chartEMA.structuralSensitivity = this.EMA_ALPHA * structuralSensitivity + (1 - this.EMA_ALPHA) * this.chartEMA.structuralSensitivity;

    this.chartData.qValue.push(this.chartEMA.qValue);
    this.chartData.score.push(this.chartEMA.score);
    this.chartData.cupProduct.push(this.chartEMA.cupProduct);
    this.chartData.structuralSensitivity.push(this.chartEMA.structuralSensitivity);

    for (const key in this.chartData) {
        if (this.chartData[key].length > this.MAX_CHART_POINTS) {
            this.chartData[key].shift();
        }
    }

    this.drawChart('qValueChart', this.chartData.qValue, 'var(--primary-blue)', -2, 2);
    this.drawChart('scoreChart', this.chartData.score, 'var(--error-red)', -10, 10);
    this.drawChart('cupProductChart', this.chartData.cupProduct, 'var(--info-green)', 0, 1);
    this.drawChart('structuralSensitivityChart', this.chartData.structuralSensitivity, 'var(--warn-orange)', 0, 0.1);
}

    async gameLoop(timestamp, isManualStep = false) {
        if (!this.isRunning && !isManualStep) return;

        if (!this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready) {
            logger.warn(`gameLoop: OWMs not ready (Main: ${this.mainAI_worldModel?.ready}, Opponent: ${this.opponent_worldModel?.ready}). Re-initializing.`);
            document.getElementById('status').textContent = 'Error: Simulation not ready, re-initializing...';
            try {
                await Promise.all([
                    this.mainAI_qualiaSheaf.initialize(),
                    this.opponent_qualiaSheaf.initialize(),
                    this.mainAI_worldModel.initialize(),
                    this.opponent_worldModel.initialize(),
                    this.mainAI.reset(),
                    this.opponentAI.reset()
                ]);
            } catch (e) {
                logger.error(`gameLoop: Re-initialization failed: ${e.message}`);
                this.stop();
                document.getElementById('status').textContent = 'Error: Critical re-initialization failed.';
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
                        logger.error(`mainAI.makeDecision error: ${err.message}`);
                        return { action: 'IDLE', chosenActionIndex: 3, corrupted: true, activations: [] };
                    }),
                    this.opponentAI.makeDecision(preGameState).catch(err => {
                        logger.error(`opponentAI.makeDecision error: ${err.message}`);
                        return { action: 'IDLE', chosenActionIndex: 3, corrupted: true, activations: [] };
                    })
                ]);

                if (mainDecision.corrupted || opponentDecision.corrupted) {
                    corruptedCount++;
                    logger.warn(`Corrupted decision (main: ${mainDecision.corrupted}, opp: ${opponentDecision.corrupted}). Count: ${corruptedCount}/${MAX_CORRUPTED}`);
                    if (corruptedCount >= MAX_CORRUPTED) {
                        logger.warn('Too many corrupted decisions; resetting AIs.');
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
                    logger.warn('Invalid game update result; resetting game.');
                    this.game.reset();
                    continue;
                }

                const postGameState = this.game.getState() || {};
                await Promise.all([
                    this.mainAI.learn(preGameState, mainDecision.chosenActionIndex, gameUpdateResult.aReward, postGameState, gameUpdateResult.isDone).catch(err => {
                        logger.error(`mainAI.learn error: ${err.message}`);
                    }),
                    this.opponentAI.learn(preGameState, opponentDecision.chosenActionIndex, gameUpdateResult.pReward, postGameState, gameUpdateResult.isDone).catch(err => {
                        logger.error(`opponentAI.learn error: ${err.message}`);
                    })
                ]);

                if (gameUpdateResult.aReward > 0.5) {
                    // This method does not exist in the provided `qualia-sheaf.js`
                    // await this.mainAI_qualiaSheaf.trainPhenoMaps(gameUpdateResult.aReward);
                }

                if (this.mainAI_worldModel.actorLoss === 0 && this.mainAI_worldModel.criticLoss === 0 && this.mainAI_worldModel.predictionLoss === 0) {
                    this.consecutiveZeroLosses++;
                    if (this.consecutiveZeroLosses > this.MAX_ZERO_LOSSES) {
                        logger.warn('Stalled learning detected; reinitializing.');
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

                this.mainStrategicAI.observe(gameUpdateResult.aReward);
                this.opponentStrategicAI.observe(gameUpdateResult.pReward);

                if (this.sheafStepCount % this.sheafAdaptFrequency === 0) {
                    await Promise.all([
                        this.mainAI_qualiaSheaf.adaptSheafTopology(this.sheafAdaptFrequency, this.sheafStepCount),
                        this.opponent_qualiaSheaf.adaptSheafTopology(this.sheafAdaptFrequency, this.sheafStepCount)
                    ]);
                    await this.mainAI_qualiaSheaf.computeStructuralSensitivity();
                }

                if (this.frameCount % 50 === 0) {
                    this.mainStrategicAI.modulateParameters();
                    this.opponentStrategicAI.modulateParameters();
                }
            }

            document.getElementById('player-score').textContent = this.game.score.player;
            document.getElementById('ai-score').textContent = this.game.score.ai;
            if (this.frameCount % 5 === 0 || isManualStep) {
                this.mainViz.update(mainDecision?.activations || [], mainDecision?.chosenActionIndex || 0);
                this.opponentViz.update(opponentDecision?.activations || [], opponentDecision?.chosenActionIndex || 0);
                this.updateVisualization();
            }
            this.game.render();
            if (isConceptVisualizationReady()) renderConceptVisualization();
        } catch (error) {
            logger.error(`gameLoop error: ${error.message}`);
            this.stop();
            document.getElementById('status').textContent = `Error: Game loop stopped due to ${error.message}`;
            await this.resetAI();
        } finally {
            if (this.isRunning && !isManualStep) requestAnimationFrame(this.boundGameLoop);
        }
    }

    toggleGame() {
        if (!this.mainAI_worldModel?.ready || !this.opponent_worldModel?.ready) {
            logger.warn(`toggleGame: OWMs not ready (Main: ${this.mainAI_worldModel?.ready}, Opponent: ${this.opponent_worldModel?.ready}).`);
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
            logger.warn('toggleFastForward: OWMs not ready.');
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

            // FIX: Use the same configuration object here for consistency.
            const sheafConfig = {
                stateDim: STATE_DIM,
                qDim: Q_DIM,
                alpha: 0.1,
                beta: 0.1,
                gamma: 0.05,
                sigma: 0.025
            };
            this.mainAI_qualiaSheaf = new EnhancedQualiaSheaf(null, sheafConfig);
            this.opponent_qualiaSheaf = new EnhancedQualiaSheaf(null, sheafConfig);

            await Promise.all([
                this.mainAI_qualiaSheaf.initialize(),
                this.opponent_qualiaSheaf.initialize()
            ]);

            this.mainAI_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], false, this.mainAI_qualiaSheaf);
            this.opponent_worldModel = new OntologicalWorldModel(STATE_DIM, ACTION_DIM, Q_DIM, [64, 64], true, this.opponent_qualiaSheaf);
            this.mainAI_qualiaSheaf.setOWM(this.mainAI_worldModel);
            this.opponent_qualiaSheaf.setOWM(this.opponent_worldModel);

            let retries = 3;
            while (retries > 0) {
                try {
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

            this.mainAI = new LearningAIAgent(STATE_DIM, ACTION_DIM, this.game, 3, this.mainAI_worldModel);
            this.opponentAI = new LearningAIAgent(STATE_DIM, ACTION_DIM, this.game, 3, this.opponent_worldModel);
            this.mainStrategicAI = new StrategicAI(this.mainAI);
            this.opponentStrategicAI = new StrategicAI(this.opponentAI);
            this.mainViz.worldModel = this.mainAI_worldModel;
            this.opponentViz.worldModel = this.opponent_worldModel;

            this.frameCount = 0;
            this.sheafStepCount = 0;
            this.consecutiveZeroLosses = 0;
            this.chartData = { qValue: [], score: [], cupProduct: [], structuralSensitivity: [] };
            this.chartEMA = { qValue: 0, score: 0, cupProduct: 0, structuralSensitivity: 0 };
            this.updatePerformanceCharts();
            this.updateVisualization();
            this.game.render();
            document.getElementById('status').textContent = 'Reset Complete.';
        } catch (e) {
            logger.error(`resetAI error: ${e.message}`);
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
            logger.warn('tuneParameters: OWMs not ready.');
            document.getElementById('status').textContent = 'Error: AI not ready';
            return;
        }
        // This method does not exist in the provided `qualia-sheaf.js`
        // await Promise.all([
        //     this.mainAI_qualiaSheaf.tuneParameters(),
        //     this.opponent_qualiaSheaf.tuneParameters()
        // ]);
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

    setupTooltips() {
        if (!window.tippy) {
            logger.warn('Tippy.js not loaded; tooltips skipped.');
            return;
        }
        tippy('#phi-display', { content: 'Φ (Phi) measures integrated information, indicating the system\'s level of consciousness.' });
        tippy('#feel-F-display', { content: 'F measures the system’s free energy, reflecting predictive divergence.' });
        tippy('#psi-norm-display', { content: '||Ψ|| represents the norm of the harmonic state, indicating qualia dynamics.' });
        tippy('#intentionality-display', { content: 'F_int quantifies directed action potential and goal-oriented focus.' });
        tippy('#free-energy', { content: 'Free Energy quantifies predictive divergence from the world model.' });
        tippy('#prediction-error', { content: 'Prediction Error measures discrepancy between predicted and actual states.' });
        tippy('#gestalt-unity', { content: 'Gestalt Unity quantifies holistic coherence across the sheaf structure.' });
        tippy('#h1-dimension', { content: 'dim H¹ indicates structural complexity and non-trivial loops in information flow.' });
        tippy('#inconsistency', { content: 'Gluing Inconsistency measures misalignment in qualia projections.' });
        tippy('#learning-rate', { content: 'Learning rate (alpha) of the AI\'s reinforcement learning algorithm.' });
        tippy('#epsilon-value', { content: 'Exploration rate (epsilon); higher values mean more random actions.' });
        tippy('#qualia-being-fill', { content: 'Being: Reflects the system’s fundamental existence and self-awareness.' });
        tippy('#qualia-intent-fill', { content: 'Intent: Captures directed action potential and goal-oriented focus.' });
        tippy('#qualia-existence-fill', { content: 'Existence: Models state persistence and environmental robustness.' });
        tippy('#qualia-emergence-fill', { content: 'Emergence: Represents non-linear state synthesis and novelty.' });
        tippy('#qualia-gestalt-fill', { content: 'Gestalt: Quantifies holistic coherence and pattern recognition.' });
        tippy('#qualia-context-fill', { content: 'Context: Incorporates environmental modulation and background information.' });
        tippy('#qualia-rel-emergence-fill', { content: 'Relational Emergence: Captures dynamic entity coupling and interactions.' });
        tippy('#alphaSlider', { content: 'α controls external sensory input influence on qualia diffusion.' });
        tippy('#betaSlider', { content: 'β adjusts diffusion strength and speed across the sheaf.' });
        tippy('#gammaSlider', { content: 'γ sets inertia for qualia updates and learning rate.' });
        tippy('#toggleSimButton', { content: 'Toggles simulation run/pause state. (Spacebar)' });
        tippy('#resetSimButton', { content: 'Resets game and AI states. (R key)' });
        tippy('#tuneButton', { content: 'Adaptively adjusts AI parameters (α, β, γ). (T key)' });
        tippy('#pauseButton', { content: 'Pauses the simulation. (P key or Spacebar)' });
        tippy('#stepButton', { content: 'Advances simulation by one step. (S key)' });
        tippy('#fastForwardButton', { content: 'Toggles fast forward mode. (F key)' });
        tippy('#vertex-0', { content: 'Agent-X: The agent\'s X-axis position.' });
        tippy('#vertex-1', { content: 'Agent-Z: The agent\'s Z-axis position.' });
        tippy('#vertex-2', { content: 'Agent-Rot: The agent\'s rotation on the Y-axis.' });
        tippy('#vertex-3', { content: 'Target-X: The target\'s X-axis position.' });
        tippy('#vertex-4', { content: 'Target-Z: The target\'s Z-axis position.' });
        tippy('#vertex-5', { content: 'Vec-DX: X-component of the vector to the target.' });
        tippy('#vertex-6', { content: 'Vec-DZ: Z-component of the vector to the target.' });
        tippy('#vertex-7', { content: 'Dist-Target: Direct distance to the target.' });
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

        window.addEventListener('keydown', (e) => {
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
                    slider.setAttribute('aria-valuetext', value.toFixed(3));
                });
            }
        });
    }
}

async function bootstrapApp() {
    if (tf) {
        try {
            await tf.setBackend('webgl');
            logger.info('TensorFlow.js WebGL backend initialized.');
            await tf.ready();
            if (tf.getBackend() !== 'webgl') {
                logger.warn('WebGL backend not active; falling back to CPU.');
                await tf.setBackend('cpu');
            }
        } catch (e) {
            logger.error(`Failed to set TensorFlow.js WebGL backend: ${e.message}`);
            await tf.setBackend('cpu');
        }
    } else {
        logger.warn('TensorFlow.js not loaded; proceeding without it.');
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
        logger.info('UI initialized; ready for interaction.');
    } catch (e) {
        logger.error(`Initialization failed: ${e.message}`);
        document.getElementById('status').textContent = `Initialization Error: ${e.message}`;
    }
}

window.addEventListener('load', bootstrapApp);
window.addEventListener('beforeunload', () => {
    cleanupLive2D();
    cleanupConceptVisualization();
    if (tf) {
        tf.disposeVariables();
        logger.info('TensorFlow.js resources disposed.');
    }
});
