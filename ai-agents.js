// --- START OF FILE ai-agents.js ---
import { OntologicalWorldModel } from './owm.js';
import { clamp, vecZeros, isFiniteVector, logger, runWorkerTask, dot, flattenMatrix } from './utils.js';

// Check for TensorFlow.js global availability
const tf = window.tf || (typeof tf !== 'undefined' ? tf : null);
if (!tf) {
    logger.error('TensorFlow.js not loaded globally. Check CDN script in HTML.');
} else {
    tf.setBackend('webgl').then(() => {
        logger.info('TensorFlow.js WebGL backend initialized.');
    }).catch(e => {
        logger.warn('Failed to set WebGL backend, falling back to CPU.', { error: e.message });
        tf.setBackend('cpu');
    });
}

/**
 * BaseAIAgent: Foundation for decision-making in a topological game environment.
 * Integrates sheaf-driven curiosity for exploration (Th. 1, Th. 3).
 */
export class BaseAIAgent {
    constructor(worldModel, learningRate = 0.001, gamma = 0.99) {
        if (!worldModel) { // Simplified check
            throw new Error("BaseAIAgent must be initialized with an OntologicalWorldModel instance.");
        }
        this.owm = worldModel;
        this.game = null; // Can be set later if needed
        this.stateDim = worldModel.stateDim;
        this.actionDim = worldModel.actionDim;
        this.memory = [];
        this.memorySize = 1000;
        this.gamma = gamma;
        this.epsilon = 0.1;
        this.epsilonDecay = 0.995;
        this.epsilonMin = 0.01;
        this.lr = learningRate;
        this.stepCount = 0;
        this.curiosityBonus = 0;
        this.temperature = 1.0;
        this.lastStateVec = vecZeros(this.stateDim);
        this.logFrequency = 30;
        this.topologicalCuriosity = 0;
        this.errorCount = 0;
        this.maxErrors = 5;
        logger.info(`BaseAIAgent initialized: stateDim=${this.stateDim}, actionDim=${this.actionDim}, isPlayerTwo=${this.owm.isPlayerTwo}`);
    }

    async initialize() {
        if (!this.owm.ready) {
            logger.warn('BaseAIAgent: OWM not ready. Initializing.');
            await this.owm.initialize();
        }
        logger.info(`BaseAIAgent ready: OWM_Ready=${this.owm.ready}`);
    }

    createStateVector(gameState) {
        if (!gameState || typeof gameState !== 'object') {
            logger.error('BaseAIAgent.createStateVector: Invalid gameState.', { gameState });
            return vecZeros(this.stateDim);
        }

        const requiredProps = ['player', 'ai', 'playerTarget', 'aiTarget'];
        for (const prop of requiredProps) {
            if (!gameState[prop] || typeof gameState[prop] !== 'object') {
                logger.error(`BaseAIAgent.createStateVector: Missing or invalid ${prop}.`, { gameState });
                return vecZeros(this.stateDim);
            }
            if (!Number.isFinite(gameState[prop].x) || !Number.isFinite(gameState[prop].z) || !Number.isFinite(gameState[prop].rot)) {
                logger.error(`BaseAIAgent.createStateVector: Non-finite values in ${prop}.`, gameState[prop]);
                return vecZeros(this.stateDim);
            }
        }

        const agent = this.owm.isPlayerTwo ? gameState.ai : gameState.player;
        const opponent = this.owm.isPlayerTwo ? gameState.player : gameState.ai;
        const target = this.owm.isPlayerTwo ? gameState.aiTarget : gameState.playerTarget;

        const worldHalfSize = this.game?.constructor?.WORLD_SIZE / 2 || 50;
        const worldSize = this.game?.constructor?.WORLD_SIZE || 100;

        const agentX_norm = clamp(agent.x / worldHalfSize, -1, 1);
        const agentZ_norm = clamp(agent.z / worldHalfSize, -1, 1);
        const agentRot_norm = clamp(agent.rot / (2 * Math.PI), -1, 1);
        const targetX_norm = clamp(target.x / worldHalfSize, -1, 1);
        const targetZ_norm = clamp(target.z / worldHalfSize, -1, 1);
        const vecX = target.x - agent.x;
        const vecZ = target.z - agent.z;
        const dist = Math.sqrt(vecX * vecX + vecZ * vecZ);
        const vecX_norm = clamp(vecX / worldSize, -1, 1);
        const vecZ_norm = clamp(vecZ / worldSize, -1, 1);
        const dist_norm = clamp(dist / worldSize, -1, 1);

        const agent3D = this.owm.isPlayerTwo ? this.game?.ai : this.game?.player;
        const rayDetections = this.game?.getRaycastDetections?.(agent3D) || { left: 0, center: 0, right: 0 };
        const rayLeft_norm = clamp(Number.isFinite(rayDetections.left) ? rayDetections.left : 0, 0, 1);
        const rayCenter_norm = clamp(Number.isFinite(rayDetections.center) ? rayDetections.center : 0, 0, 1);
        const rayRight_norm = clamp(Number.isFinite(rayDetections.right) ? rayDetections.right : 0, 0, 1);

        const oppVecX = opponent.x - agent.x;
        const oppVecZ = opponent.z - agent.z;
        const oppVecX_norm = clamp(oppVecX / worldSize, -1, 1);
        const oppVecZ_norm = clamp(oppVecZ / worldSize, -1, 1);

        const stateVec = new Float32Array([
            agentX_norm, agentZ_norm, agentRot_norm,
            targetX_norm, targetZ_norm,
            vecX_norm, vecZ_norm, dist_norm,
            rayLeft_norm, rayCenter_norm, rayRight_norm,
            oppVecX_norm, oppVecZ_norm
        ]);

        if (!isFiniteVector(stateVec) || stateVec.length !== this.stateDim) {
            logger.error('BaseAIAgent.createStateVector: Invalid state vector.', { stateVec });
            return vecZeros(this.stateDim);
        }
        return stateVec;
    }

    async computeTopologicalCuriosity() {
        const sheaf = this.owm.qualiaSheaf;
        const h1 = Number.isFinite(sheaf.h1Dimension) ? sheaf.h1Dimension : 1;
        const cup = Number.isFinite(sheaf.cup_product_intensity) ? sheaf.cup_product_intensity : 0;
        const inconsistency = Number.isFinite(sheaf.inconsistency) ? sheaf.inconsistency : 0;

        // Th. 15: Hierarchical curiosity from sheaf
        const states = this.memory.map(m => m.state).filter(isFiniteVector);
        const topoScore = await runWorkerTask('topologicalScore', { states, filtration: sheaf.correlationMatrix }, 10000).catch(e => {
            logger.error(`BaseAIAgent.computeTopologicalCuriosity: Worker error: ${e.message}`);
            return { score: 0 };
        });
        const score = Number.isFinite(topoScore.score) ? topoScore.score : 0;

        // Th. 17: Incorporate Floquet rhythmic awareness
        const rhythmInfluence = sheaf.rhythmicallyAware ? sheaf.phi * 0.1 : 0;
        this.topologicalCuriosity = clamp(h1 * 0.1 + cup * 0.05 - inconsistency * 0.1 + score * 0.2 + rhythmInfluence, 0, 1);
        return this.topologicalCuriosity;
    }

    async learn(preGameState, actionIndex, reward, newGameState, isDone) {
        if (!this.owm.ready) {
            logger.warn('BaseAIAgent: OWM not ready for learning. Skipping.');
            return { actorLoss: 0, criticLoss: 0, predictionLoss: 0 };
        }

        const stateVec = this.createStateVector(preGameState);
        const nextStateVec = this.createStateVector(newGameState);

        if (!isFiniteVector(stateVec) || !isFiniteVector(nextStateVec) || stateVec.length !== this.stateDim || nextStateVec.length !== this.stateDim) {
            logger.error('BaseAIAgent.learn: Invalid state or nextState vectors.', { stateVec, nextStateVec });
            this.errorCount++;
            if (this.errorCount >= this.maxErrors) {
                logger.warn('BaseAIAgent: Too many errors in learn. Triggering partial reset.');
                this.partialReset();
            }
            return { actorLoss: 0, criticLoss: 0, predictionLoss: 0 };
        }

        this.stepCount++;
        this.errorCount = 0;

        // Th. 17: Floquet multiplier for dynamic learning rate
        let lrAdjustment = 1.0;
        if (typeof flattenMatrix === 'function') {
            try {
                const correlationMatrix = this.owm.qualiaSheaf.correlationMatrix;
                if (Array.isArray(correlationMatrix) && correlationMatrix.length > 0 && Array.isArray(correlationMatrix[0])) {
                    const floquetResult = await runWorkerTask('complexEigenvalues', { matrix: flattenMatrix(correlationMatrix) }, 10000);
                    const maxFloquet = floquetResult.length > 0 ? Math.max(...floquetResult.map(v => Math.sqrt(v.re * v.re + v.im * v.im))) : 1;
                    lrAdjustment = clamp(1 + 0.02 * (maxFloquet - 1), 0.8, 1.2);
                }
            } catch (e) {
                logger.error(`BaseAIAgent.learn: Error computing Floquet multipliers: ${e.message}`);
                lrAdjustment = 1.0;
            }
        } else {
            logger.error('BaseAIAgent.learn: flattenMatrix is not defined. Skipping Floquet adjustment.');
            lrAdjustment = 1.0;
        }

        // Th. 1 & 3: Sheaf-driven curiosity and free-energy reward
        const h1Influence = clamp(this.owm.qualiaSheaf.h1Dimension / (this.owm.qualiaSheaf.graph.edges.length || 1), 0, 1);
        const coherence = this.owm.qualiaSheaf.coherence || 0;
        const curiosityBonus = clamp(this.owm.predictionError, 0, 1) * 0.01 * (1 + h1Influence * 0.5 + coherence * 0.2);
        let totalReward = Number.isFinite(reward) ? reward + curiosityBonus + this.topologicalCuriosity * 0.05 : curiosityBonus;
        if (actionIndex === 3) totalReward -= 0.01; // IDLE penalty

        const nextFullInput = this.owm._getFullInputVector(nextStateVec);
        const { stateValue: nextValue, anticipatoryReward: nextAnticipatoryReward, corrupted: nextStateCorrupted } = await this.owm.forward(nextFullInput).catch(e => {
            logger.error(`BaseAIAgent.learn: OWM forward error: ${e.message}`);
            return { stateValue: 0, anticipatoryReward: 0, corrupted: true };
        });

        const safeNextValue = Number.isFinite(nextValue) && !nextStateCorrupted ? nextValue : 0;
        const safeNextAnticipatoryReward = Number.isFinite(nextAnticipatoryReward) && !nextStateCorrupted ? nextAnticipatoryReward : 0;

        const targetValue = isDone ? totalReward : totalReward + this.gamma * (safeNextValue + 0.1 * safeNextAnticipatoryReward);

        // Th. 3: Free-energy prior for learning
        const prior = await this.owm.qualiaSheaf.computeFreeEnergyPrior(nextStateVec, this.owm.hiddenState);
        const { actorLoss, criticLoss, predictionLoss } = await this.owm.learn(
            targetValue - this.owm.lastStateValue,
            targetValue,
            nextStateVec,
            this.lr * lrAdjustment,
            prior
        ).catch(e => {
            logger.error(`BaseAIAgent.learn: OWM learn error: ${e.message}`);
            this.errorCount++;
            if (this.errorCount >= this.maxErrors) {
                logger.warn('BaseAIAgent: Too many errors in learn. Triggering partial reset.');
                this.partialReset();
            }
            return { actorLoss: 0, criticLoss: 0, predictionLoss: 0 };
        });

        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
        await this.computeTopologicalCuriosity().catch(e => {
            logger.error(`BaseAIAgent.learn: Error computing topological curiosity: ${e.message}`);
            this.topologicalCuriosity = 0;
        });
        const uncertaintyFactor = 1 + clamp(this.owm.predictionError + this.topologicalCuriosity, 0, 5) * 0.05;
        this.epsilon = clamp(this.epsilon * uncertaintyFactor, this.epsilonMin, 1.0);
        this.epsilon = Number.isFinite(this.epsilon) ? this.epsilon : this.epsilonMin;
        this.temperature = clamp(this.temperature * 0.999, 0.5, 2.0);

        this.memory.push({ state: stateVec, action: actionIndex, reward: totalReward, nextState: nextStateVec, done: isDone, topoScore: this.topologicalCuriosity });
        if (this.memory.length > this.memorySize) {
            this.prioritizeMemory();
        }

        if (this.stepCount % this.logFrequency === 0) {
            logger.info(`BaseAIAgent learned: actorLoss=${actorLoss.toFixed(3)}, criticLoss=${criticLoss.toFixed(3)}, predictionLoss=${predictionLoss.toFixed(3)}, epsilon=${this.epsilon.toFixed(3)}, topoCuriosity=${this.topologicalCuriosity.toFixed(3)}`);
        }

        return { actorLoss, criticLoss, predictionLoss };
    }

    prioritizeMemory() {
        if (this.memory.length <= this.memorySize / 2) return;
        this.memory.sort((a, b) => (b.topoScore || 0) - (a.topoScore || 0));
        this.memory = this.memory.slice(0, this.memorySize);
    }

    partialReset() {
        this.epsilon = 0.1;
        this.temperature = 1.0;
        this.curiosityBonus = 0;
        this.topologicalCuriosity = 0;
        this.errorCount = 0;
        logger.info('BaseAIAgent: Partial reset performed to recover from errors.');
    }

    async modulateParameters() {
        const safePhi = Number.isFinite(this.owm.qualiaSheaf.phi) ? this.owm.qualiaSheaf.phi : 0.01;
        const safeH1 = Number.isFinite(this.owm.qualiaSheaf.h1Dimension) ? this.owm.qualiaSheaf.h1Dimension : 1.0;
        let maxFloquet = 1;
        if (typeof flattenMatrix === 'function') {
            try {
                const correlationMatrix = this.owm.qualiaSheaf.correlationMatrix;
                if (Array.isArray(correlationMatrix) && correlationMatrix.length > 0 && Array.isArray(correlationMatrix[0])) {
                    const floquetResult = await runWorkerTask('complexEigenvalues', { matrix: flattenMatrix(correlationMatrix) }, 10000);
                    maxFloquet = floquetResult.length > 0 ? Math.max(...floquetResult.map(v => Math.sqrt(v.re * v.re + v.im * v.im))) : 1;
                }
            } catch (e) {
                logger.error(`BaseAIAgent.modulateParameters: Error computing Floquet multipliers: ${e.message}`);
            }
        } else {
            logger.error('BaseAIAgent.modulateParameters: flattenMatrix is not defined. Skipping Floquet adjustment.');
        }

        // Th. 17: Rhythmic modulation
        const rhythmInfluence = this.owm.qualiaSheaf.rhythmicallyAware ? 0.02 * safePhi : 0;
        this.lr = clamp(this.lr * (1 + 0.01 * safePhi - 0.005 * safeH1 + 0.02 * (maxFloquet - 1) + rhythmInfluence), 0.0001, 0.01);
        this.gamma = clamp(this.gamma * (1 + 0.005 * safePhi), 0.9, 0.999);
        await this.owm.qualiaSheaf.tuneParameters();

        if (this.stepCount % this.logFrequency === 0) {
            logger.info(`BaseAIAgent parameters modulated: lr=${this.lr.toFixed(5)}, gamma=${this.gamma.toFixed(3)}, maxFloquet=${maxFloquet.toFixed(3)}, rhythmInfluence=${rhythmInfluence.toFixed(3)}`);
        }
    }

    reset() {
        this.owm.resetRecurrentState();
        this.epsilon = 0.1;
        this.temperature = 1.0;
        this.memory = [];
        this.stepCount = 0;
        this.curiosityBonus = 0;
        this.topologicalCuriosity = 0;
        this.errorCount = 0;
        this.lr = 0.001;
        this.lastStateVec.fill(0);
        logger.info(`BaseAIAgent reset: isPlayerTwo=${this.owm.isPlayerTwo}`);
    }
}

export class LearningAIAgent extends BaseAIAgent {
    constructor(worldModel, aiResponseTime = 3) {
        super(worldModel, 0.0005, 0.99); // Pass worldModel up, set learning rate
        this.aiResponseTime = Math.max(1, aiResponseTime);
        this.actionQueue = [];
        this.avgStateValue = 0;
        logger.info('LearningAIAgent constructed.');
    }

    async fallbackAction(stateVec) {
        // FIX: Ensure a consistently shaped (but zero-filled) activations array is returned
        // to prevent potential downstream errors, e.g., in visualization.
        const fallbackActivations = [
            vecZeros(this.owm.inputDim),
            vecZeros(this.owm.recurrentStateSize),
            vecZeros(this.owm.recurrentStateSize),
            vecZeros(this.actionDim)
        ];

        // FIX: Add a safety check to ensure `computeHarmonicState` exists on the sheaf instance before calling it.
        if (typeof this.owm.qualiaSheaf.computeHarmonicState !== 'function') {
            logger.error('LearningAIAgent.fallbackAction: computeHarmonicState is not a function. Defaulting to IDLE.');
            return { action: [0, 0, 0, 1], chosenActionIndex: 3, corrupted: true, activations: fallbackActivations };
        }
        
        const psi = await this.owm.qualiaSheaf.computeHarmonicState();
        if (!isFiniteVector(psi)) {
            logger.warn('LearningAIAgent.fallbackAction: Invalid harmonic state. Defaulting to IDLE.');
            return { action: [0, 0, 0, 1], chosenActionIndex: 3, corrupted: true, activations: fallbackActivations };
        }
        
        const qValues = await this.owm.getQValues?.(stateVec) || new Float32Array(this.actionDim).fill(0);
        const scores = qValues.map(q => Number.isFinite(q) ? dot(psi, vecZeros(this.stateDim).fill(q)) : 0);
        const chosenActionIndex = scores.reduce((iMax, x, i) => x > scores[iMax] ? i : iMax, 0);
        const actions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

        return { action: actions[chosenActionIndex], chosenActionIndex, corrupted: true, activations: fallbackActivations };
    }

    async makeDecision(gameState) {
        if (!this.owm.ready) {
            logger.warn('LearningAIAgent: OWM not ready; choosing IDLE action.');
            return { action: [0, 0, 0, 1], chosenActionIndex: 3, corrupted: true, activations: [] };
        }

        if (this.actionQueue.length >= this.aiResponseTime) {
            return this.actionQueue.shift();
        }

        const actions = [
            { name: 'FORWARD', vec: [1, 0, 0, 0] },
            { name: 'LEFT', vec: [0, 1, 0, 0] },
            { name: 'RIGHT', vec: [0, 0, 1, 0] },
            { name: 'IDLE', vec: [0, 0, 0, 1] }
        ];
        let chosenActionIndex = 3;

        const stateVec = this.createStateVector(gameState);
        this.lastStateVec = stateVec;
        const temperature = clamp(1.0 / (1 + Math.exp(-this.stepCount / 1000)), 0.1, 1.0);
        this.temperature = temperature;

        // Th. 3 & 17: Sheaf-driven epsilon with Floquet rhythm
        const coherence = this.owm.qualiaSheaf.coherence || 0;
        const rhythmInfluence = this.owm.qualiaSheaf.rhythmicallyAware ? this.owm.qualiaSheaf.phi * 0.2 : 0;
        const scaledEpsilon = this.epsilon * temperature * (1 + this.topologicalCuriosity * 0.2 + coherence * 0.1 + rhythmInfluence);

        const { action: actionString, chosenActionIndex: owmChosenActionIndex, actionProbs, stateValue, activations, variance, anticipatoryReward, corrupted } = await this.owm.chooseAction(stateVec, scaledEpsilon).catch(e => {
            logger.error(`LearningAIAgent.makeDecision: OWM chooseAction error: ${e.message}`);
            return { action: 'IDLE', chosenActionIndex: 3, actionProbs: [], stateValue: 0, activations: [], variance: [], anticipatoryReward: 0, corrupted: true };
        });

        if (corrupted) {
            logger.warn('LearningAIAgent: World model returned corrupted decision. Using fallback.');
            return await this.fallbackAction(stateVec);
        }

        chosenActionIndex = owmChosenActionIndex;
        this.avgStateValue = Number.isFinite(stateValue) ? stateValue : 0;

        // Th. 3: Curiosity bonus with free-energy influence
        const curiosity = clamp(variance.reduce((sum, v) => sum + v, 0) / variance.length, 0, 1);
        this.curiosityBonus = 0.1 * curiosity * (1 + this.owm.qualiaSheaf.h1Dimension * 0.05 + this.owm.freeEnergy * 0.02);

        const decision = { action: actions[chosenActionIndex].vec, activations, corrupted: false, chosenActionIndex };
        this.actionQueue.push(decision);

        if (this.stepCount % this.logFrequency === 0) {
        }
        return this.actionQueue.shift() || decision;
    }
}

export class StrategicAIAgent extends BaseAIAgent {
    constructor(worldModel, aiResponseTime = 3) {
        super(worldModel, 0.001, 0.99); // Pass worldModel up, set learning rate
        this.aiResponseTime = Math.max(1, aiResponseTime);
        this.actionQueue = [];
        this.planningHorizon = 5;
        this.avgStateValue = 0;
        logger.info(`StrategicAIAgent constructed with planningHorizon=${this.planningHorizon}`);
    }

    async fallbackAction(stateVec) {
        const fallbackActivations = [
            vecZeros(this.owm.inputDim),
            vecZeros(this.owm.recurrentStateSize),
            vecZeros(this.owm.recurrentStateSize),
            vecZeros(this.actionDim)
        ];
        // FIX: Add a safety check to ensure `computeHarmonicState` exists on the sheaf instance before calling it.
        if (typeof this.owm.qualiaSheaf.computeHarmonicState !== 'function') {
            logger.error('StrategicAIAgent.fallbackAction: computeHarmonicState is not a function. Defaulting to IDLE.');
            return { action: [0, 0, 0, 1], chosenActionIndex: 3, corrupted: true, activations: fallbackActivations };
        }
        const psi = await this.owm.qualiaSheaf.computeHarmonicState();
        if (!isFiniteVector(psi)) {
            logger.warn('StrategicAIAgent.fallbackAction: Invalid harmonic state. Defaulting to IDLE.');
            return { action: [0, 0, 0, 1], chosenActionIndex: 3, corrupted: true, activations: fallbackActivations };
        }
        const qValues = await this.owm.getQValues?.(stateVec) || new Float32Array(this.actionDim).fill(0);
        const scores = qValues.map(q => Number.isFinite(q) ? dot(psi, vecZeros(this.stateDim).fill(q)) : 0);
        const chosenActionIndex = scores.reduce((iMax, x, i) => x > scores[iMax] ? i : iMax, 0);
        const actions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
        return { action: actions[chosenActionIndex], chosenActionIndex, corrupted: false, activations: [] };
    }

    async makeDecision(gameState) {
        if (!this.owm.ready) {
            logger.warn('StrategicAIAgent: OWM not ready; choosing IDLE action.');
            return { action: [0, 0, 0, 1], chosenActionIndex: 3, corrupted: true, activations: [] };
        }

        if (this.actionQueue.length >= this.aiResponseTime) {
            return this.actionQueue.shift();
        }

        const actions = [
            { name: 'FORWARD', vec: [1, 0, 0, 0] },
            { name: 'LEFT', vec: [0, 1, 0, 0] },
            { name: 'RIGHT', vec: [0, 0, 1, 0] },
            { name: 'IDLE', vec: [0, 0, 0, 1] }
        ];

        const stateVec = this.createStateVector(gameState);
        this.lastStateVec = stateVec;
        const temperature = clamp(1.0 / (1 + Math.exp(-this.stepCount / 1000)), 0.1, 1.0);
        this.temperature = temperature;

        // Th. 3 & 17: Sheaf-driven epsilon with Floquet rhythm
        const coherence = this.owm.qualiaSheaf.coherence || 0;
        const rhythmInfluence = this.owm.qualiaSheaf.rhythmicallyAware ? this.owm.qualiaSheaf.phi * 0.2 : 0;
        const scaledEpsilon = this.epsilon * temperature * (1 + this.topologicalCuriosity * 0.2 + coherence * 0.1 + rhythmInfluence);

        // FIX: Add a safety check to ensure `computeHarmonicState` exists on the sheaf instance before calling it.
        if (typeof this.owm.qualiaSheaf.computeHarmonicState !== 'function') {
            logger.error('StrategicAIAgent.makeDecision: computeHarmonicState is not a function. Defaulting to fallback.');
            return await this.fallbackAction(stateVec);
        }
        const psi = await this.owm.qualiaSheaf.computeHarmonicState();
        const cup = Number.isFinite(this.owm.qualiaSheaf.cup_product_intensity) ? this.owm.qualiaSheaf.cup_product_intensity : 0;
        let bestAction = 3;
        let bestValue = -Infinity;

        for (let action = 0; action < this.actionDim; action++) {
            let simulatedStateVec = stateVec.slice();
            let totalValue = 0;

            for (let t = 0; t < this.planningHorizon; t++) {
                const simulatedFullInput = this.owm._getFullInputVector(simulatedStateVec);
                const { actionLogits, stateValue, variance, anticipatoryReward, corrupted: simulatedCorrupted } = await this.owm.forward(simulatedFullInput).catch(e => {
                    logger.error(`StrategicAIAgent.makeDecision: OWM forward error: ${e.message}`);
                    return { actionLogits: [], stateValue: 0, variance: [], anticipatoryReward: 0, corrupted: true };
                });

                if (simulatedCorrupted) {
                    logger.warn('StrategicAIAgent: Simulated forward pass corrupted. Breaking planning loop.');
                    totalValue = -Infinity;
                    break;
                }

                // Th. 3: Free-energy and coherence influence
                const freeEnergy = this.owm.freeEnergy || 0;
                totalValue += (Number.isFinite(stateValue) ? stateValue : 0) + 0.1 * (Number.isFinite(anticipatoryReward) ? anticipatoryReward : 0) * Math.pow(this.gamma, t);
                const curiosity = clamp(variance.reduce((sum, v) => sum + v, 0) / variance.length, 0, 1);
                totalValue += 0.05 * curiosity + 0.1 * cup * dot(psi, simulatedStateVec) - 0.02 * freeEnergy;

                const softmaxProbs = this.owm.softmax(actionLogits);
                const actionToTake = Math.random() < scaledEpsilon
                    ? Math.floor(Math.random() * this.actionDim)
                    : softmaxProbs.reduce((maxIdx, p, i) => p > softmaxProbs[maxIdx] ? i : iMax, 0);

                simulatedStateVec = await this.simulateNextState(simulatedStateVec, actionToTake);
                if (!isFiniteVector(simulatedStateVec)) {
                    logger.warn('StrategicAIAgent: Invalid simulated state; breaking planning loop.');
                    totalValue = -Infinity;
                    break;
                }
            }

            if (totalValue > bestValue) {
                bestValue = totalValue;
                bestAction = action;
            }
        }

        const { action: actionString, chosenActionIndex: owmChosenActionIndex, actionProbs, stateValue, activations, variance, anticipatoryReward, corrupted } = await this.owm.chooseAction(stateVec, scaledEpsilon).catch(e => {
            logger.error(`StrategicAIAgent.makeDecision: OWM chooseAction error: ${e.message}`);
            return { action: 'IDLE', chosenActionIndex: 3, actionProbs: [], stateValue: 0, activations: [], variance: [], anticipatoryReward: 0, corrupted: true };
        });

        if (corrupted) {
            logger.warn('StrategicAIAgent: World model returned corrupted decision. Using fallback.');
            return await this.fallbackAction(stateVec);
        }

        this.avgStateValue = Number.isFinite(stateValue) ? stateValue : 0;
        this.curiosityBonus = 0.1 * clamp(variance.reduce((sum, v) => sum + v, 0) / variance.length, 0, 1);

        const decision = { action: actions[bestAction].vec, activations, corrupted: false, chosenActionIndex: bestAction };
        this.actionQueue.push(decision);

        if (this.stepCount % this.logFrequency === 0) {
            logger.info(`StrategicAIAgent planned action=${actions[bestAction].name} (idx=${bestAction}), bestValue=${bestValue.toFixed(3)}, curiosityBonus=${this.curiosityBonus.toFixed(3)}, freeEnergy=${this.owm.freeEnergy.toFixed(3)}`);
        }
        return this.actionQueue.shift() || decision;
    }

    async simulateNextState(rawStateVec, action) {
        if (!isFiniteVector(rawStateVec)) {
            logger.warn('StrategicAIAgent.simulateNextState: Invalid input state. Returning zeros.');
            return vecZeros(this.stateDim);
        }
        const fullInput = this.owm._getFullInputVector(rawStateVec);
        const { nextStatePrediction, corrupted } = await this.owm.forward(fullInput).catch(e => {
            logger.error(`StrategicAIAgent.simulateNextState: OWM forward error: ${e.message}`);
            return { nextStatePrediction: vecZeros(this.stateDim), corrupted: true };
        });

        if (!isFiniteVector(nextStatePrediction) || corrupted) {
            logger.warn('StrategicAIAgent.simulateNextState: Invalid or corrupted prediction. Returning zeros.');
            return vecZeros(this.stateDim);
        }
        return nextStatePrediction.map(v => clamp(v + (Math.random() - 0.5) * 0.1, -10, 10));
    }
}

export class StrategicAI {
    constructor(worldModel) {
        if (!worldModel) {
            throw new Error('StrategicAI requires an OntologicalWorldModel instance.');
        }
        // FIX: The StrategicAI should wrap a LearningAIAgent, not a BaseAIAgent,
        // as it relies on the `makeDecision` method which is defined in the learning agent.
        this.learningAI = new LearningAIAgent(worldModel);
        this.rewardHistory = [];
        this.HISTORY_SIZE = 200;
        this.epsilonModulationRate = 0.005;
        this.learningRateModulationRate = 0.005;
        logger.info('StrategicAI constructed.');
    }

    // Add this method for compatibility with the game loop
    async makeDecision(gameState) {
        // FIX: Add a safety check for the existence of the chooseAction method before calling it.
        if (typeof this.chooseAction !== 'function') {
            logger.error('StrategicAI.makeDecision: chooseAction is not a function. Falling back to learningAI.');
            return await this.learningAI.makeDecision(gameState);
        }
        const stateVec = this.learningAI.createStateVector(gameState);
        return await this.chooseAction(stateVec, gameState);
    }

   // In ai-agents.js, inside the StrategicAI class

    async chooseAction(stateVec, gameState) {
        if (!isFiniteVector(stateVec)) {
            logger.warn('StrategicAI.chooseAction: Invalid state vector. Returning IDLE.');
            return { action: 'IDLE', chosenActionIndex: 3, corrupted: true, F_qualia: 0, activations: [] };
        }

        const sheaf = this.learningAI.owm.qualiaSheaf;
        await sheaf.diffuseQualia(stateVec).catch(e => {
            logger.error(`StrategicAI.chooseAction: Sheaf diffuseQualia error: ${e.message}`);
        });

        if (!sheaf.ready) {
            logger.warn('StrategicAI.chooseAction: Sheaf not ready. Falling back to base agent.');
            return await this.learningAI.makeDecision(gameState);
        }

        const { phi, h1Dimension, cup_product_intensity, inconsistency, coherence } = sheaf;
        const F_qualia = Number.isFinite(phi) && Number.isFinite(h1Dimension) && Number.isFinite(cup_product_intensity) && Number.isFinite(inconsistency)
            ? phi * h1Dimension * cup_product_intensity * (1 - inconsistency) * (1 + coherence * 0.5)
            : 0;
        
        const psi = await sheaf.computeHarmonicState().catch(e => {
            logger.error(`StrategicAI.chooseAction: Sheaf computeHarmonicState error: ${e.message}`);
            return vecZeros(this.learningAI.stateDim);
        });

        if (!isFiniteVector(psi)) {
            logger.warn('StrategicAI.chooseAction: Invalid harmonic state. Falling back to base agent.');
            return await this.learningAI.makeDecision(gameState);
        }
        
        const { action: baseAction, chosenActionIndex, activations, corrupted } = await this.learningAI.makeDecision(gameState);

        // If the base agent's decision was corrupted, we should not proceed with strategic override.
        if (corrupted) {
            return { action: 'IDLE', chosenActionIndex: 3, corrupted: true, F_qualia: F_qualia, activations: [] };
        }

        const qValues = await this.learningAI.owm.getQValues?.(stateVec) || new Float32Array(this.learningAI.actionDim).fill(0);
        if (!isFiniteVector(qValues)) {
            logger.warn('StrategicAI.chooseAction: Invalid Q-values. Returning base decision.');
            return { action: baseAction, chosenActionIndex, F_qualia, activations, corrupted: false };
        }

        const prior = await sheaf.computeFreeEnergyPrior(stateVec, qValues);
        const biasedQ = qValues.map((q, i) => q + prior[i] * 0.1 * coherence);

        const awarenessCascade = (sheaf.selfAware ? 1 : 0) + (sheaf.hierarchicallyAware ? 1 : 0) +
                                (sheaf.diachronicallyAware ? 1 : 0) + (sheaf.rhythmicallyAware ? 1.5 : 0);
        const gradF_qualia = biasedQ.map(q => Number.isFinite(q) && Number.isFinite(cup_product_intensity) && Number.isFinite(inconsistency)
            ? q * cup_product_intensity * (1 - inconsistency) * (1 + awarenessCascade * 0.05)
            : 0);
        const scores = biasedQ.map((q, idx) => {
            const psiDot = Number.isFinite(gradF_qualia[idx]) ? dot(psi, gradF_qualia[idx]) : 0;
            return q + this.learningAI.epsilon * psiDot + this.learningAI.topologicalCuriosity * 0.1 + awarenessCascade * 0.02;
        });

        const actionIndexMod = scores.reduce((iMax, x, i) => Number.isFinite(x) && x > scores[iMax] ? i : iMax, 0);
        const action = ['FORWARD', 'LEFT', 'RIGHT', 'IDLE'][actionIndexMod];

        // --- START OF FIX ---
        // Pass the original activations from the learning agent through in the final return object.
        return { 
            action, 
            chosenActionIndex: actionIndexMod, 
            F_qualia, 
            activations, // Now included!
            corrupted: false
        };
        // --- END OF FIX ---
    }

    observe(reward) {
        if (Number.isFinite(reward)) {
            this.rewardHistory.push(reward);
            if (this.rewardHistory.length > this.HISTORY_SIZE) {
                this.rewardHistory.shift();
            }
            // Th. 1: Update topological curiosity with coherence
            const sheaf = this.learningAI.owm.qualiaSheaf;
            this.learningAI.topologicalCuriosity = clamp(this.learningAI.topologicalCuriosity + sheaf.coherence * 0.01, 0, 1);
        }
    }


    modulateParameters() {
        if (this.rewardHistory.length < this.HISTORY_SIZE / 2) return;

        const avgReward = this.rewardHistory.reduce((a, b) => a + b, 0) / this.rewardHistory.length;
        const predError = Number.isFinite(this.learningAI.owm.predictionError) ? this.learningAI.owm.predictionError : 0;
        const gestaltUnity = Number.isFinite(this.learningAI.owm.qualiaSheaf.gestaltUnity) ? this.learningAI.owm.qualiaSheaf.gestaltUnity : 0.5;
        const h1Dimension = Number.isFinite(this.learningAI.owm.qualiaSheaf.h1Dimension) ? this.learningAI.owm.qualiaSheaf.h1Dimension : 1.0;

        // Th. 3 & 17: Free-energy and rhythmic modulation
        const freeEnergy = this.learningAI.owm.freeEnergy || 0;
        const rhythmInfluence = this.learningAI.owm.qualiaSheaf.rhythmicallyAware ? this.learningAI.owm.qualiaSheaf.phi * 0.05 : 0;
        if (avgReward < 0.05 && predError > 1.0) {
            this.learningAI.lr = clamp(this.learningAI.lr * (1 + this.learningRateModulationRate + freeEnergy * 0.01), 0.001, 0.05);
            this.learningAI.epsilon = clamp(this.learningAI.epsilon * (1 + this.epsilonModulationRate + this.learningAI.topologicalCuriosity * 0.01 + rhythmInfluence), this.learningAI.epsilonMin, 1.0);
        } else if (avgReward > 0.2 && predError < 0.5) {
            this.learningAI.lr = clamp(this.learningAI.lr * (1 - this.learningRateModulationRate - freeEnergy * 0.005), 0.001, 0.05);
        }

        // Th. 14–17: Awareness cascade for exploration
        const awarenessCascade = (this.learningAI.owm.qualiaSheaf.selfAware ? 1 : 0) +
                                (this.learningAI.owm.qualiaSheaf.hierarchicallyAware ? 1 : 0) +
                                (this.learningAI.owm.qualiaSheaf.diachronicallyAware ? 1 : 0) +
                                (this.learningAI.owm.qualiaSheaf.rhythmicallyAware ? 1.5 : 0);
        const explorationModifier = (1 - gestaltUnity) + (h1Dimension * 0.1) + this.learningAI.topologicalCuriosity * 0.2 + awarenessCascade * 0.05;
        this.learningAI.epsilon = clamp(this.learningAI.epsilon * (1 + (explorationModifier - 0.5) * 0.01), this.learningAI.epsilonMin, 1.0);

        this.learningAI.lr = Number.isFinite(this.learningAI.lr) ? this.learningAI.lr : 0.001;
        this.learningAI.epsilon = Number.isFinite(this.learningAI.epsilon) ? this.learningAI.epsilon : this.learningAI.epsilonMin;

        if (this.learningAI.stepCount % this.learningAI.logFrequency === 0) {
            logger.info(`StrategicAI modulated: lr=${this.learningAI.lr.toFixed(5)}, epsilon=${this.learningAI.epsilon.toFixed(3)}, topoCuriosity=${this.learningAI.topologicalCuriosity.toFixed(3)}, awarenessCascade=${awarenessCascade.toFixed(1)}`);
        }
    }
}
// --- END OF FILE ai-agents.js ---
