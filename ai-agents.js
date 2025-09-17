// --- START OF FILE ai-agent.js ---
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
 * Base AI Agent class for decision-making and learning in an environment.
 */
export class BaseAIAgent {
    constructor(stateDim = 13, actionDim = 4, game = null, isPlayerTwo = false, ontologicalWorldModel = null) {
        this.game = game;
        this.owm = ontologicalWorldModel || new OntologicalWorldModel(stateDim, actionDim, 7, [64, 64], isPlayerTwo);
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.memory = [];
        this.memorySize = 1000;
        this.gamma = 0.99;
        this.epsilon = 0.1;
        this.epsilonDecay = 0.995;
        this.epsilonMin = 0.01;
        this.lr = 0.001;
        this.stepCount = 0;
        this.curiosityBonus = 0;
        this.temperature = 1.0;
        this.lastStateVec = vecZeros(stateDim);
        this.logFrequency = 30;
        this.topologicalCuriosity = 0;
        this.errorCount = 0; // Track consecutive errors
        this.maxErrors = 5; // Threshold for partial reset
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

        // Use worker to compute topological score
        const states = this.memory.map(m => m.state).filter(isFiniteVector);
        const topoScore = await runWorkerTask('topologicalScore', { states, filtration: sheaf.correlationMatrix }, 10000).catch(e => {
            logger.error(`BaseAIAgent.computeTopologicalCuriosity: Worker error: ${e.message}`);
            return { score: 0 };
        });
        const score = Number.isFinite(topoScore.score) ? topoScore.score : 0;

        this.topologicalCuriosity = clamp(h1 * 0.1 + cup * 0.05 - inconsistency * 0.1 + score * 0.2, 0, 1);
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
        this.errorCount = 0; // Reset error count on successful input

        // Compute Floquet multipliers for dynamic learning rate
        let lrAdjustment = 1.0;
        if (typeof flattenMatrix === 'function') {
// ... in modulateParameters()
try {
    const correlationMatrix = this.owm.qualiaSheaf.correlationMatrix;
    // Add validation for the correlation matrix
    if (Array.isArray(correlationMatrix) && correlationMatrix.length > 0 && Array.isArray(correlationMatrix[0])) {
        const floquetResult = await runWorkerTask('complexEigenvalues', { matrix: flattenMatrix(correlationMatrix) }, 10000);
        maxFloquet = floquetResult.length > 0 ? Math.max(...floquetResult.map(v => Math.sqrt(v.re * v.re + v.im * v.im))) : 1;
    }
} catch (e) {
    logger.error(`BaseAIAgent.modulateParameters: Error computing Floquet multipliers: ${e.message}`);
}
        } else {
            logger.error('BaseAIAgent.learn: flattenMatrix is not defined. Skipping Floquet adjustment.');
            lrAdjustment = 1.0;
        }

        const h1Influence = clamp(this.owm.qualiaSheaf.h1Dimension / (this.owm.qualiaSheaf.graph.edges.length || 1), 0, 1);
        const curiosityBonus = clamp(this.owm.predictionError, 0, 1) * 0.01 * (1 + h1Influence * 0.5);
        let totalReward = Number.isFinite(reward) ? reward + curiosityBonus + this.topologicalCuriosity * 0.05 : curiosityBonus;
        if (actionIndex === 3) totalReward -= 0.01;

        const nextFullInput = this.owm._getFullInputVector(nextStateVec);
        const { stateValue: nextValue, anticipatoryReward: nextAnticipatoryReward, corrupted: nextStateCorrupted } = await this.owm.forward(nextFullInput).catch(e => {
            logger.error(`BaseAIAgent.learn: OWM forward error: ${e.message}`);
            return { stateValue: 0, anticipatoryReward: 0, corrupted: true };
        });

        const safeNextValue = Number.isFinite(nextValue) && !nextStateCorrupted ? nextValue : 0;
        const safeNextAnticipatoryReward = Number.isFinite(nextAnticipatoryReward) && !nextStateCorrupted ? nextAnticipatoryReward : 0;

        const targetValue = isDone ? totalReward : totalReward + this.gamma * (safeNextValue + 0.1 * safeNextAnticipatoryReward);

        const { actorLoss, criticLoss, predictionLoss } = await this.owm.learn(
            targetValue - this.owm.lastStateValue,
            targetValue,
            nextStateVec,
            this.lr * lrAdjustment
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

        // Sort by topological score and keep top entries
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
                const floquetResult = await runWorkerTask('complexEigenvalues', { matrix: flattenMatrix(this.owm.qualiaSheaf.correlationMatrix) }, 10000);
                maxFloquet = floquetResult.length > 0 ? Math.max(...floquetResult.map(v => Math.sqrt(v.re * v.re + v.im * v.im))) : 1;
            } catch (e) {
                logger.error(`BaseAIAgent.modulateParameters: Error computing Floquet multipliers: ${e.message}`);
            }
        } else {
            logger.error('BaseAIAgent.modulateParameters: flattenMatrix is not defined. Skipping Floquet adjustment.');
        }

        this.lr = clamp(this.lr * (1 + 0.01 * safePhi - 0.005 * safeH1 + 0.02 * (maxFloquet - 1)), 0.0001, 0.01);
        this.gamma = clamp(this.gamma * (1 + 0.005 * safePhi), 0.9, 0.999);
        await this.owm.qualiaSheaf.tuneParameters();

        if (this.stepCount % this.logFrequency === 0) {
            logger.info(`BaseAIAgent parameters modulated: lr=${this.lr.toFixed(5)}, gamma=${this.gamma.toFixed(3)}, maxFloquet=${maxFloquet.toFixed(3)}`);
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
    constructor(stateDim = 13, actionDim = 4, game = null, aiResponseTime = 3, ontologicalWorldModel = null) {
        super(stateDim, actionDim, game, ontologicalWorldModel?.isPlayerTwo || false, ontologicalWorldModel);
        this.aiResponseTime = Math.max(1, aiResponseTime);
        this.actionQueue = [];
        this.avgStateValue = 0;
        logger.info('LearningAIAgent constructed.');
    }

    async fallbackAction(stateVec) {
        const psi = await this.owm.qualiaSheaf.computeHarmonicState();
        if (!isFiniteVector(psi)) {
            logger.warn('LearningAIAgent.fallbackAction: Invalid harmonic state. Defaulting to IDLE.');
            return { action: [0, 0, 0, 1], chosenActionIndex: 3, corrupted: true, activations: [] };
        }
        const qValues = await this.owm.getQValues?.(stateVec) || new Float32Array(this.actionDim).fill(0);
        const scores = qValues.map(q => Number.isFinite(q) ? dot(psi, vecZeros(this.stateDim).fill(q)) : 0);
        const chosenActionIndex = scores.reduce((iMax, x, i) => x > scores[iMax] ? i : iMax, 0);
        const actions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
        return { action: actions[chosenActionIndex], chosenActionIndex, corrupted: false, activations: [] };
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
        const scaledEpsilon = this.epsilon * temperature * (1 + this.topologicalCuriosity * 0.2);

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

        const curiosity = clamp(variance.reduce((sum, v) => sum + v, 0) / variance.length, 0, 1);
        this.curiosityBonus = 0.1 * curiosity * (1 + this.owm.qualiaSheaf.h1Dimension * 0.05);

        const decision = { action: actions[chosenActionIndex].vec, activations, corrupted: false, chosenActionIndex };
        this.actionQueue.push(decision);

        if (this.stepCount % this.logFrequency === 0) {
            logger.info(`LearningAIAgent decided action=${actionString} (idx=${chosenActionIndex}), stateValue=${stateValue.toFixed(3)}, curiosityBonus=${this.curiosityBonus.toFixed(3)}`);
        }
        return this.actionQueue.shift() || decision;
    }
}

export class StrategicAIAgent extends BaseAIAgent {
    constructor(stateDim = 13, actionDim = 4, game = null, aiResponseTime = 3, ontologicalWorldModel = null) {
        super(stateDim, actionDim, game, ontologicalWorldModel?.isPlayerTwo || false, ontologicalWorldModel);
        this.aiResponseTime = Math.max(1, aiResponseTime);
        this.actionQueue = [];
        this.planningHorizon = 5;
        this.avgStateValue = 0;
        logger.info(`StrategicAIAgent constructed with planningHorizon=${this.planningHorizon}`);
    }

    async fallbackAction(stateVec) {
        const psi = await this.owm.qualiaSheaf.computeHarmonicState();
        if (!isFiniteVector(psi)) {
            logger.warn('StrategicAIAgent.fallbackAction: Invalid harmonic state. Defaulting to IDLE.');
            return { action: [0, 0, 0, 1], chosenActionIndex: 3, corrupted: true, activations: [] };
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
        const scaledEpsilon = this.epsilon * temperature * (1 + this.topologicalCuriosity * 0.2);

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

                totalValue += (Number.isFinite(stateValue) ? stateValue : 0) + 0.1 * (Number.isFinite(anticipatoryReward) ? anticipatoryReward : 0) * Math.pow(this.gamma, t);
                const curiosity = clamp(variance.reduce((sum, v) => sum + v, 0) / variance.length, 0, 1);
                totalValue += 0.05 * curiosity + 0.1 * cup * dot(psi, simulatedStateVec);

                const softmaxProbs = this.owm.softmax(actionLogits);
                const actionToTake = Math.random() < scaledEpsilon
                    ? Math.floor(Math.random() * this.actionDim)
                    : softmaxProbs.reduce((maxIdx, p, i) => p > softmaxProbs[maxIdx] ? i : maxIdx, 0);

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
            logger.info(`StrategicAIAgent planned action=${actions[bestAction].name} (idx=${bestAction}), bestValue=${bestValue.toFixed(3)}, curiosityBonus=${this.curiosityBonus.toFixed(3)}`);
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
    constructor(learningAI) {
        if (!(learningAI instanceof LearningAIAgent || learningAI instanceof StrategicAIAgent)) {
            throw new Error('StrategicAI requires a LearningAIAgent or StrategicAIAgent instance.');
        }
        this.learningAI = learningAI;
        this.rewardHistory = [];
        this.HISTORY_SIZE = 200;
        this.epsilonModulationRate = 0.005;
        this.learningRateModulationRate = 0.005;
        logger.info('StrategicAI constructed.');
    }

    async chooseAction(stateVec, gameState) {
        if (!isFiniteVector(stateVec)) {
            logger.warn('StrategicAI.chooseAction: Invalid state vector. Returning IDLE.');
            return { action: 'IDLE', actionIndex: 3, F_qualia: 0 };
        }

        const sheaf = this.learningAI.owm.qualiaSheaf;
        await sheaf.diffuseQualia(stateVec).catch(e => {
            logger.error(`StrategicAI.chooseAction: Sheaf diffuseQualia error: ${e.message}`);
        });
        if (!sheaf.ready) {
            logger.warn('StrategicAI.chooseAction: Sheaf not ready. Falling back to base agent.');
            return await this.learningAI.makeDecision(gameState);
        }

        const { phi, h1Dimension, cup_product_intensity, inconsistency } = sheaf;
        const F_qualia = Number.isFinite(phi) && Number.isFinite(h1Dimension) && Number.isFinite(cup_product_intensity) && Number.isFinite(inconsistency)
            ? phi * h1Dimension * cup_product_intensity * (1 - inconsistency)
            : 0;

        const psi = await sheaf.computeHarmonicState().catch(e => {
            logger.error(`StrategicAI.chooseAction: Sheaf computeHarmonicState error: ${e.message}`);
            return vecZeros(this.learningAI.stateDim);
        });
        if (!isFiniteVector(psi)) {
            logger.warn('StrategicAI.chooseAction: Invalid harmonic state. Falling back to base agent.');
            return await this.learningAI.makeDecision(gameState);
        }

        const qualiaInfluence = sheaf.mapQualiaToGame?.(stateVec, null) || new Float32Array(this.learningAI.actionDim).fill(0);
        if (!isFiniteVector(qualiaInfluence)) {
            logger.warn('StrategicAI.chooseAction: Invalid qualia influence. Using zeros.');
            qualiaInfluence.fill(0);
        }

        const { action: baseAction, chosenActionIndex, activations } = await this.learningAI.makeDecision(gameState);
        const qValues = await this.learningAI.owm.getQValues?.(stateVec) || new Float32Array(this.learningAI.actionDim).fill(0);
        if (!isFiniteVector(qValues)) {
            logger.warn('StrategicAI.chooseAction: Invalid Q-values. Returning base decision.');
            return { action: baseAction, actionIndex: chosenActionIndex, F_qualia };
        }

        const gradF_qualia = qValues.map(q => Number.isFinite(q) && Number.isFinite(cup_product_intensity) && Number.isFinite(inconsistency)
            ? q * cup_product_intensity * (1 - inconsistency)
            : 0);
        const scores = qValues.map((q, idx) => {
            const psiDot = Number.isFinite(gradF_qualia[idx]) ? dot(psi, gradF_qualia[idx]) : 0;
            return q + this.learningAI.epsilon * psiDot + qualiaInfluence[idx] + this.learningAI.topologicalCuriosity * 0.1;
        });

        const actionIndexMod = scores.reduce((iMax, x, i) => Number.isFinite(x) && x > scores[iMax] ? i : iMax, 0);
        const action = ['FORWARD', 'LEFT', 'RIGHT', 'IDLE'][actionIndexMod];

        if (this.learningAI.stepCount % this.learningAI.logFrequency === 0) {
            logger.info(`StrategicAI action=${action} (idx=${actionIndexMod}), F_qualia=${F_qualia.toFixed(4)}, phi=${phi.toFixed(3)}, topoCuriosity=${this.learningAI.topologicalCuriosity.toFixed(3)}`);
        }
        return { action, actionIndex: actionIndexMod, F_qualia };
    }

    observe(reward) {
        if (Number.isFinite(reward)) {
            this.rewardHistory.push(reward);
            if (this.rewardHistory.length > this.HISTORY_SIZE) {
                this.rewardHistory.shift();
            }
        }
    }

    modulateParameters() {
        if (this.rewardHistory.length < this.HISTORY_SIZE / 2) return;

        const avgReward = this.rewardHistory.reduce((a, b) => a + b, 0) / this.rewardHistory.length;
        const predError = Number.isFinite(this.learningAI.owm.predictionError) ? this.learningAI.owm.predictionError : 0;
        const gestaltUnity = Number.isFinite(this.learningAI.owm.qualiaSheaf.gestaltUnity) ? this.learningAI.owm.qualiaSheaf.gestaltUnity : 0.5;
        const h1Dimension = Number.isFinite(this.learningAI.owm.qualiaSheaf.h1Dimension) ? this.learningAI.owm.qualiaSheaf.h1Dimension : 1.0;

        if (avgReward < 0.05 && predError > 1.0) {
            this.learningAI.lr = clamp(this.learningAI.lr * (1 + this.learningRateModulationRate), 0.001, 0.05);
            this.learningAI.epsilon = clamp(this.learningAI.epsilon * (1 + this.epsilonModulationRate + this.learningAI.topologicalCuriosity * 0.01), this.learningAI.epsilonMin, 1.0);
        } else if (avgReward > 0.2 && predError < 0.5) {
            this.learningAI.lr = clamp(this.learningAI.lr * (1 - this.learningRateModulationRate), 0.001, 0.05);
        }

        const explorationModifier = (1 - gestaltUnity) + (h1Dimension * 0.1) + this.learningAI.topologicalCuriosity * 0.2;
        this.learningAI.epsilon = clamp(this.learningAI.epsilon * (1 + (explorationModifier - 0.5) * 0.01), this.learningAI.epsilonMin, 1.0);

        this.learningAI.lr = Number.isFinite(this.learningAI.lr) ? this.learningAI.lr : 0.001;
        this.learningAI.epsilon = Number.isFinite(this.learningAI.epsilon) ? this.learningAI.epsilon : this.learningAI.epsilonMin;

        if (this.learningAI.stepCount % this.learningAI.logFrequency === 0) {
            logger.info(`StrategicAI modulated: lr=${this.learningAI.lr.toFixed(5)}, epsilon=${this.learningAI.epsilon.toFixed(3)}, topoCuriosity=${this.learningAI.topologicalCuriosity.toFixed(3)}`);
        }
    }
}
// --- END OF FILE ai-agent.js ---
