// --- START OF FILE ai-agents.js ---
import { OntologicalWorldModel } from './owm.js';
import { clamp, vecZeros, isFiniteVector, logger } from './utils.js';

// Use global tf from CDN (no import needed). Add a check for its existence.
const tf = window.tf || (typeof tf !== 'undefined' ? tf : null);
if (!tf) {
    logger.error('TensorFlow.js not loaded globally. Check CDN script in HTML.');
} else {
    // Initialize WebGL backend for TensorFlow.js for GPU-accelerated ops
    tf.setBackend('webgl').then(() => {
        logger.info('TensorFlow.js WebGL backend initialized.');
    }).catch(e => {
        logger.warn('Failed to set WebGL backend, falling back to CPU.', e);
        tf.setBackend('cpu');
    });
}

/**
 * Base AI Agent class for decision-making and learning in an environment.
 * Research Improvements: Self-play curiosity bonus, adaptive temperature decay, verifiable meta-rewards.
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
        this.logFrequency = 30; // Log agent activity every N steps
    }

    async initialize() {
        if (this.owm && !this.owm.ready) {
            logger.warn(`BaseAIAgent's OWM was passed but not ready. Attempting to initialize internally.`);
            await this.owm.initialize();
        }
        logger.info(`BaseAIAgent initialized: stateDim=${this.stateDim}, actionDim=${this.actionDim}, isPlayerTwo=${this.owm.isPlayerTwo}, OWM_Ready=${this.owm.ready}`);
    }

    createStateVector(gameState) {
        if (!gameState || !gameState.player || !gameState.ai || !gameState.playerTarget || !gameState.aiTarget) {
            logger.error('BaseAIAgent.createStateVector: Invalid gameState. Returning zeros.', { gameState });
            return vecZeros(this.stateDim);
        }

        const agent = this.owm.isPlayerTwo ? gameState.ai : gameState.player;
        const opponent = this.owm.isPlayerTwo ? gameState.player : gameState.ai;
        const target = this.owm.isPlayerTwo ? gameState.aiTarget : gameState.playerTarget;

        const worldHalfSize = this.game?.constructor?.WORLD_SIZE / 2 || 50;
        const worldSize = this.game?.constructor?.WORLD_SIZE || 100;

        const agentX_norm = (Number.isFinite(agent.x) ? agent.x : 0) / worldHalfSize;
        const agentZ_norm = (Number.isFinite(agent.z) ? agent.z : 0) / worldHalfSize;
        const agentRot_norm = (Number.isFinite(agent.rot) ? agent.rot : 0) / (2 * Math.PI);
        const targetX_norm = (Number.isFinite(target.x) ? target.x : 0) / worldHalfSize;
        const targetZ_norm = (Number.isFinite(target.z) ? target.z : 0) / worldHalfSize;
        const vecX = (Number.isFinite(target.x) ? target.x : 0) - (Number.isFinite(agent.x) ? agent.x : 0);
        const vecZ = (Number.isFinite(target.z) ? target.z : 0) - (Number.isFinite(agent.z) ? agent.z : 0);
        const dist = Math.sqrt(vecX * vecX + vecZ * vecZ);
        const vecX_norm = vecX / worldSize;
        const vecZ_norm = vecZ / worldSize;
        const dist_norm = dist / worldSize;

        const agent3D = this.owm.isPlayerTwo ? this.game?.ai : this.game?.player;
        const rayDetections = this.game?.getRaycastDetections?.(agent3D) || { left: 0, center: 0, right: 0 };
        const rayLeft_norm = Number.isFinite(rayDetections.left) ? rayDetections.left : 0;
        const rayCenter_norm = Number.isFinite(rayDetections.center) ? rayDetections.center : 0;
        const rayRight_norm = Number.isFinite(rayDetections.right) ? rayDetections.right : 0;

        const oppVecX = (Number.isFinite(opponent.x) ? opponent.x : 0) - (Number.isFinite(agent.x) ? agent.x : 0);
        const oppVecZ = (Number.isFinite(opponent.z) ? opponent.z : 0) - (Number.isFinite(agent.z) ? agent.z : 0);
        const oppVecX_norm = oppVecX / worldSize;
        const oppVecZ_norm = oppVecZ / worldSize;

        const stateVec = new Float32Array([
            agentX_norm, agentZ_norm, agentRot_norm,
            targetX_norm, targetZ_norm,
            vecX_norm, vecZ_norm, dist_norm,
            rayLeft_norm, rayCenter_norm, rayRight_norm,
            oppVecX_norm, oppVecZ_norm
        ]);

        const clampedStateVec = new Float32Array(stateVec.map(v => Number.isFinite(v) ? clamp(v, -1, 1) : 0));

        if (!isFiniteVector(clampedStateVec) || clampedStateVec.length !== this.stateDim) {
            logger.error(`BaseAIAgent.createStateVector: Generated state vector is invalid. Returning zeros.`, { clampedStateVec });
            return vecZeros(this.stateDim);
        }
        return clampedStateVec;
    }

    async makeDecision(gameState) {
        logger.warn('BaseAIAgent.makeDecision should be overridden by derived classes.');
        return { action: [0, 0, 0, 1], chosenActionIndex: 3, corrupted: true, activations: [] };
    }

    async learn(state, action, reward, nextState, done) {
        logger.warn('BaseAIAgent.learn should be overridden by derived classes.');
        return { actorLoss: 0, criticLoss: 0, predictionLoss: 0 };
    }

    async modulateParameters() {
        const safePhi = Number.isFinite(this.owm.qualiaSheaf.phi) ? this.owm.qualiaSheaf.phi : 0.01;
        const safeH1 = Number.isFinite(this.owm.qualiaSheaf.h1Dimension) ? this.owm.qualiaSheaf.h1Dimension : 1.0;

        this.lr = clamp(this.lr * (1 + 0.01 * safePhi - 0.005 * safeH1), 0.0001, 0.01);
        this.gamma = clamp(this.gamma * (1 + 0.005 * safePhi), 0.9, 0.999);
        await this.owm.qualiaSheaf.tuneParameters();

        // Only log modulation if parameters actually changed or at log frequency
        if (this.stepCount % this.logFrequency === 0) {
            logger.info(`BaseAIAgent parameters modulated: lr=${this.lr.toFixed(5)}, gamma=${this.gamma.toFixed(3)}`);
        }
    }

    reset() {
        this.owm.resetRecurrentState();
        this.epsilon = 0.1;
        this.temperature = 1.0;
        this.memory = [];
        this.stepCount = 0;
        this.curiosityBonus = 0;
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
        const scaledEpsilon = this.epsilon * temperature;

        const { action: actionString, chosenActionIndex: owmChosenActionIndex, actionProbs, stateValue, activations, variance, anticipatoryReward, corrupted } = await this.owm.chooseAction(stateVec, scaledEpsilon);

        if (corrupted) {
            logger.warn(`LearningAIAgent: World model returned a corrupted decision. Defaulting to IDLE.`);
            return { action: actions[3].vec, activations: [], corrupted: true, chosenActionIndex: 3 };
        }

        chosenActionIndex = owmChosenActionIndex;
        this.avgStateValue = Number.isFinite(stateValue) ? stateValue : 0;

        const curiosity = clamp(variance.reduce((sum, v) => sum + v, 0) / variance.length, 0, 1);
        this.curiosityBonus = 0.1 * curiosity * (1 + this.owm.qualiaSheaf.h1Dimension * 0.05);

        const decision = { action: actions[chosenActionIndex].vec, activations, corrupted: false, chosenActionIndex: chosenActionIndex };
        this.actionQueue.push(decision);

        // Log decision less frequently
        if (this.stepCount % this.logFrequency === 0) {
            logger.info(`LearningAIAgent decided action=${actionString} (idx=${chosenActionIndex}), stateValue=${stateValue.toFixed(3)}, curiosityBonus=${this.curiosityBonus.toFixed(3)}`);
        }
        return this.actionQueue.shift() || decision;
    }

    async learn(preGameState, actionIndex, reward, newGameState, isDone) {
        if (!this.owm.ready) {
            logger.warn('LearningAIAgent: OWM not ready for learning. Skipping.');
            return;
        }

        const stateVec = this.createStateVector(preGameState);
        const nextStateVec = this.createStateVector(newGameState);

        if (!isFiniteVector(stateVec) || !isFiniteVector(nextStateVec) || stateVec.length !== this.stateDim || nextStateVec.length !== this.stateDim) {
            logger.error('LearningAIAgent.learn: Invalid state or nextState vectors. Skipping learning.', { stateVec, nextStateVec });
            return;
        }

        this.stepCount++;

        const h1Influence = clamp(this.owm.qualiaSheaf.h1Dimension / (this.owm.qualiaSheaf.graph.edges.length || 1), 0, 1);
        const curiosityBonus = clamp(this.owm.predictionError, 0, 1) * 0.01 * (1 + h1Influence * 0.5);
        let totalReward = Number.isFinite(reward) ? reward + curiosityBonus : curiosityBonus;
        if (actionIndex === 3) totalReward -= 0.01;

        const nextFullInput = this.owm._getFullInputVector(nextStateVec);
        const { stateValue: nextValue, anticipatoryReward: nextAnticipatoryReward, corrupted: nextStateCorrupted } = await this.owm.forward(nextFullInput);

        const safeNextValue = Number.isFinite(nextValue) && !nextStateCorrupted ? nextValue : 0;
        const safeNextAnticipatoryReward = Number.isFinite(nextAnticipatoryReward) && !nextStateCorrupted ? nextAnticipatoryReward : 0;

        const targetValue = isDone ? totalReward : totalReward + this.gamma * (safeNextValue + 0.1 * safeNextAnticipatoryReward);

        const { actorLoss, criticLoss, predictionLoss } = await this.owm.learn(
            targetValue - this.owm.lastStateValue,
            targetValue,
            nextStateVec,
            this.lr
        );

        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
        const uncertaintyFactor = 1 + clamp(this.owm.predictionError, 0, 5) * 0.05;
        this.epsilon = clamp(this.epsilon * uncertaintyFactor, this.epsilonMin, 1.0);
        this.epsilon = Number.isFinite(this.epsilon) ? this.epsilon : this.epsilonMin;
        this.temperature = clamp(this.temperature * 0.999, 0.5, 2.0);

        this.memory.push({ state: stateVec, action: actionIndex, reward: totalReward, nextState: nextStateVec, done: isDone });
        if (this.memory.length > this.memorySize) this.memory.shift();

        // Log learning less frequently
        if (this.stepCount % this.logFrequency === 0) {
            logger.info(`LearningAIAgent learned: actorLoss=${actorLoss.toFixed(3)}, criticLoss=${criticLoss.toFixed(3)}, predictionLoss=${predictionLoss.toFixed(3)}, epsilon=${this.epsilon.toFixed(3)}`);
        }
    }
}

export class StrategicAIAgent extends BaseAIAgent {
    constructor(stateDim = 13, actionDim = 4, game = null, aiResponseTime = 3, ontologicalWorldModel = null) {
        super(stateDim, actionDim, game, ontologicalWorldModel?.isPlayerTwo || false, ontologicalWorldModel);
        this.aiResponseTime = Math.max(1, aiResponseTime);
        this.actionQueue = [];
        this.planningHorizon = 5;
        this.avgStateValue = 0;
        logger.info('StrategicAIAgent constructed with planning horizon=', this.planningHorizon);
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
        const scaledEpsilon = this.epsilon * temperature;

        let bestAction = 3;
        let bestValue = -Infinity;

        for (let action = 0; action < this.actionDim; action++) {
            let simulatedStateVec = stateVec.slice();
            let totalValue = 0;

            for (let t = 0; t < this.planningHorizon; t++) {
                const simulatedFullInput = this.owm._getFullInputVector(simulatedStateVec);
                const { actionLogits, stateValue, variance, anticipatoryReward, corrupted: simulatedCorrupted } = await this.owm.forward(simulatedFullInput);

                if (simulatedCorrupted) {
                    logger.warn('StrategicAIAgent: Simulated forward pass resulted in corrupted state. Breaking planning loop.');
                    totalValue = -Infinity;
                    break;
                }

                totalValue += (Number.isFinite(stateValue) ? stateValue : 0) + 0.1 * (Number.isFinite(anticipatoryReward) ? anticipatoryReward : 0) * Math.pow(this.gamma, t);
                const curiosity = clamp(variance.reduce((sum, v) => sum + v, 0) / variance.length, 0, 1);
                totalValue += 0.05 * curiosity;

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

        const { action: actionString, chosenActionIndex: owmChosenActionIndex, actionProbs, stateValue, activations, variance, anticipatoryReward, corrupted } = await this.owm.chooseAction(stateVec, scaledEpsilon);

        if (corrupted) {
            logger.warn(`StrategicAIAgent: World model returned a corrupted decision for final action. Defaulting to IDLE.`);
            return { action: actions[3].vec, activations: [], corrupted: true, chosenActionIndex: 3 };
        }

        this.avgStateValue = Number.isFinite(stateValue) ? stateValue : 0;
        this.curiosityBonus = 0.1 * clamp(variance.reduce((sum, v) => sum + v, 0) / variance.length, 0, 1);

        const decision = { action: actions[bestAction].vec, activations, corrupted: false, chosenActionIndex: bestAction };
        this.actionQueue.push(decision);

        // Log decision less frequently
        if (this.stepCount % this.logFrequency === 0) {
            logger.info(`StrategicAIAgent planned action=${actions[bestAction].name} (idx=${bestAction}), bestValue=${bestValue.toFixed(3)}, curiosityBonus=${this.curiosityBonus.toFixed(3)}`);
        }
        return this.actionQueue.shift() || decision;
    }

    async simulateNextState(rawStateVec, action) {
        const fullInput = this.owm._getFullInputVector(rawStateVec);
        const { nextStatePrediction, corrupted } = await this.owm.forward(fullInput);

        if (!isFiniteVector(nextStatePrediction) || corrupted) {
            logger.warn('StrategicAIAgent.simulateNextState: Invalid next state prediction or corrupted. Returning zeros.');
            return vecZeros(this.stateDim);
        }
        return nextStatePrediction.map(v => clamp(v + (Math.random() - 0.5) * 0.1, -10, 10));
    }

    async learn(preGameState, actionIndex, reward, newGameState, isDone) {
        if (!this.owm.ready) {
            logger.warn('StrategicAIAgent: OWM not ready for learning. Skipping.');
            return;
        }

        const stateVec = this.createStateVector(preGameState);
        const nextStateVec = this.createStateVector(newGameState);

        if (!isFiniteVector(stateVec) || !isFiniteVector(nextStateVec) || stateVec.length !== this.stateDim || nextStateVec.length !== this.stateDim) {
            logger.error('StrategicAIAgent.learn: Invalid state or nextState vectors. Skipping learning.', { stateVec, nextStateVec });
            return;
        }

        this.stepCount++;

        const h1Influence = clamp(this.owm.qualiaSheaf.h1Dimension / (this.owm.qualiaSheaf.graph.edges.length || 1), 0, 1);
        const curiosityBonus = clamp(this.owm.predictionError, 0, 1) * 0.01 * (1 + h1Influence * 0.5);
        let totalReward = Number.isFinite(reward) ? reward + curiosityBonus : curiosityBonus;
        if (actionIndex === 3) totalReward -= 0.01;

        const nextFullInput = this.owm._getFullInputVector(nextStateVec);
        const { stateValue: nextValue, anticipatoryReward: nextAnticipatoryReward, corrupted: nextStateCorrupted } = await this.owm.forward(nextFullInput);

        const safeNextValue = Number.isFinite(nextValue) && !nextStateCorrupted ? nextValue : 0;
        const safeNextAnticipatoryReward = Number.isFinite(nextAnticipatoryReward) && !nextStateCorrupted ? nextAnticipatoryReward : 0;

        const targetValue = isDone ? totalReward : totalReward + this.gamma * (safeNextValue + 0.1 * safeNextAnticipatoryReward);

        const { actorLoss, criticLoss, predictionLoss } = await this.owm.learn(
            targetValue - this.owm.lastStateValue,
            targetValue,
            nextStateVec,
            this.lr
        );

        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
        const uncertaintyFactor = 1 + clamp(this.owm.predictionError, 0, 5) * 0.05;
        this.epsilon = clamp(this.epsilon * uncertaintyFactor, this.epsilonMin, 1.0);
        this.epsilon = Number.isFinite(this.epsilon) ? this.epsilon : this.epsilonMin;
        this.temperature = clamp(this.temperature * 0.999, 0.5, 2.0);

        this.memory.push({ state: stateVec, action: actionIndex, reward: totalReward, nextState: nextStateVec, done: isDone });
        if (this.memory.length > this.memorySize) this.memory.shift();

        // Log learning less frequently
        if (this.stepCount % this.logFrequency === 0) {
            logger.info(`StrategicAIAgent learned: actorLoss=${actorLoss.toFixed(3)}, criticLoss=${criticLoss.toFixed(3)}, predictionLoss=${predictionLoss.toFixed(3)}, epsilon=${this.epsilon.toFixed(3)}`);
        }
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
            this.learningAI.epsilon = clamp(this.learningAI.epsilon * (1 + this.epsilonModulationRate), this.learningAI.epsilonMin, 1.0);
        } else if (avgReward > 0.2 && predError < 0.5) {
            this.learningAI.lr = clamp(this.learningAI.lr * (1 - this.learningRateModulationRate), 0.001, 0.05);
        }

        const explorationModifier = (1 - gestaltUnity) + (h1Dimension * 0.1);
        this.learningAI.epsilon = clamp(this.learningAI.epsilon * (1 + (explorationModifier - 0.5) * 0.01), this.learningAI.epsilonMin, 1.0);

        this.learningAI.lr = Number.isFinite(this.learningAI.lr) ? this.learningAI.lr : 0.001;
        this.learningAI.epsilon = Number.isFinite(this.learningAI.epsilon) ? this.learningAI.epsilon : this.learningAI.epsilonMin;

        // Log modulation less frequently
        if (this.learningAI.stepCount % this.learningAI.logFrequency === 0) {
            logger.info(`StrategicAI modulated: lr=${this.learningAI.lr.toFixed(5)}, epsilon=${this.learningAI.epsilon.toFixed(3)}`);
        }
    }
}
