// --- START OF FILE ai-agents.js ---
// --- FIXED VERSION: Removed ES import, uses global tf for browser CDN ---
import { OntologicalWorldModel } from './owm.js';
import { clamp, vecZeros, isFiniteVector, logger, norm2 } from './utils.js';

// Use global tf from CDN (no import needed). Add a check for its existence.
const tf = window.tf || (typeof tf !== 'undefined' ? tf : null);
if (!tf) {
    logger.error('TensorFlow.js not loaded globally. Check CDN script in HTML.');
} else {
    // Initialize WebGL backend for TensorFlow.js
    tf.setBackend('webgl').then(() => {
        logger.info('TensorFlow.js WebGL backend initialized.');
    }).catch(e => {
        logger.warn('Failed to set WebGL backend, falling back to CPU.', e);
        tf.setBackend('cpu');
    });
}

/**
 * Implements a Reinforcement Learning agent using an Ontological World Model.
 * Manages exploration-exploitation balance with softmax policy and state representation.
 */
export class LearningAI {
    /**
     * @param {OntologicalWorldModel} worldModel - The OWM this AI will use.
     * @param {ThreeDeeGame} game - The game instance, needed for raycasting.
     * @param {boolean} isPlayerTwo - True if this AI controls the second agent.
     * @param {number} aiResponseTime - Number of frames between AI decisions.
     */
    constructor(worldModel, game, isPlayerTwo = false, aiResponseTime = 3) {
        this.worldModel = worldModel;
        this.game = game;
        this.isPlayerTwo = isPlayerTwo;
        this.aiResponseTime = Math.max(1, aiResponseTime);
        this.actionQueue = [];

        this.epsilon = 1.0;
        this.epsilonMin = 0.05;
        this.epsilonDecay = 0.9995;
        this.learningRate = 0.01;
        this.temperature = 1.0; // Softmax temperature for exploration
        
        this.lastStateVec = vecZeros(this.worldModel.stateDim);
        this.lastActionIndex = 3; // Corresponds to 'IDLE'
        this.lastActivations = [];
        this.avgStateValue = 0;
    }

    /**
     * Creates a normalized state vector from the current 3D game state, including raycast data.
     * @param {Object} gameState - The raw game state object.
     * @returns {Float32Array} The normalized state vector.
     */
    createStateVector(gameState) {
        const agent = this.isPlayerTwo ? gameState.ai : gameState.player;
        const opponent = this.isPlayerTwo ? gameState.player : gameState.ai;
        const target = this.isPlayerTwo ? gameState.aiTarget : gameState.playerTarget;

        const worldHalfSize = this.game.constructor.WORLD_SIZE / 2;
        const worldSize = this.game.constructor.WORLD_SIZE;
        
        const agentX_norm = (agent.x || 0) / worldHalfSize;
        const agentZ_norm = (agent.z || 0) / worldHalfSize;
        const agentRot_norm = (agent.rotY || 0) / Math.PI;
        const targetX_norm = (target.x || 0) / worldHalfSize;
        const targetZ_norm = (target.z || 0) / worldHalfSize;
        const vecX = (target.x || 0) - (agent.x || 0);
        const vecZ = (target.z || 0) - (agent.z || 0);
        const dist = Math.sqrt(vecX*vecX + vecZ*vecZ);
        const vecX_norm = vecX / worldSize;
        const vecZ_norm = vecZ / worldSize;
        const dist_norm = dist / worldSize;

        const agent3D = this.isPlayerTwo ? this.game.ai : this.game.player;
        const rayDetections = this.game.getRaycastDetections(agent3D);
        const rayLeft_norm = rayDetections.left;
        const rayCenter_norm = rayDetections.center;
        const rayRight_norm = rayDetections.right;

        const oppVecX = (opponent.x || 0) - (agent.x || 0);
        const oppVecZ = (opponent.z || 0) - (agent.z || 0);
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

        if (!isFiniteVector(clampedStateVec) || clampedStateVec.length !== this.worldModel.stateDim) {
            logger.error(`LearningAI.createStateVector: Generated state vector is invalid. Returning zeros.`, {clampedStateVec});
            return vecZeros(this.worldModel.stateDim);
        }
        return clampedStateVec;
    }

    /**
     * Computes softmax probabilities for exploration.
     * @param {Float32Array} actionProbs - Raw action probabilities from the model.
     * @returns {Float32Array} Softmax probabilities.
     */
    softmax(actionProbs) {
        if (!isFiniteVector(actionProbs)) return vecZeros(actionProbs.length);
        const maxProb = Math.max(...actionProbs);
        const exps = actionProbs.map(p => Math.exp((p - maxProb) / this.temperature));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return new Float32Array(exps.map(e => e / sumExps));
    }

    /**
     * Makes a decision for the AI action using softmax-based exploration.
     * @param {Object} gameState - The current game state.
     * @returns {Promise<{action: number[], activations: Float32Array[], corrupted: boolean, chosenActionIndex: number}>}
     */
    async makeDecision(gameState) {
        if (this.actionQueue.length >= this.aiResponseTime) {
            return this.actionQueue.shift();
        }

        const actions = [
            { name: 'MOVE_FORWARD', vec: [1, 0, 0, 0] }, 
            { name: 'TURN_LEFT', vec: [0, 1, 0, 0] }, 
            { name: 'TURN_RIGHT', vec: [0, 0, 1, 0] },
            { name: 'IDLE', vec: [0, 0, 0, 1] }
        ];
        let actionIndex = 3; // Default to IDLE

        const stateVec = this.createStateVector(gameState);
        this.lastStateVec = stateVec;

        const { actionProbs, stateValue, activations, corrupted } = await this.worldModel.predict(stateVec);
        this.lastActivations = activations;

        if (corrupted) {
            logger.warn(`LearningAI (${this.isPlayerTwo ? 'AI' : 'Opponent'}): World model predicted a corrupted state. Defaulting to IDLE.`);
            this.lastActionIndex = 3;
            const decision = { action: actions[3].vec, activations: this.lastActivations, corrupted: true, chosenActionIndex: 3 };
            this.actionQueue.push(decision);
            return this.actionQueue.shift() || decision;
        }

        this.avgStateValue = Number.isFinite(stateValue) ? stateValue : 0;

        // Softmax-based exploration
        if (Math.random() < this.epsilon) {
            const softmaxProbs = this.softmax(actionProbs);
            let rand = Math.random();
            let cumulative = 0;
            for (let i = 0; i < softmaxProbs.length; i++) {
                cumulative += softmaxProbs[i];
                if (rand <= cumulative) {
                    actionIndex = i;
                    break;
                }
            }
        } else {
            if (actionProbs && isFiniteVector(actionProbs)) {
                actionIndex = actionProbs.indexOf(Math.max(...actionProbs));
            } else {
                logger.warn(`LearningAI (${this.isPlayerTwo ? 'AI' : 'Opponent'}): Received invalid action probabilities, defaulting to IDLE.`);
                actionIndex = 3;
            }
        }
        
        this.lastActionIndex = actionIndex;
        const decision = { action: actions[actionIndex].vec, activations: this.lastActivations, corrupted: false, chosenActionIndex: actionIndex };
        this.actionQueue.push(decision);

        return this.actionQueue.shift() || decision;
    }

    /**
     * Performs an Actor-Critic learning update step.
     * @param {Object} preGameState - The game state before the action was taken.
     * @param {number} actionIndex - The index of the action that was taken.
     * @param {number} reward - The reward received from the environment.
     * @param {Object} newGameState - The new state of the game.
     * @param {boolean} isDone - True if the episode is finished.
     * @returns {Promise<void>}
     */
    async learn(preGameState, actionIndex, reward, newGameState, isDone) {
        if (preGameState) {
            this.lastStateVec = this.createStateVector(preGameState);
        }
        const nextStateVec = this.createStateVector(newGameState);

        const curiosityBonus = clamp(this.worldModel.predictionError, 0, 1) * 0.01;
        let totalReward = reward + curiosityBonus;

        if (actionIndex === 3) {
            totalReward -= 0.01;
        }
        
        await this.worldModel.learn(this.lastStateVec, actionIndex, totalReward, nextStateVec, isDone, this.learningRate);
        
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
            this.temperature = clamp(this.temperature * 0.999, 0.5, 2.0); // Decay temperature
        }

        const uncertaintyFactor = 1 + clamp(this.worldModel.predictionError, 0, 5) * 0.05;
        this.epsilon = clamp(this.epsilon * uncertaintyFactor, this.epsilonMin, 1.0);
        this.epsilon = Number.isFinite(this.epsilon) ? this.epsilon : this.epsilonMin;
    }

    /**
     * Resets the AI's internal state and learning parameters.
     */
    reset() {
        this.worldModel.resetRecurrentState();
        this.epsilon = 1.0;
        this.temperature = 1.0;
        this.avgStateValue = 0;
        this.actionQueue = [];
        this.learningRate = 0.01;
        logger.info(`LearningAI for ${this.isPlayerTwo ? 'AI' : 'Opponent'} has been reset.`);
    }
}

/**
 * Monitors a LearningAI's performance and internal metrics to adapt its learning rate and exploration.
 */
export class StrategicAI {
    constructor(learningAI) {
        this.learningAI = learningAI;
        this.rewardHistory = [];
        this.HISTORY_SIZE = 200;
        this.epsilonModulationRate = 0.005;
        this.learningRateModulationRate = 0.005;
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
        const predError = Number.isFinite(this.learningAI.worldModel.predictionError) ? this.learningAI.worldModel.predictionError : 0;
        const gestaltUnity = Number.isFinite(this.learningAI.worldModel.qualiaSheaf.gestaltUnity) ? this.learningAI.worldModel.qualiaSheaf.gestaltUnity : 0.5;
        const h1Dimension = Number.isFinite(this.learningAI.worldModel.qualiaSheaf.h1Dimension) ? this.learningAI.worldModel.qualiaSheaf.h1Dimension : 1.0;

        if (avgReward < 0.05 && predError > 1.0) {
            this.learningAI.learningRate = clamp(this.learningAI.learningRate * (1 + this.learningRateModulationRate), 0.001, 0.05);
            this.learningAI.epsilon = clamp(this.learningAI.epsilon * (1 + this.epsilonModulationRate), this.learningAI.epsilonMin, 1.0);
        } else if (avgReward > 0.2 && predError < 0.5) {
            this.learningAI.learningRate = clamp(this.learningAI.learningRate * (1 - this.learningRateModulationRate), 0.001, 0.05);
        }

        const explorationModifier = (1 - gestaltUnity) + (h1Dimension * 0.1);
        this.learningAI.epsilon = clamp(this.learningAI.epsilon * (1 + (explorationModifier - 0.5) * 0.01), this.learningAI.epsilonMin, 1.0);

        this.learningAI.learningRate = Number.isFinite(this.learningAI.learningRate) ? this.learningAI.learningRate : 0.01;
        this.learningAI.epsilon = Number.isFinite(this.learningAI.epsilon) ? this.learningAI.epsilon : this.learningAI.epsilonMin;
    }
}
// --- END OF FILE ai-agents.js ---
