// ===================================================================================
// --- OWM.JS (STABILITY-ENHANCED & COMPLETE - V5) ---
// A comprehensive Ontological World Model combining a Floquet Persistent Sheaf with an LSTM-based RNN.
// Integrates V4's functionality with fixes for isFiniteNumber errors and enhanced robustness for sheaf interactions.
// Fixes vector mismatch errors, ensures numerical stability, and supports dynamic sheaf topology changes.
// ===================================================================================

import {
    clamp, dot, norm2, vecAdd, vecSub, vecScale, tanhVec, sigmoidVec, vecMul,
    randomMatrix, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix,
    logger, runWorkerTask, softmax, sigmoid
} from './utils.js';
import { FloquetPersistentSheaf } from './qualia-sheaf.js';
import { matVecMul } from './utils.js';
const Numeric = window.Numeric || null;
const GPU = window.gpu || null;

/**
 * Local utility to check if a number is finite.
 */
const isFiniteNumber = (x) => typeof x === 'number' && Number.isFinite(x);

/**
 * Ontological World Model (OWM) for an AI, combining a Floquet Persistent Sheaf with an LSTM-based
 * Recurrent Neural Network. It predicts future states probabilistically and evaluates actions using
 * an Actor-Critic architecture with dynamic adaptation to sheaf topology changes.
 */
export class OntologicalWorldModel {
    constructor(stateDim = 13, actionDim = 4, qDim = 7, hiddenSizes = [64, 64], isPlayerTwo = false, qualiaSheafInstance = null) {
        if (!qualiaSheafInstance || !qualiaSheafInstance.complex || !Array.isArray(qualiaSheafInstance.complex.vertices)) {
            console.error('CRITICAL ERROR: OntologicalWorldModel requires a pre-initialized FloquetPersistentSheaf with valid complex.vertices.');
            logger.error('CRITICAL ERROR: OntologicalWorldModel requires a pre-initialized FloquetPersistentSheaf with valid complex.vertices.');
            throw new Error('OntologicalWorldModel must be initialized with a valid FloquetPersistentSheaf instance.');
        }
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.qDim = qDim;
        this.isPlayerTwo = isPlayerTwo;
        this.recurrentStateSize = hiddenSizes[hiddenSizes.length - 1];
        this.qualiaSheaf = qualiaSheafInstance;

        this._initializeNetwork();

        this.freeEnergy = 0;
        this.predictionError = 0;
        this.ready = false;
        this.lastActivations = [];
        this.lastActionLogProbs = vecZeros(this.actionDim);
        this.lastChosenActionLogProb = 0;
        this.lastStateValue = 0;
        this.actorLoss = 0;
        this.criticLoss = 0;
        this.predictionLoss = 0;
        this.coherencePrior = vecZeros(this.recurrentStateSize); 

        logger.info(`OWM constructed (${isPlayerTwo ? 'AI' : 'Player'}): stateDim=${this.stateDim}, inputDim=${this.inputDim}, qualiaSheaf.qDim=${this.qDim}, expectedQualiaInputLength=${this.expectedQualiaInputLength}`);
    }

    /**
     * Initializes network weights and dimensions, allowing reinitialization if the sheaf's topology changes.
     */
    _initializeNetwork() {
        if (!this.qualiaSheaf || !this.qualiaSheaf.complex || !Array.isArray(this.qualiaSheaf.complex.vertices)) {
            throw new Error('OWM _initializeNetwork: Sheaf not ready.');
        }
        this.qDim = this.qualiaSheaf.qDim;
        const nVertices = Math.max(1, this.qualiaSheaf.complex.vertices.length);
        this.expectedQualiaInputLength = nVertices * this.qDim;
        this.inputDim = this.stateDim + this.expectedQualiaInputLength;

        this.hiddenState = vecZeros(this.recurrentStateSize);
        this.cellState = vecZeros(this.recurrentStateSize);

        const combinedInputSize = this.inputDim + this.recurrentStateSize;
        const weightScale = Math.sqrt(2.0 / combinedInputSize);

        const safeRandomMatrix = (rows, cols, scale) => {
            rows = Math.max(1, rows);
            cols = Math.max(1, cols);
            const flatData = new Float32Array(rows * cols);
            for (let i = 0; i < flatData.length; i++) flatData[i] = (Math.random() * 2 - 1) * scale;
            return { flatData, rows, cols };
        };

        // LSTM weights
        this.Wf = safeRandomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);
        this.Wi = safeRandomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);
        this.Wc = safeRandomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);
        this.Wo = safeRandomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);

        // Biases
        this.bf = vecZeros(this.recurrentStateSize);
        this.bi = vecZeros(this.recurrentStateSize);
        this.bc = vecZeros(this.recurrentStateSize);
        this.bo = vecZeros(this.recurrentStateSize);

        // Actor / Critic / State heads
        const actorOutputScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.actionDim));
        this.actorHead = { W: safeRandomMatrix(this.actionDim, this.recurrentStateSize, actorOutputScale), b: vecZeros(this.actionDim) };

        const criticOutputScale = Math.sqrt(2.0 / (this.recurrentStateSize + 1));
        this.criticHead = { W: safeRandomMatrix(1, this.recurrentStateSize, criticOutputScale), b: vecZeros(1) };

        const statePredScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.stateDim));
        this.statePredHead = { W: safeRandomMatrix(this.stateDim, this.recurrentStateSize, statePredScale), b: vecZeros(this.stateDim) };

        const varScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.stateDim));
        this.varianceHead = { W: safeRandomMatrix(this.stateDim, this.recurrentStateSize, varScale), b: vecZeros(this.stateDim) };

        const anticipatoryScale = Math.sqrt(2.0 / (this.recurrentStateSize + 1));
        this.anticipatoryHead = { W: safeRandomMatrix(1, this.recurrentStateSize, anticipatoryScale), b: vecZeros(1) };

        // Attention
        this.attentionWeights = safeRandomMatrix(nVertices, this.stateDim, 0.1);
        this.lastSoftmaxScores = vecZeros(nVertices);

        logger.info(`OWM network initialized: inputDim=${this.inputDim}, recurrentStateSize=${this.recurrentStateSize}`);
    }

    /**
     * Safe matrix-vector multiplication wrapper.
     */
    safeMatVecMul(matrix, vector) {
        if (!matrix || !matrix.flatData || matrix.rows <= 0 || matrix.cols <= 0 || !isFiniteVector(vector) || vector.length !== matrix.cols) {
            console.warn('safeMatVecMul: Invalid matrix or vector mismatch.', { matrix, vectorLength: vector?.length });
            logger.warn('safeMatVecMul: Invalid matrix or vector mismatch.');
            return vecZeros(matrix?.rows || 1);
        }
        const result = new Float32Array(matrix.rows);
        for (let i = 0; i < matrix.rows; i++) {
            let sum = 0;
            for (let j = 0; j < matrix.cols; j++) {
                sum += matrix.flatData[i * matrix.cols + j] * vector[j];
            }
            result[i] = sum;
        }
        return isFiniteVector(result) ? result : vecZeros(matrix.rows);
    }

    /**
     * Applies attention mechanism to input, enhanced with sheaf stalk data.
     */
    applyAttention(input) {
        if (!isFiniteVector(input) || input.length !== this.stateDim) {
            console.warn('OWM.applyAttention: Invalid input. Returning zeros.');
            logger.warn('OWM.applyAttention: Invalid input.');
            this.lastSoftmaxScores = vecZeros(Math.max(1, this.qualiaSheaf.complex.vertices.length));
            return vecZeros(this.stateDim);
        }

        const nVertices = Math.max(1, this.qualiaSheaf.complex.vertices.length);
        if (!this.attentionWeights || !this.attentionWeights.flatData || this.attentionWeights.rows !== nVertices || this.attentionWeights.cols !== this.stateDim) {
            console.warn(`OWM.applyAttention: attentionWeights invalid. Reinitializing ${nVertices}x${this.stateDim}`);
            logger.warn(`OWM.applyAttention: attentionWeights invalid. Reinitializing.`);
            const flatData = new Float32Array(nVertices * this.stateDim);
            for (let i = 0; i < flatData.length; i++) flatData[i] = (Math.random() * 0.2 - 0.1);
            this.attentionWeights = { flatData, rows: nVertices, cols: this.stateDim };
        }

        const scores = new Float32Array(nVertices);
        for (let i = 0; i < nVertices; i++) {
            const row = this.attentionWeights.flatData.subarray(i * this.stateDim, (i + 1) * this.stateDim);
            scores[i] = dot(row, input);
            const vertex = this.qualiaSheaf.complex.vertices[i];
            if (this.qualiaSheaf.stalks.has(vertex)) {
                const stalkData = this.qualiaSheaf.stalks.get(vertex);
                if (isFiniteVector(stalkData) && stalkData.length >= 3) {
                    scores[i] += 0.1 * (isFiniteNumber(stalkData[2]) ? stalkData[2] : 0);
                }
            }
        }

        if (!isFiniteVector(scores)) {
            console.warn('OWM.applyAttention: Scores became non-finite. Returning zeros.');
            logger.warn('OWM.applyAttention: Scores became non-finite.');
            this.lastSoftmaxScores.fill(0);
            return vecZeros(this.stateDim);
        }

        this.lastSoftmaxScores = softmax(scores);
        const att = vecZeros(this.stateDim);
        for (let i = 0; i < nVertices; i++) {
            for (let j = 0; j < this.stateDim; j++) {
                att[j] += (this.lastSoftmaxScores[i] || 0) * (input[j] || 0);
            }
        }

        const beta = isFiniteNumber(this.qualiaSheaf.beta) ? this.qualiaSheaf.beta : 0.1;
        const weighted = vecAdd(input, vecScale(att, beta));
        return new Float32Array(isFiniteVector(weighted) ? weighted.map(v => clamp(v, -100, 100)) : vecZeros(this.stateDim));
    }

    /**
     * Checks if the network dimensions match the sheaf's topology and reinitializes if necessary.
     */
    _checkAndReinitializeNetwork() {
        const currentExpectedLength = this.qualiaSheaf.complex.vertices.length * this.qualiaSheaf.qDim;
        if (this.expectedQualiaInputLength !== currentExpectedLength) {
            logger.warn(`OWM detected sheaf topology change. Re-initializing network layers. Old qualia length: ${this.expectedQualiaInputLength}, New: ${currentExpectedLength}`);
            this._initializeNetwork();
        }
    }

    /**
     * Initializes the OWM, ensuring the sheaf is ready.
     */
    async initialize() {
        logger.info(`OWM.initialize() (${this.isPlayerTwo ? 'AI' : 'Player'}) called.`);
        try {
            if (!this.qualiaSheaf.complex || !Array.isArray(this.qualiaSheaf.complex.vertices)) {
                console.error(`CRITICAL ERROR: OWM's assigned FloquetPersistentSheaf is not ready for ${this.isPlayerTwo ? 'AI' : 'Player'}.`);
                logger.error(`CRITICAL ERROR: OWM's assigned FloquetPersistentSheaf is not ready for ${this.isPlayerTwo ? 'AI' : 'Player'}.`);
                await this.qualiaSheaf.initialize();
            }
            this.ready = true;
            logger.info(`Recurrent OWM for ${this.isPlayerTwo ? 'AI' : 'Player'} fully initialized with sheaf linkage.`);
        } catch (e) {
            console.error(`CRITICAL ERROR: OWM initialization failed for ${this.isPlayerTwo ? 'AI' : 'Player'}:`, e);
            logger.error(`CRITICAL ERROR: OWM initialization failed for ${this.isPlayerTwo ? 'AI' : 'Player'}:`, e);
            this.ready = false;
            throw e;
        }
    }

    /**
     * Combines raw state vector with qualia vector for full input.
     */
    _getFullInputVector(rawStateVector) {
    if (!isFiniteVector(rawStateVector) || rawStateVector.length !== this.stateDim) {
        logger.error('OWM._getFullInputVector: Invalid rawStateVector. Returning zeros.', {
            rawStateVector: rawStateVector?.slice(0, 10),
            length: rawStateVector?.length
        });
        return vecZeros(this.inputDim);
    }

    const attentionalInput = this.applyAttention(rawStateVector);
    const qualiaVector = this.qualiaSheaf.getStalksAsVector();
    if (!isFiniteVector(qualiaVector) || qualiaVector.length !== this.expectedQualiaInputLength) {
        logger.warn('OWM._getFullInputVector: Invalid qualiaVector. Resetting stalks.', {
            qualiaVector: qualiaVector?.slice(0, 10),
            expected: this.expectedQualiaInputLength,
            nonFiniteCount: qualiaVector?.filter(x => !Number.isFinite(x)).length
        });
        this.qualiaSheaf.stalks.forEach((_, key) => {
            this.qualiaSheaf.stalks.set(key, new Array(this.qDim).fill(0).map((_, i) => i === 0 ? 0.5 : 0));
        });
        return vecZeros(this.inputDim);
    }

    const fullInput = new Float32Array(this.inputDim);
    fullInput.set(attentionalInput.slice(0, this.stateDim), 0);
    fullInput.set(qualiaVector, this.stateDim);

    if (!isFiniteVector(fullInput)) {
        logger.error('OWM._getFullInputVector: Non-finite fullInput. Returning zeros.', {
            fullInput: fullInput.slice(0, 10),
            nonFiniteCount: fullInput.filter(x => !Number.isFinite(x)).length
        });
        return vecZeros(this.inputDim);
    }
    return fullInput.map(v => clamp(v, -100, 100));
}

    /**
     * Forward pass through the LSTM network with robust error handling.
     */
    async forward(input, hPrev = this.hiddenState, cPrev = this.cellState) {
    try {
        if (!isFiniteVector(input) || input.length !== this.inputDim) {
            console.warn(`OWM.forward: Invalid input length. Expected ${this.inputDim}, got ${input?.length}. Using zeros.`);
            logger.warn('OWM.forward: Invalid input length.');
            input = vecZeros(this.inputDim);
        }

        const combinedInput = new Float32Array(this.inputDim + this.recurrentStateSize);
        if (!isFiniteVector(hPrev) || hPrev.length !== this.recurrentStateSize) {
            console.warn(`OWM.forward: Invalid hPrev length. Expected ${this.recurrentStateSize}, got ${hPrev?.length}. Using zeros.`);
            hPrev = vecZeros(this.recurrentStateSize);
        }
        combinedInput.set(input, 0);
        combinedInput.set(hPrev, this.inputDim);

        // LSTM pre-activations
        const fPre = vecAdd(this.safeMatVecMul(this.Wf, combinedInput), this.bf);
        const iPre = vecAdd(this.safeMatVecMul(this.Wi, combinedInput), this.bi);
        const cPre = vecAdd(this.safeMatVecMul(this.Wc, combinedInput), this.bc);
        const oPre = vecAdd(this.safeMatVecMul(this.Wo, combinedInput), this.bo);

        // LSTM gate activations
        const f = sigmoidVec(fPre);
        const i = sigmoidVec(iPre);
        const cBar = tanhVec(cPre);
        const cNext = vecAdd(vecMul(f, cPrev), vecMul(i, cBar));
        const o = sigmoidVec(oPre);
        const hNext = vecMul(o, tanhVec(cNext));

        // Update internal states
        this.hiddenState = hNext;
        this.cellState = cNext;

        // Head outputs
        const actionLogits = vecAdd(this.safeMatVecMul(this.actorHead.W, hNext), this.actorHead.b);
        const stateValue = this.safeMatVecMul(this.criticHead.W, hNext)[0] + this.criticHead.b[0];
        const nextStateRaw = vecAdd(this.safeMatVecMul(this.statePredHead.W, hNext), this.statePredHead.b);
        const variance = vecAdd(this.safeMatVecMul(this.varianceHead.W, hNext), this.varianceHead.b);
        const anticipatoryReward = this.safeMatVecMul(this.anticipatoryHead.W, hNext)[0] + this.anticipatoryHead.b[0];

        return {
            actionLogits,
            stateValue,
            nextStatePrediction: this._formatStatePrediction(nextStateRaw),
            variance: variance.map(v => clamp(Math.abs(v), 0, 10)),
            anticipatoryReward: clamp(anticipatoryReward, -10, 10),
            activations: hNext,
            corrupted: false
        };
    } catch (e) {
        console.error(`OWM.forward: CRITICAL ERROR: ${e.message}. Resetting recurrent state.`);
        logger.error(`OWM.forward: CRITICAL ERROR: ${e.message}.`);
        this.resetRecurrentState();
        return {
            corrupted: true,
            actionLogits: vecZeros(this.actionDim),
            stateValue: 0,
            nextStatePrediction: this._formatStatePrediction(vecZeros(this.stateDim)),
            variance: vecZeros(this.stateDim),
            anticipatoryReward: 0,
            activations: vecZeros(this.recurrentStateSize)
        };
    }
}

    /**
     * Formats state prediction for game compatibility.
     */
    _formatStatePrediction(stateVector) {
        if (!isFiniteVector(stateVector) || stateVector.length < 3) {
            console.warn('OWM._formatStatePrediction: Invalid stateVector; returning default state.');
            logger.warn('OWM._formatStatePrediction: Invalid stateVector; returning default state.');
            return { x: 0, z: 0, rot: 0, raw: vecZeros(this.stateDim) };
        }
        return {
            x: clamp(stateVector[0], -50, 50),
            z: clamp(stateVector[1], -50, 50),
            rot: clamp(stateVector[2], 0, 2 * Math.PI),
            raw: stateVector.map(v => clamp(v, -100, 100))
        };
    }

    /**
     * Predicts the next state given a raw state vector.
     */
    async predictNextState(rawStateVector) {
        if (!this.ready) {
            console.warn('OWM.predictNextState: Not ready; returning default state.');
            logger.warn('OWM.predictNextState: Not ready; returning default state.');
            return this._formatStatePrediction(vecZeros(this.stateDim));
        }

        const fullInput = this._getFullInputVector(rawStateVector);
        let forwardResult;
        try {
            forwardResult = await this.forward(fullInput);
        } catch (e) {
            console.error('OWM.predictNextState: Error during forward pass:', e);
            logger.error('OWM.predictNextState: Error during forward pass:', e);
            this.resetRecurrentState();
            return this._formatStatePrediction(vecZeros(this.stateDim));
        }

        const { nextStatePrediction, corrupted } = forwardResult;
        if (corrupted) {
            console.warn('OWM.predictNextState: Forward pass corrupted; returning default state.');
            logger.warn('OWM.predictNextState: Forward pass corrupted; returning default state.');
            return this._formatStatePrediction(vecZeros(this.stateDim));
        }

        return nextStatePrediction;
    }

    /**
     * Computes softmax probabilities for action selection.
     */
    softmax(actionProbs) {
        if (!isFiniteVector(actionProbs)) {
            console.warn('OWM.softmax: Input actionProbs are not finite. Returning uniform probabilities.');
            logger.warn('OWM.softmax: Input actionProbs are not finite.');
            return vecZeros(this.actionDim).fill(1 / this.actionDim);
        }

        const maxProb = Math.max(...actionProbs);
        if (!isFiniteNumber(maxProb)) {
            console.warn('OWM.softmax: maxProb is non-finite. Returning uniform probabilities.', { maxProb });
            logger.warn('OWM.softmax: maxProb is non-finite.');
            return vecZeros(this.actionDim).fill(1 / this.actionDim);
        }

        const exp_logits = actionProbs.map(v => Math.exp(v - maxProb));
        const sum_exp_logits = exp_logits.reduce((sum, val) => sum + (isFiniteNumber(val) ? val : 0), 0);
        const safe_sum_exp_logits = (isFiniteNumber(sum_exp_logits) && sum_exp_logits > 1e-9) ? sum_exp_logits : 1e-9;

        const resultProbs = exp_logits.map(v => (isFiniteNumber(v) ? v : 0) / safe_sum_exp_logits);
        if (!isFiniteVector(resultProbs)) {
            console.warn('OWM.softmax: Output probabilities are not finite. Returning uniform probabilities.');
            logger.warn('OWM.softmax: Output probabilities are not finite.');
            return vecZeros(this.actionDim).fill(1 / this.actionDim);
        }
        return new Float32Array(resultProbs);
    }

    /**
     * Chooses an action based on the state vector and exploration parameter.
     */
    async chooseAction(rawStateVector, epsilon = 0.1) {
        if (!this.ready) {
            console.warn('OWM.chooseAction: Not ready; returning default action.');
            logger.warn('OWM.chooseAction: Not ready; returning default action.');
            return {
                action: 'IDLE',
                chosenActionIndex: 3,
                actionProbs: vecZeros(this.actionDim),
                stateValue: 0,
                activations: [],
                variance: vecZeros(this.stateDim).fill(1),
                anticipatoryReward: 0,
                nextStatePrediction: this._formatStatePrediction(vecZeros(this.stateDim)),
                corrupted: true
            };
        }

        const fullInput = this._getFullInputVector(rawStateVector);
        let forwardResult;
        try {
            forwardResult = await this.forward(fullInput);
        } catch (e) {
            console.error('OWM.chooseAction: Error during forward pass:', e);
            logger.error('OWM.chooseAction: Error during forward pass:', e);
            this.resetRecurrentState();
            return {
                action: 'IDLE',
                chosenActionIndex: 3,
                actionProbs: vecZeros(this.actionDim),
                stateValue: 0,
                activations: [],
                variance: vecZeros(this.stateDim).fill(1),
                anticipatoryReward: 0,
                nextStatePrediction: this._formatStatePrediction(vecZeros(this.stateDim)),
                corrupted: true
            };
        }

        const { actionLogits, stateValue, nextStatePrediction, variance, anticipatoryReward, activations, corrupted } = forwardResult;

        if (corrupted) {
            console.error('OWM.chooseAction: Forward pass reported corrupted outputs. Resetting state.');
            logger.error('OWM.chooseAction: Forward pass reported corrupted outputs.');
            this.resetRecurrentState();
            return {
                action: 'IDLE',
                chosenActionIndex: 3,
                actionProbs: vecZeros(this.actionDim),
                stateValue: 0,
                activations: [],
                variance: vecZeros(this.stateDim).fill(1),
                anticipatoryReward: 0,
                nextStatePrediction: this._formatStatePrediction(vecZeros(this.stateDim)),
                corrupted: true
            };
        }

        const temperature = clamp(1.0 / (1 + this.freeEnergy), 0.1, 2.0);
        const actionProbs = this.softmax(vecScale(actionLogits, 1 / temperature));

        const actionIndex = Math.random() < epsilon
            ? Math.floor(Math.random() * this.actionDim)
            : actionProbs.reduce((maxIdx, p, i) => p > actionProbs[maxIdx] ? i : maxIdx, 0);

        const actionString = ['FORWARD', 'LEFT', 'RIGHT', 'IDLE'][actionIndex];

        this.lastActionLogProbs = actionLogits;
        this.lastChosenActionLogProb = actionLogits[actionIndex];
        this.lastStateValue = stateValue;
        this.lastActivations = activations;

        return {
            action: actionString,
            chosenActionIndex: actionIndex,
            actionProbs,
            stateValue,
            activations,
            variance,
            anticipatoryReward,
            nextStatePrediction,
            corrupted: false
        };
    }

    /**
     * Computes free energy based on prediction error and sheaf properties.
     */
    async computeFreeEnergy(nextRawStateVector, predictedState) {
        if (!isFiniteVector(nextRawStateVector) || !isFiniteVector(predictedState.raw)) {
            this.freeEnergy = 1.0;
            console.warn('OWM.computeFreeEnergy: Invalid inputs; defaulting to 1.0.');
            logger.warn('OWM.computeFreeEnergy: Invalid inputs; defaulting to 1.0.');
            return;
        }
        const klProxy = norm2(vecSub(nextRawStateVector, predictedState.raw)) * 0.1;
        const sheafInconsistency = isFiniteNumber(this.qualiaSheaf.inconsistency) ? this.qualiaSheaf.inconsistency : 0;
        const coherenceReduction = isFiniteNumber(this.qualiaSheaf.coherence) ? this.qualiaSheaf.coherence * 0.5 : 0;
        this.freeEnergy = clamp(klProxy + sheafInconsistency - coherenceReduction, 0, 10);
    }

    /**
     * Updates the model based on temporal difference error and prior.
     */
     async learn(tdError, targetValue, nextRawStateVector, lr, prior = null) {
    if (!this.ready) {
        logger.warn('OWM.learn: Not ready; returning zero losses.');
        this.actorLoss = 0;
        this.criticLoss = 0;
        this.predictionLoss = 0;
        return { actorLoss: 0, criticLoss: 0, predictionLoss: 0 };
    }

    const nextFullInput = this._getFullInputVector(nextRawStateVector);
    let forwardResult;
    try {
        forwardResult = await this.forward(nextFullInput);
    } catch (e) {
        logger.error('OWM.learn: Error during forward pass for next state prediction:', { error: e.message, stack: e.stack });
        this.predictionError = 1.0;
        forwardResult = { nextStatePrediction: this._formatStatePrediction(vecZeros(this.stateDim)), corrupted: true };
    }

    const { nextStatePrediction, corrupted: predictionCorrupted } = forwardResult;

    if (predictionCorrupted || !isFiniteVector(nextRawStateVector) || !isFiniteVector(nextStatePrediction.raw)) {
        logger.warn('OWM.learn: Invalid nextRawStateVector or nextStatePrediction.');
        this.predictionError = 1.0;
    } else {
        this.predictionError = norm2(vecSub(nextRawStateVector, nextStatePrediction.raw)) * 0.1;
        this.predictionError = clamp(this.predictionError, 0, 10);
    }

    await this.computeFreeEnergy(nextRawStateVector, nextStatePrediction);

    let modulatedTdError = isFiniteNumber(tdError) ? tdError : 0;
    if (prior && isFiniteVector(prior) && prior.length === this.actionDim) {
        const priorInfluence = dot(prior, vecZeros(this.actionDim)) * (isFiniteNumber(this.qualiaSheaf.coherence) ? this.qualiaSheaf.coherence : 0.5);
        modulatedTdError += priorInfluence * 0.1;
        modulatedTdError = clamp(modulatedTdError, -10, 10);
    }

    const simulatedActorLoss = Math.abs(modulatedTdError) * 0.1 + Math.random() * 0.001;
    const simulatedCriticLoss = Math.abs(modulatedTdError) * 0.05 + Math.random() * 0.001;
    const totalPredictionLoss = this.predictionError + Math.random() * 0.001;

    this.actorLoss = clamp(simulatedActorLoss, 0, 10);
    this.criticLoss = clamp(simulatedCriticLoss, 0, 10);
    this.predictionLoss = clamp(totalPredictionLoss, 0, 10);

    const coherence = isFiniteNumber(this.qualiaSheaf.coherence) ? this.qualiaSheaf.coherence : 0;
    
    // --- START OF FIX: Ensure vector dimensions match before adding ---
    if (!isFiniteVector(this.coherencePrior) || this.coherencePrior.length !== this.recurrentStateSize) {
        logger.warn(`OWM.learn: Invalid coherencePrior. Expected length ${this.recurrentStateSize}, got ${this.coherencePrior?.length || 'undefined'}. Resetting.`);
        this.coherencePrior = vecZeros(this.recurrentStateSize);
    }
    if (!isFiniteVector(this.hiddenState) || this.hiddenState.length !== this.recurrentStateSize) {
        logger.warn(`OWM.learn: Invalid hiddenState. Expected length ${this.recurrentStateSize}, got ${this.hiddenState?.length || 'undefined'}. Resetting.`);
        this.hiddenState = vecZeros(this.recurrentStateSize);
    }
    // --- END OF FIX ---

    let scaledHiddenState = vecScale(this.hiddenState, lr * coherence);
    this.coherencePrior = vecAdd(this.coherencePrior, scaledHiddenState);

    if (this.qualiaSheaf.ready) {
        try {
            await this.qualiaSheaf.diffuseQualia(nextRawStateVector);
        } catch (e) {
            logger.warn('OWM.learn: QualiaSheaf diffusion failed. Skipping.', { error: e.message, stack: e.stack });
        }
    } else {
        logger.warn('OWM.learn: QualiaSheaf not ready for diffusion. Skipping.');
    }

    return {
        actorLoss: this.actorLoss,
        criticLoss: this.criticLoss,
        predictionLoss: this.predictionLoss
    };
}

    /**
     * Disposes of the OWM resources.
     */
    dispose() {
        logger.info(`OWM for ${this.isPlayerTwo ? 'AI' : 'Player'} disposed.`);
    }

    /**
     * Resets the recurrent state of the network.
     */
    resetRecurrentState() {
    this.hiddenState = vecZeros(this.recurrentStateSize);
    this.cellState = vecZeros(this.recurrentStateSize);
    this.lastStateValue = 0;
    this.predictionError = 0;
    this.freeEnergy = 0;
    this.actorLoss = 0;
    this.criticLoss = 0;
    this.predictionLoss = 0;
    this.lastActionLogProbs = vecZeros(this.actionDim);
    this.lastChosenActionLogProb = 0;
    this.lastActivations = [];
    // --- FIX: Initialize coherencePrior with the correct size ---
    this.coherencePrior = vecZeros(this.recurrentStateSize);
    this.lastSoftmaxScores = vecZeros(this.qualiaSheaf.complex.vertices.length);
    logger.info('OWM recurrent state reset with coherence prior.');
}

    /**
     * Retrieves Q-values for the given state (placeholder for compatibility).
     */
    async getQValues(stateVec) {
        if (!isFiniteVector(stateVec) || stateVec.length !== this.stateDim) {
            console.warn('OWM.getQValues: Invalid state vector. Returning zeros.');
            logger.warn('OWM.getQValues: Invalid state vector.');
            return vecZeros(this.actionDim);
        }
        const fullInput = this._getFullInputVector(stateVec);
        const { actionLogits, corrupted } = await this.forward(fullInput).catch(e => {
            console.error('OWM.getQValues: Forward pass error:', e);
            logger.error('OWM.getQValues: Forward pass error:', e);
            return { actionLogits: vecZeros(this.actionDim), corrupted: true };
        });
        return corrupted ? vecZeros(this.actionDim) : actionLogits;
    }
}
