// --- START OF FILE owm.js ---

import {
    clamp, dot, norm2, vecAdd, vecSub, vecScale, tanhVec, sigmoidVec, vecMul,
    randomMatrix, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix,
    logger, runWorkerTask, softmax
} from './utils.js';
import { FloquetPersistentSheaf } from './qualia-sheaf.js';

const Numeric = window.Numeric || null;
const GPU = window.gpu || null;

/**
 * Represents an Ontological World Model (OWM) for an AI, combining a Floquet Persistent Sheaf with an LSTM-based
 * Recurrent Neural Network. It predicts future states probabilistically and evaluates actions using an Actor-Critic
 * architecture with dynamic adaptation to sheaf topology changes.
 */
export class OntologicalWorldModel {
    constructor(stateDim = 13, actionDim = 4, qDim = 7, hiddenSizes = [64, 64], isPlayerTwo = false, qualiaSheafInstance = null) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.isPlayerTwo = isPlayerTwo;
        this.recurrentStateSize = hiddenSizes[hiddenSizes.length - 1];

        if (!qualiaSheafInstance || !qualiaSheafInstance.complex || !Array.isArray(qualiaSheafInstance.complex.vertices)) {
            console.error('CRITICAL ERROR: OntologicalWorldModel requires a pre-initialized FloquetPersistentSheaf with valid complex.vertices.');
            logger.error('CRITICAL ERROR: OntologicalWorldModel requires a pre-initialized FloquetPersistentSheaf with valid complex.vertices.');
            throw new Error('OntologicalWorldModel must be initialized with a valid FloquetPersistentSheaf instance.');
        }
        this.qualiaSheaf = qualiaSheafInstance;

        this._initializeNetwork();

        this.freeEnergy = 0;
        this.predictionError = 0;
        this.ready = false;
        this.lastActivations = [];
        this.lastActionLogProbs = vecZeros(this.actionDim);
        this.lastStateValue = 0;
        this.lastChosenActionLogProb = 0;
        this.actorLoss = 0;
        this.criticLoss = 0;
        this.predictionLoss = 0;

        logger.info(`OWM constructed (${isPlayerTwo ? 'AI' : 'Player'}): stateDim=${this.stateDim}, inputDim=${this.inputDim}, qualiaSheaf.qDim=${this.qualiaSheaf.qDim}, expectedQualiaInputLength=${this.expectedQualiaInputLength}`);
    }

    /**
     * Initializes network weights and dimensions, allowing reinitialization if the sheaf's topology changes.
     */
    _initializeNetwork() {
        this.qDim = this.qualiaSheaf.qDim;
        const nVertices = this.qualiaSheaf.complex.vertices.length;
        this.expectedQualiaInputLength = nVertices * this.qDim;
        this.inputDim = this.stateDim + this.expectedQualiaInputLength;

        this.hiddenState = vecZeros(this.recurrentStateSize);
        this.cellState = vecZeros(this.recurrentStateSize);

        const combinedInputSize = this.inputDim + this.recurrentStateSize;
        const weightScale = Math.sqrt(2.0 / combinedInputSize);

        this.Wf = randomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);
        this.Wi = randomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);
        this.Wc = randomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);
        this.Wo = randomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);

        [this.Wf, this.Wi, this.Wc, this.Wo].forEach((m, i) => {
            if (!isFiniteMatrix(m)) {
                console.error(`ERROR: Non-finite LSTM weight matrix detected at index ${i} during OWM construction. Reinitializing to zeros.`, m);
                logger.error(`Non-finite LSTM weight matrix detected at index ${i}; reinitializing to zeros.`);
                m.forEach(row => row.fill(0));
            }
        });

        this.bf = vecZeros(this.recurrentStateSize);
        this.bi = vecZeros(this.recurrentStateSize);
        this.bc = vecZeros(this.recurrentStateSize);
        this.bo = vecZeros(this.recurrentStateSize);
        [this.bf, this.bi, this.bc, this.bo].forEach((v, i) => {
            if (!isFiniteVector(v)) {
                console.warn(`WARN: Non-finite LSTM bias vector detected at index ${i} during OWM construction. Setting to zeros.`, v);
                logger.warn(`Non-finite LSTM bias vector detected at index ${i}; setting to zeros.`);
                v.fill(0);
            }
        });

        const actorOutputScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.actionDim));
        this.actorHead = { W: randomMatrix(this.actionDim, this.recurrentStateSize, actorOutputScale), b: vecZeros(this.actionDim) };

        const criticOutputScale = Math.sqrt(2.0 / (this.recurrentStateSize + 1));
        this.criticHead = { W: randomMatrix(1, this.recurrentStateSize, criticOutputScale), b: vecZeros(1) };

        const statePredScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.stateDim));
        this.statePredHead = { W: randomMatrix(this.stateDim, this.recurrentStateSize, statePredScale), b: vecZeros(this.stateDim) };

        const varScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.stateDim));
        this.varianceHead = { W: randomMatrix(this.stateDim, this.recurrentStateSize, varScale), b: vecZeros(this.stateDim) };
        if (!isFiniteMatrix(this.varianceHead.W) || !isFiniteVector(this.varianceHead.b)) {
            console.error('ERROR: Non-finite variance head detected during OWM initialization; reinitializing to zeros.', this.varianceHead);
            logger.error('Non-finite variance head detected during initialization; reinitializing to zeros.');
            this.varianceHead.W.forEach(row => row.fill(0));
            this.varianceHead.b.fill(0);
        }

        const anticipatoryScale = Math.sqrt(2.0 / (this.recurrentStateSize + 1));
        this.anticipatoryHead = { W: randomMatrix(1, this.recurrentStateSize, anticipatoryScale), b: vecZeros(1) };

        this.attentionWeights = randomMatrix(nVertices, this.stateDim, 0.1);
        this.lastSoftmaxScores = vecZeros(nVertices);
        this.coherencePrior = vecZeros(this.inputDim);

        logger.info(`OWM network (re)initialized. New inputDim=${this.inputDim}`);
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

    _getFullInputVector(rawStateVector) {
        if (!isFiniteVector(rawStateVector) || rawStateVector.length !== this.stateDim) {
            console.error('OWM._getFullInputVector: Invalid rawStateVector. Returning zeros.', { rawStateVector });
            logger.error('OWM._getFullInputVector: Invalid rawStateVector. Returning zeros.', { rawStateVector });
            return vecZeros(this.inputDim);
        }

        const attentionalInput = this.applyAttention(rawStateVector);
        const qualiaVector = this.qualiaSheaf.getStalksAsVector();
        if (!isFiniteVector(qualiaVector) || qualiaVector.length !== this.expectedQualiaInputLength) {
            console.error('OWM._getFullInputVector: Invalid qualiaVector from sheaf. Resetting and returning zeros.', { qualiaVector, expected: this.expectedQualiaInputLength, got: qualiaVector.length });
            logger.error('OWM._getFullInputVector: Invalid qualiaVector from sheaf. Resetting and returning zeros.', { qualiaVector });
            this.qualiaSheaf.stalks.forEach((stalk, key) => {
                if (!isFiniteVector(stalk)) {
                    this.qualiaSheaf.stalks.set(key, vecZeros(this.qDim));
                }
            });
            return vecZeros(this.inputDim);
        }

        const fullInput = new Float32Array(this.inputDim);
        fullInput.set(attentionalInput.slice(0, this.stateDim), 0);
        fullInput.set(qualiaVector, this.stateDim);

        if (!isFiniteVector(fullInput)) {
            console.error('OWM._getFullInputVector: Generated fullInput is non-finite. Returning zeros.', { fullInput });
            logger.error('OWM._getFullInputVector: Generated fullInput is non-finite. Returning zeros.', { fullInput });
            return vecZeros(this.inputDim);
        }
        return fullInput;
    }

    applyAttention(input) {
        if (!isFiniteVector(input) || input.length !== this.stateDim) {
            console.error('OWM.applyAttention: Invalid input dimension or non-finite. Returning zeros.', { expected: this.stateDim, got: input.length, input_finite: isFiniteVector(input) });
            logger.error('OWM.applyAttention: Invalid input dimension or non-finite. Returning zeros.');
            this.lastSoftmaxScores.fill(0);
            return vecZeros(this.stateDim);
        }

        const nVertices = this.qualiaSheaf.complex.vertices.length;
        if (!this.attentionWeights || this.attentionWeights.length !== nVertices || (this.attentionWeights.length > 0 && this.attentionWeights[0].length !== this.stateDim)) {
            console.warn(`OWM.applyAttention: attentionWeights have incorrect dimensions (${this.attentionWeights?.length}x${this.attentionWeights[0]?.length}). Reinitializing to ${nVertices}x${this.stateDim}.`);
            this.attentionWeights = randomMatrix(nVertices, this.stateDim, 0.1);
            this.lastSoftmaxScores = vecZeros(nVertices);
        }

        const scores = this.qualiaSheaf.complex.vertices.map((vertex, i) => {
            const weights_row = this.attentionWeights[i];
            const compatible_w_row = weights_row.slice(0, input.length);
            let score = dot(compatible_w_row, input);
            if (this.qualiaSheaf.stalks.has(vertex)) {
                const stalkData = this.qualiaSheaf.stalks.get(vertex);
                if (isFiniteVector(stalkData)) {
                    score += 0.1 * (stalkData[2] || 0);
                }
            }
            return score;
        });

        if (!isFiniteVector(scores)) {
            console.warn('OWM.applyAttention: Scores became non-finite. Returning zeros.', { scores });
            logger.warn('OWM.applyAttention: Scores became non-finite. Returning zeros.');
            this.lastSoftmaxScores.fill(0);
            return vecZeros(this.stateDim);
        }

        this.lastSoftmaxScores = softmax(new Float32Array(scores));
        const att = vecZeros(input.length);
        for (let i = 0; i < this.lastSoftmaxScores.length; i++) {
            for (let j = 0; j < input.length; j++) {
                att[j] += (this.lastSoftmaxScores[i] || 0) * (input[j] || 0);
            }
        }
        const weightedAttended = vecAdd(input, vecScale(att, this.qualiaSheaf.beta));
        return new Float32Array(isFiniteVector(weightedAttended) ? weightedAttended.map(v => clamp(v, -100, 100)) : vecZeros(this.stateDim));
    }

    async forward(input, recurrentInput = null) {
        if (!this.ready) {
            console.warn('OWM.forward: Not ready; returning corrupted.');
            logger.warn('OWM.forward: Not ready; returning corrupted.');
            return { corrupted: true };
        }

        this._checkAndReinitializeNetwork();

        if (!isFiniteVector(input) || input.length !== this.inputDim) {
            console.error('OWM.forward: Invalid input for current network dimensions. Returning corrupted.', { expected: this.inputDim, got: input.length });
            logger.error('OWM.forward: Invalid input for current network dimensions. Returning corrupted.', { expected: this.inputDim, got: input.length });
            this.resetRecurrentState();
            return { corrupted: true };
        }

        const hPrev = recurrentInput || this.hiddenState;
        const cPrev = recurrentInput ? vecZeros(this.recurrentStateSize) : this.cellState;
        const combinedInput = new Float32Array([...input, ...hPrev]);

        const [fPre, iPre, cPre, oPre] = await Promise.all([
            runWorkerTask('matVecMul', { matrix: flattenMatrix(this.Wf), vector: combinedInput }),
            runWorkerTask('matVecMul', { matrix: flattenMatrix(this.Wi), vector: combinedInput }),
            runWorkerTask('matVecMul', { matrix: flattenMatrix(this.Wc), vector: combinedInput }),
            runWorkerTask('matVecMul', { matrix: flattenMatrix(this.Wo), vector: combinedInput })
        ]);

        const forgetGate = sigmoidVec(vecAdd(fPre, this.bf));
        const inputGate = sigmoidVec(vecAdd(iPre, this.bi));
        const candidateCell = tanhVec(vecAdd(cPre, this.bc));
        const outputGate = sigmoidVec(vecAdd(oPre, this.bo));

        if (![forgetGate, inputGate, candidateCell, outputGate].every(isFiniteVector)) {
            console.error('OWM.forward: Non-finite LSTM gates detected; resetting state and returning safe defaults.', { forgetGate, inputGate, candidateCell, outputGate });
            logger.error('OWM.forward: Non-finite LSTM gates detected; resetting state and returning safe defaults.');
            this.resetRecurrentState();
            return { corrupted: true };
        }

        this.cellState = vecAdd(vecMul(forgetGate, cPrev), vecMul(inputGate, candidateCell));
        this.hiddenState = vecMul(outputGate, tanhVec(this.cellState));

        if (!isFiniteVector(this.cellState) || !isFiniteVector(this.hiddenState)) {
            console.error('OWM.forward: Non-finite LSTM cell or hidden state detected; resetting state.', { cellState: this.cellState, hiddenState: this.hiddenState });
            logger.error('OWM.forward: Non-finite LSTM cell or hidden state detected; resetting state.');
            this.resetRecurrentState();
            return { corrupted: true };
        }

        const [rawActionLogits, rawStateValue, rawNextState, rawLogVar, rawAnticipatory] = await Promise.all([
            runWorkerTask('matVecMul', { matrix: flattenMatrix(this.actorHead.W), vector: this.hiddenState }),
            runWorkerTask('matVecMul', { matrix: flattenMatrix(this.criticHead.W), vector: this.hiddenState }),
            runWorkerTask('matVecMul', { matrix: flattenMatrix(this.statePredHead.W), vector: this.hiddenState }),
            runWorkerTask('matVecMul', { matrix: flattenMatrix(this.varianceHead.W), vector: this.hiddenState }),
            runWorkerTask('matVecMul', { matrix: flattenMatrix(this.anticipatoryHead.W), vector: this.hiddenState })
        ]);

        const actionLogits = vecAdd(rawActionLogits, this.actorHead.b);
        const stateValue = vecAdd(rawStateValue, this.criticHead.b)[0] || 0;
        const nextStatePrediction = vecAdd(rawNextState, this.statePredHead.b);
        const logVar = vecAdd(rawLogVar, this.varianceHead.b);
        const anticipatoryReward = vecAdd(rawAnticipatory, this.anticipatoryHead.b)[0] || 0;
        const variance = new Float32Array(logVar.map(v => Math.max(Math.exp(clamp(v, -20, 20)), 1e-6)));

        if (!isFiniteVector(actionLogits) || !Number.isFinite(stateValue) || !isFiniteVector(nextStatePrediction) || !isFiniteVector(variance) || !Number.isFinite(anticipatoryReward)) {
            console.error('OWM.forward: Non-finite outputs from heads detected. Returning corrupted.', { actionLogits, stateValue, nextStatePrediction, variance, anticipatoryReward });
            logger.error('OWM.forward: Non-finite outputs from heads detected. Returning corrupted.');
            this.resetRecurrentState();
            return { corrupted: true };
        }

        const activations = [input.slice(), this.cellState.slice(), this.hiddenState.slice(), actionLogits.slice()];

        return {
            actionLogits,
            stateValue,
            nextStatePrediction: this._formatStatePrediction(nextStatePrediction),
            variance,
            anticipatoryReward,
            activations,
            corrupted: false
        };
    }

    _formatStatePrediction(stateVector) {
        if (!isFiniteVector(stateVector) || stateVector.length < 3) {
            console.warn('OWM._formatStatePrediction: Invalid stateVector; returning default state.', { stateVector });
            logger.warn('OWM._formatStatePrediction: Invalid stateVector; returning default state.', { stateVector });
            return { x: 0, z: 0, rot: 0, raw: vecZeros(this.stateDim) };
        }
        return {
            x: clamp(stateVector[0], -50, 50),
            z: clamp(stateVector[1], -50, 50),
            rot: clamp(stateVector[2], 0, 2 * Math.PI),
            raw: stateVector
        };
    }

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

    softmax(actionProbs) {
        if (!isFiniteVector(actionProbs)) {
            console.warn('OWM.softmax: Input actionProbs are not finite. Returning uniform probabilities.');
            logger.warn('OWM.softmax: Input actionProbs are not finite. Returning uniform probabilities.');
            return vecZeros(this.actionDim).fill(1 / this.actionDim);
        }

        const maxProb = Math.max(...actionProbs);
        if (!Number.isFinite(maxProb)) {
            console.warn('OWM.softmax: maxProb is non-finite. Returning uniform probabilities.', { maxProb, actionProbs });
            logger.warn('OWM.softmax: maxProb is non-finite. Returning uniform probabilities.');
            return vecZeros(this.actionDim).fill(1 / this.actionDim);
        }

        const exp_logits = new Float32Array(actionProbs.length);
        for (let i = 0; i < actionProbs.length; i++) {
            const val = Math.exp(actionProbs[i] - maxProb);
            exp_logits[i] = Number.isFinite(val) ? val : 0;
        }

        let sum_exp_logits = exp_logits.reduce((sum, val) => sum + val, 0);
        const safe_sum_exp_logits = (Number.isFinite(sum_exp_logits) && sum_exp_logits > 1e-9) ? sum_exp_logits : 1e-9;

        const resultProbs = new Float32Array(actionProbs.length);
        for (let i = 0; i < actionProbs.length; i++) {
            const val = exp_logits[i] / safe_sum_exp_logits;
            resultProbs[i] = Number.isFinite(val) ? val : 0;
        }

        if (!isFiniteVector(resultProbs)) {
            console.warn('OWM.softmax: Output probabilities are not finite after calculation. Returning uniform probabilities.');
            logger.warn('OWM.softmax: Output probabilities are not finite after calculation. Returning uniform probabilities.');
            return vecZeros(this.actionDim).fill(1 / this.actionDim);
        }
        return resultProbs;
    }

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
            console.error('OWM.chooseAction: Forward pass reported corrupted outputs. Resetting state.', forwardResult);
            logger.error('OWM.chooseAction: Forward pass reported corrupted outputs. Resetting state.');
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

        const temperature = 1.0 / (1 + this.freeEnergy);
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

    async computeFreeEnergy(nextRawStateVector, predictedState) {
        if (!isFiniteVector(nextRawStateVector) || !isFiniteVector(predictedState.raw)) {
            this.freeEnergy = 1.0;
            console.warn('OWM.computeFreeEnergy: Invalid inputs; defaulting to 1.0.');
            logger.warn('OWM.computeFreeEnergy: Invalid inputs; defaulting to 1.0.');
            return;
        }
        const klProxy = norm2(vecSub(nextRawStateVector, predictedState.raw)) * 0.1;
        const sheafInconsistency = this.qualiaSheaf.inconsistency || 0;
        const coherenceReduction = (this.qualiaSheaf.coherence || 0) * 0.5;
        this.freeEnergy = clamp(klProxy + sheafInconsistency - coherenceReduction, 0, 10);
    }

    async learn(tdError, targetValue, nextRawStateVector, lr, prior = null) {
        if (!this.ready) {
            console.warn('OWM.learn: Not ready; returning zero losses.');
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
            console.error('OWM.learn: Error during forward pass for next state prediction:', e);
            logger.error('OWM.learn: Error during forward pass for next state prediction:', e);
            this.predictionError = 1.0;
            forwardResult = { nextStatePrediction: this._formatStatePrediction(vecZeros(this.stateDim)), corrupted: true };
        }

        const { nextStatePrediction, corrupted: predictionCorrupted } = forwardResult;

        if (predictionCorrupted || !isFiniteVector(nextRawStateVector) || !isFiniteVector(nextStatePrediction.raw)) {
            console.warn('OWM.learn: Invalid nextRawStateVector or nextStatePrediction for prediction loss. Setting to 1.0.', { nextRawStateVector, nextStatePrediction });
            logger.warn('OWM.learn: Invalid nextRawStateVector or nextStatePrediction for prediction loss. Setting to 1.0.');
            this.predictionError = 1.0;
        } else {
            this.predictionError = norm2(vecSub(nextRawStateVector, nextStatePrediction.raw)) * 0.1;
            this.predictionError = clamp(this.predictionError, 0, 10);
        }

        await this.computeFreeEnergy(nextRawStateVector, nextStatePrediction);

        let modulatedTdError = tdError;
        if (prior && isFiniteVector(prior)) {
            const priorInfluence = dot(prior.slice(0, this.actionDim), vecZeros(this.actionDim)) * (this.qualiaSheaf.coherence || 0.5);
            modulatedTdError += priorInfluence * 0.1;
            modulatedTdError = clamp(modulatedTdError, -10, 10);
        }

        const simulatedActorLoss = Math.abs(modulatedTdError) * 0.1 + Math.random() * 0.001;
        const simulatedCriticLoss = Math.abs(modulatedTdError) * 0.05 + Math.random() * 0.001;
        const totalPredictionLoss = this.predictionError + Math.random() * 0.001;

        this.actorLoss = clamp(simulatedActorLoss, 0, 10);
        this.criticLoss = clamp(simulatedCriticLoss, 0, 10);
        this.predictionLoss = clamp(totalPredictionLoss, 0, 10);

        this.coherencePrior = vecAdd(this.coherencePrior, vecScale(this.hiddenState, lr * (this.qualiaSheaf.coherence || 0)));

        if (this.qualiaSheaf.ready) {
            await this.qualiaSheaf.diffuseQualia(nextRawStateVector);
        } else {
            console.warn('OWM.learn: QualiaSheaf not ready for diffusion during learning. Skipping diffusion.');
            logger.warn('OWM.learn: QualiaSheaf not ready for diffusion during learning. Skipping diffusion.');
        }

        return {
            actorLoss: this.actorLoss,
            criticLoss: this.criticLoss,
            predictionLoss: this.predictionLoss
        };
    }

    dispose() {
        logger.info(`OWM for ${this.isPlayerTwo ? 'AI' : 'Player'} disposed.`);
    }

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
        this.coherencePrior = vecZeros(this.inputDim);
        logger.info('OWM recurrent state reset with coherence prior.');
    }
}
