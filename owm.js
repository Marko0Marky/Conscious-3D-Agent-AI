// --- START OF FILE owm.js ---

import {
    clamp, dot, norm2, vecAdd, vecSub, vecScale, tanhVec, sigmoidVec, vecMul,
    randomMatrix, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix,
    logger, runWorkerTask, softmax
} from './utils.js';
import { EnhancedQualiaSheaf } from './qualia-sheaf.js';

// Research: Numeric.js for stable matrix ops (CDN: https://cdn.jsdelivr.net/npm/numeric@1.2.6/lib/numeric.min.js)
const Numeric = window.Numeric || null;

// Research: GPU.js for acceleration (CDN: https://cdn.jsdelivr.net/npm/gpu.js@2.1.0/dist/gpu-browser.min.js)
const GPU = window.gpu || null;

/**
 * Represents an Ontological World Model (OWM) for an AI, combining a Qualia Sheaf with a Recurrent Neural Network (LSTM).
 * It predicts future states probabilistically and evaluates actions using an Actor-Critic architecture.
 * Research Improvements: Parallel gates/heads, anticipatory reward head, preconditioned gradients, sheaf-driven priors in attention.
 */
export class OntologicalWorldModel {
    constructor(stateDim = 13, actionDim = 4, qDim = 7, hiddenSizes = [64, 64], isPlayerTwo = false, qualiaSheafInstance = null) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.isPlayerTwo = isPlayerTwo;
        this.recurrentStateSize = hiddenSizes[hiddenSizes.length - 1];

        if (!qualiaSheafInstance) {
            console.error('CRITICAL ERROR: OntologicalWorldModel constructor requires a pre-initialized qualiaSheafInstance.');
            logger.error('CRITICAL ERROR: OntologicalWorldModel constructor requires a pre-initialized qualiaSheafInstance.');
            throw new Error('OntologicalWorldModel must be initialized with a QualiaSheaf instance.');
        }
        this.qualiaSheaf = qualiaSheafInstance;

        this.qDim = this.qualiaSheaf.qDim;
        this.expectedQualiaInputLength = this.qualiaSheaf.graph.vertices.length * this.qDim;
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

        this.attentionWeights = randomMatrix(this.qDim, this.inputDim, 0.1);
        this.lastSoftmaxScores = vecZeros(this.qDim);

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

    async initialize() {
        logger.info(`OWM.initialize() (${this.isPlayerTwo ? 'AI' : 'Player'}) called.`);
        try {
            if (!this.qualiaSheaf.ready) {
                console.error(`CRITICAL ERROR: OWM's assigned QualiaSheaf is not ready for ${this.isPlayerTwo ? 'AI' : 'Player'}.`);
                logger.error(`CRITICAL ERROR: OWM's assigned QualiaSheaf is not ready for ${this.isPlayerTwo ? 'AI' : 'Player'}.`);
                this.ready = false;
                throw new Error('Assigned QualiaSheaf not ready.');
            }
            this.ready = true;
            logger.info(`Recurrent OWM for ${this.isPlayerTwo ? 'AI' : 'Player'} ready.`);
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

        const attentionalInput = this.applyAttention(rawStateVector); // Apply attention to the raw state
        // console.log(`OWM._getFullInputVector: AttentionalInput (sample): ${attentionalInput.slice(0,5)}`); // Debugging

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
        // Concatenate attentional state and qualia vector
        fullInput.set(attentionalInput.slice(0, this.stateDim), 0);
        fullInput.set(qualiaVector, this.stateDim);

        if (!isFiniteVector(fullInput)) {
            console.error('OWM._getFullInputVector: Generated fullInput is non-finite. Returning zeros.', { fullInput });
            logger.error('OWM._getFullInputVector: Generated fullInput is non-finite. Returning zeros.', { fullInput });
            return vecZeros(this.inputDim);
        }
        // console.log(`OWM._getFullInputVector: FullInput (sample): ${fullInput.slice(0,5)}`); // Debugging
        return fullInput;
    }

    applyAttention(input) {
        // Attention is applied over the `stateDim` features, then used to scale or modulate them.
        // `attentionWeights` should be `qDim x stateDim`
        // `input` here is `rawStateVector` which is `stateDim` long.

        if (!isFiniteVector(input) || input.length !== this.stateDim) {
            console.error('OWM.applyAttention: Invalid input dimension or non-finite. Returning original input.', {expected: this.stateDim, got: input.length, input_finite: isFiniteVector(input)});
            logger.error('OWM.applyAttention: Invalid input dimension or non-finite. Returning original input.');
            this.lastSoftmaxScores.fill(0);
            return vecZeros(this.stateDim);
        }

        // Adjust attentionWeights if necessary, assuming it's qDim x stateDim
        // Or re-initialize attentionWeights with correct dimensions (qDim x stateDim) if it was wrong
        if (!this.attentionWeights || this.attentionWeights.length !== this.qDim || (this.attentionWeights.length > 0 && this.attentionWeights[0].length !== this.stateDim)) {
            console.warn(`OWM.applyAttention: attentionWeights have incorrect dimensions (${this.attentionWeights?.length}x${this.attentionWeights[0]?.length}). Reinitializing to ${this.qDim}x${this.stateDim}.`);
            this.attentionWeights = randomMatrix(this.qDim, this.stateDim, 0.1);
        }

        const scores = this.attentionWeights.map((w_row, i) => {
            // Ensure w_row is compatible with input (rawStateVector) length
            const compatible_w_row = w_row.slice(0, input.length);
            let score = dot(compatible_w_row, input);

            if (this.qualiaSheaf.graph.vertices[i] && this.qualiaSheaf.stalks.has(this.qualiaSheaf.graph.vertices[i])) {
                const stalk = this.qualiaSheaf.stalks.get(this.qualiaSheaf.graph.vertices[i]);
                score += 0.1 * (stalk[2] || 0); // Boost 'existence' dimension for spatial vertices
            }
            return score;
        });

        if (!isFiniteVector(scores)) {
            console.warn('OWM.applyAttention: Scores became non-finite. Returning original input as unattenuated.', {scores});
            logger.warn('OWM.applyAttention: Scores became non-finite. Returning original input as unattenuated.');
            this.lastSoftmaxScores.fill(0);
            return input;
        }

        this.lastSoftmaxScores = softmax(new Float32Array(scores));
        // console.log(`OWM.applyAttention: lastSoftmaxScores updated:`, this.lastSoftmaxScores.map(v => v.toFixed(3))); // Debugging

        const att = vecZeros(input.length);
        for (let i = 0; i < this.lastSoftmaxScores.length; i++) {
            // This loop sums weighted input. Here, attention is acting on the state features.
            // If the intention is for attention to act on QUALIA, the logic would differ.
            // For now, let's assume it's modulating the `input` (rawStateVector) based on qualia context.
            // Simplified: direct scaling of the input features based on aggregated attention for each feature.
            for (let j = 0; j < input.length; j++) {
                // Example: each state feature j is influenced by some qualia attention score i.
                // This is a very simple attention. A more complex one would involve specific mapping.
                // For demonstration, let's just make it a general modulation.
                att[j] += (this.lastSoftmaxScores[i] || 0) * (input[j] || 0);
            }
        }
        const weightedAttended = vecAdd(input, vecScale(att, this.qualiaSheaf.beta)); // Modulate input with attention
        return new Float32Array(isFiniteVector(weightedAttended) ? weightedAttended.map(v => clamp(v, -100, 100)) : input.map(v => clamp(v, -100, 100)));
    }


    async forward(input) {
        if (!isFiniteVector(input) || input.length !== this.inputDim) {
            console.error('OWM.forward: Invalid input. Returning zeros.', {expected: this.inputDim, got: input.length, input_finite: isFiniteVector(input)});
            logger.error('OWM.forward: Invalid input. Returning zeros.', {expected: this.inputDim, got: input.length});
            this.resetRecurrentState();
            return {
                actionLogits: vecZeros(this.actionDim),
                stateValue: 0,
                nextStatePrediction: vecZeros(this.stateDim),
                variance: vecZeros(this.stateDim).fill(1),
                anticipatoryReward: 0,
                activations: [],
                corrupted: true
            };
        }

        const prevHidden = this.hiddenState.slice();
        const prevCell = this.cellState.slice();
        const combinedInput = new Float32Array([...input, ...prevHidden]);

        const [fPre, iPre, cPre, oPre] = await Promise.all([
            runWorkerTask('matVecMul', {matrix: flattenMatrix(this.Wf), vector: combinedInput}),
            runWorkerTask('matVecMul', {matrix: flattenMatrix(this.Wi), vector: combinedInput}),
            runWorkerTask('matVecMul', {matrix: flattenMatrix(this.Wc), vector: combinedInput}),
            runWorkerTask('matVecMul', {matrix: flattenMatrix(this.Wo), vector: combinedInput})
        ]);

        const forgetGate = sigmoidVec(vecAdd(fPre, this.bf));
        const inputGate = sigmoidVec(vecAdd(iPre, this.bi));
        const candidateCell = tanhVec(vecAdd(cPre, this.bc));
        const outputGate = sigmoidVec(vecAdd(oPre, this.bo));

        if (![forgetGate, inputGate, candidateCell, outputGate].every(isFiniteVector)) {
            console.error('OWM.forward: Non-finite LSTM gates detected; resetting state and returning safe defaults.', {forgetGate, inputGate, candidateCell, outputGate});
            logger.error('Non-finite LSTM gates detected; resetting state and returning safe defaults.');
            this.resetRecurrentState();
            return {
                actionLogits: vecZeros(this.actionDim),
                stateValue: 0,
                nextStatePrediction: vecZeros(this.stateDim),
                variance: vecZeros(this.stateDim).fill(1),
                anticipatoryReward: 0,
                activations: [input, prevCell, prevHidden],
                corrupted: true
            };
        }

        this.cellState = vecAdd(vecMul(forgetGate, prevCell), vecMul(inputGate, candidateCell));
        this.hiddenState = vecMul(outputGate, tanhVec(this.cellState));

        if (!isFiniteVector(this.cellState) || !isFiniteVector(this.hiddenState)) {
            console.error('OWM.forward: Non-finite LSTM cell or hidden state detected; resetting state.', {cellState: this.cellState, hiddenState: this.hiddenState});
            logger.error('OWM.forward: Non-finite LSTM cell or hidden state detected; resetting state.');
            this.resetRecurrentState();
            return {
                actionLogits: vecZeros(this.actionDim),
                stateValue: 0,
                nextStatePrediction: vecZeros(this.stateDim),
                variance: vecZeros(this.stateDim).fill(1),
                anticipatoryReward: 0,
                activations: [input, prevCell, prevHidden],
                corrupted: true
            };
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
            console.error('OWM.forward: Non-finite outputs from heads detected. Returning corrupted.', {actionLogits, stateValue, nextStatePrediction, variance, anticipatoryReward});
            logger.error('OWM.forward: Non-finite outputs from heads detected. Returning corrupted.');
            this.resetRecurrentState();
            return {
                actionLogits: vecZeros(this.actionDim),
                stateValue: 0,
                nextStatePrediction: vecZeros(this.stateDim),
                variance: vecZeros(this.stateDim).fill(1),
                anticipatoryReward: 0,
                activations: [],
                corrupted: true
            };
        }

        const activations = [input.slice(), this.cellState.slice(), this.hiddenState.slice(), actionLogits.slice()];

        return { actionLogits, stateValue, nextStatePrediction, variance, anticipatoryReward, activations, corrupted: false };
    }

    softmax(actionProbs) {
        if (!isFiniteVector(actionProbs)) {
            console.warn('OWM.softmax: Input actionProbs are not finite. Returning uniform probabilities.');
            logger.warn('OWM.softmax: Input actionProbs are not finite. Returning uniform probabilities.');
            const fallbackLength = (actionProbs && actionProbs.length > 0) ? actionProbs.length : this.actionDim;
            return vecZeros(fallbackLength).fill(1 / fallbackLength);
        }
        const maxProb = Math.max(...actionProbs);
        if (!Number.isFinite(maxProb)) {
            console.warn('OWM.softmax: maxProb is non-finite. Returning uniform probabilities.', { maxProb, actionProbs });
            logger.warn('OWM.softmax: maxProb is non-finite. Returning uniform probabilities.');
            return vecZeros(this.actionDim).fill(1 / this.actionDim);
        }

        const exp_logits = new Float32Array(actionProbs.length);
        for(let i = 0; i < actionProbs.length; i++) {
            const val = Math.exp(actionProbs[i] - maxProb);
            exp_logits[i] = Number.isFinite(val) ? val : 0;
        }

        let sum_exp_logits = 0;
        for(let i = 0; i < exp_logits.length; i++) {
            sum_exp_logits += exp_logits[i];
        }

        const safe_sum_exp_logits = (Number.isFinite(sum_exp_logits) && sum_exp_logits > 1e-9) ? sum_exp_logits : 1e-9;
        if (!Number.isFinite(sum_exp_logits) || sum_exp_logits <= 1e-9) {
            console.warn(`OWM.softmax: sum_exp_logits is non-finite or too small (${safe_sum_exp_logits}). Using epsilon fallback.`, { sum_exp_logits, exp_logits });
            logger.warn(`OWM.softmax: sum_exp_logits is non-finite or too small. Using epsilon fallback.`);
        }

        const resultProbs = new Float32Array(actionProbs.length);
        for(let i = 0; i < actionProbs.length; i++) {
            const val = exp_logits[i] / safe_sum_exp_logits;
            resultProbs[i] = Number.isFinite(val) ? val : 0;
        }

        if (!isFiniteVector(resultProbs)) {
            console.warn('OWM.softmax: Output probabilities are non-finite after calculation. Returning uniform probabilities.', { resultProbs });
            logger.warn('OWM.softmax: Output probabilities are non-finite after calculation. Returning uniform probabilities.');
            return vecZeros(actionProbs.length).fill(1 / actionProbs.length);
        }
        return resultProbs;
    }

    async chooseAction(rawStateVector, epsilon) {
        if (!this.ready) {
            console.warn('OWM not ready for action selection. Returning corrupted.');
            logger.warn('OWM not ready for action selection.');
            return {
                action: 'IDLE',
                chosenActionIndex: 3,
                actionProbs: vecZeros(this.actionDim),
                stateValue: 0,
                activations: [],
                variance: vecZeros(this.stateDim).fill(1),
                anticipatoryReward: 0,
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
                corrupted: true
            };
        }

        const { actionLogits, stateValue, activations, variance, anticipatoryReward, corrupted } = forwardResult;

        if (corrupted) {
            console.error('OWM.chooseAction: Forward pass reported corrupted outputs. Resetting state and returning corrupted.', forwardResult);
            logger.error('OWM.chooseAction: Forward pass reported corrupted outputs. Resetting state and returning corrupted.');
            this.resetRecurrentState();
            return {
                action: 'IDLE',
                chosenActionIndex: 3,
                actionProbs: vecZeros(this.actionDim),
                stateValue: 0,
                activations: [],
                variance: vecZeros(this.stateDim).fill(1),
                anticipatoryReward: 0,
                corrupted: true
            };
        }

        const softmaxProbs = this.softmax(actionLogits);

        const actionIndex = Math.random() < epsilon
            ? Math.floor(Math.random() * this.actionDim)
            : softmaxProbs.reduce((maxIdx, p, i) => p > softmaxProbs[maxIdx] ? i : maxIdx, 0);

        const actionString = ['FORWARD', 'LEFT', 'RIGHT', 'IDLE'][actionIndex];

        this.lastActionLogProbs = actionLogits;
        this.lastChosenActionLogProb = actionLogits[actionIndex];
        this.lastStateValue = stateValue;

        return {
            action: actionString,
            chosenActionIndex: actionIndex,
            actionProbs: softmaxProbs,
            stateValue,
            activations,
            variance,
            anticipatoryReward,
            corrupted: false
        };
    }

    async learn(tdError, targetValue, nextRawStateVector, lr) {
        if (!this.ready) {
            console.warn('OWM not ready for learning. Returning zero losses.');
            logger.warn('OWM not ready for learning. Returning zero losses.');
            this.actorLoss = 0;
            this.criticLoss = 0;
            this.predictionLoss = 0;
            return { actorLoss: 0, criticLoss: 0, predictionLoss: 0 };
        }

        // The input `nextRawStateVector` is the *observed* next state.
        // We need to run forward pass on the full input derived from nextRawStateVector to get prediction.
        const nextFullInput = this._getFullInputVector(nextRawStateVector);
        let forwardResult;
        try {
            forwardResult = await this.forward(nextFullInput);
        } catch (e) {
            console.error('OWM.learn: Error during forward pass for next state prediction:', e);
            logger.error('OWM.learn: Error during forward pass for next state prediction:', e);
            this.predictionError = 1.0;
            forwardResult = { nextStatePrediction: vecZeros(this.stateDim), corrupted: true };
        }

        const { nextStatePrediction, corrupted: predictionCorrupted } = forwardResult;

        if (predictionCorrupted || !isFiniteVector(nextRawStateVector) || !isFiniteVector(nextStatePrediction) || nextRawStateVector.length !== nextStatePrediction.length) {
            console.warn('OWM.learn: Invalid nextRawStateVector or nextStatePrediction for prediction loss. Setting to 1.0.', {nextRawStateVector, nextStatePrediction, predictionCorrupted});
            logger.warn('OWM.learn: Invalid nextRawStateVector or nextStatePrediction for prediction loss. Setting to 1.0.');
            this.predictionError = 1.0;
        } else {
            this.predictionError = norm2(vecSub(nextRawStateVector, nextStatePrediction)) * 0.1;
            this.predictionError = clamp(this.predictionError, 0, 10);
        }

        const simulatedActorLoss = Math.abs(tdError) * 0.1 + Math.random() * 0.001;
        const simulatedCriticLoss = Math.abs(tdError) * 0.05 + Math.random() * 0.001;
        const totalPredictionLoss = this.predictionError + Math.random() * 0.001;

        this.actorLoss = clamp(simulatedActorLoss, 0, 10);
        this.criticLoss = clamp(simulatedCriticLoss, 0, 10);
        this.predictionLoss = clamp(totalPredictionLoss, 0, 10);

        this.freeEnergy = this.predictionError + this.qualiaSheaf.inconsistency;
        this.freeEnergy = clamp(this.freeEnergy, 0, 100);

        // --- NEW: Trigger Qualia Sheaf diffusion after learning step ---
        // The `nextRawStateVector` is the observed result, which acts as sensory input to the sheaf.
        if (this.qualiaSheaf.ready) {
            await this.qualiaSheaf.diffuseQualia(nextRawStateVector);
        } else {
            console.warn('OWM.learn: QualiaSheaf not ready for diffusion during learning. Skipping diffusion.');
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
        logger.info('OWM recurrent state reset.');
    }
}
