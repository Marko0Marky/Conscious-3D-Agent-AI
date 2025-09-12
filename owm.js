// --- START OF FILE owm.js ---
import {
    clamp, dot, norm2, vecAdd, vecSub, vecScale, tanhVec, sigmoidVec, vecMul,
    randomMatrix, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix,
    logger, runWorkerTask, softmax // Import softmax from utils
} from './utils.js';
import { EnhancedQualiaSheaf } from './qualia-sheaf.js';

/**
 * Helper function for softmax
 * @param {Float32Array} logits - The raw output from the actor head.
 * @returns {Float32Array} Action probabilities.
 */


/**
 * Represents an Ontological World Model (OWM) for an AI, combining a Qualia Sheaf with a Recurrent Neural Network (LSTM-like).
 * It predicts future states and evaluates actions using an Actor-Critic architecture.
 */
export class OntologicalWorldModel {
    /**
     * @param {number} stateDim - Dimension of the input state vector.
     * @param {number} actionDim - Dimension of the action space.
     * @param {number} qDim - Dimension of qualia vectors.
     * @param {number[]} hiddenSizes - Array of hidden layer sizes for the recurrent network.
     * @param {boolean} isPlayerTwo - True if this OWM belongs to the second player (AI).
     */
    constructor(stateDim = 13, actionDim = 4, qDim = 7, hiddenSizes = [64, 64], isPlayerTwo = false) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.isPlayerTwo = isPlayerTwo;
        this.recurrentStateSize = hiddenSizes[hiddenSizes.length - 1];
        
        this.qualiaSheaf = new EnhancedQualiaSheaf(null, 8, qDim, 0.1, 0.1, 0.05);
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
        this.bf = vecZeros(this.recurrentStateSize);
        this.bi = vecZeros(this.recurrentStateSize);
        this.bc = vecZeros(this.recurrentStateSize);
        this.bo = vecZeros(this.recurrentStateSize);
        
        // --- Actor-Critic specific heads ---
        const actorOutputScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.actionDim));
        this.actorHead = { W: randomMatrix(this.actionDim, this.recurrentStateSize, actorOutputScale), b: vecZeros(this.actionDim) };
        
        const criticOutputScale = Math.sqrt(2.0 / (this.recurrentStateSize + 1));
        this.criticHead = { W: randomMatrix(1, this.recurrentStateSize, criticOutputScale), b: vecZeros(1) };
        
        const statePredScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.stateDim));
        this.statePredHead = { W: randomMatrix(this.stateDim, this.recurrentStateSize, statePredScale), b: vecZeros(this.stateDim) };

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

        logger.info(`OWM constructed (${isPlayerTwo ? 'AI' : 'Player'}): stateDim=${this.stateDim}, inputDim=${this.inputDim}, qualiaSheaf.qDim=${this.qualiaSheaf.qDim}, expectedQualiaInputLength=${this.expectedQualiaInputLength}`);
    }

    async initialize() {
        logger.info(`OWM.initialize() (${this.isPlayerTwo ? 'AI' : 'Player'}) called.`);
        try {
            await this.qualiaSheaf.initialize();
            this.ready = true;
            logger.info(`Recurrent OWM for ${this.isPlayerTwo ? 'AI' : 'Player'} ready.`);
        } catch (e) {
            logger.error(`Error during Recurrent OWM initialization (${this.isPlayerTwo ? 'AI' : 'Player'}):`, e);
            this.ready = false;
            throw e;
        }
    }

    async forward(input) {
        if (!isFiniteVector(input) || input.length !== this.inputDim) {
            logger.error('OWM.forward: Invalid input. Returning zeros.', {expected: this.inputDim, got: input.length});
            this.resetRecurrentState();
            return { actionLogits: vecZeros(this.actionDim), stateValue: 0, nextStatePrediction: vecZeros(this.stateDim), activations: [] };
        }
        
        const activations = [input.slice()];
        
        const clampedInput = new Float32Array(input.map(v => clamp(v, -100, 100)));
        const clampedHiddenState = new Float32Array(this.hiddenState.map(v => clamp(v, -100, 100)));
        const combinedInput = new Float32Array([...clampedInput, ...clampedHiddenState]);

        if (!isFiniteVector(combinedInput)) {
            logger.error('OWM.forward: Combined input for LSTM is non-finite. Resetting and returning zeros.');
            this.resetRecurrentState();
            return { actionLogits: vecZeros(this.actionDim), stateValue: 0, nextStatePrediction: vecZeros(this.stateDim), activations: [] };
        }

        const Wf_flat = flattenMatrix(this.Wf);
        const Wi_flat = flattenMatrix(this.Wi);
        const Wc_flat = flattenMatrix(this.Wc);
        const Wo_flat = flattenMatrix(this.Wo);
        
        try {
            const ft_raw = await runWorkerTask('matVecMul', {matrix: Wf_flat, vector: combinedInput}, 2000);
            const it_raw = await runWorkerTask('matVecMul', {matrix: Wi_flat, vector: combinedInput}, 2000);
            const c_tilde_raw = await runWorkerTask('matVecMul', {matrix: Wc_flat, vector: combinedInput}, 2000);
            const ot_raw = await runWorkerTask('matVecMul', {matrix: Wo_flat, vector: combinedInput}, 2000);

            if (!isFiniteVector(ft_raw) || !isFiniteVector(it_raw) || !isFiniteVector(c_tilde_raw) || !isFiniteVector(ot_raw)) {
                 logger.error('OWM.forward: A raw vector from worker is non-finite. Resetting and returning zeros.');
                 this.resetRecurrentState();
                 return { actionLogits: vecZeros(this.actionDim), stateValue: 0, nextStatePrediction: vecZeros(this.stateDim), activations: [] };
            }

            const ft = new Float32Array(sigmoidVec(vecAdd(ft_raw, this.bf)).map(v => clamp(v, 0, 1)));
            const it = new Float32Array(sigmoidVec(vecAdd(it_raw, this.bi)).map(v => clamp(v, 0, 1)));
            const c_tilde = new Float32Array(tanhVec(vecAdd(c_tilde_raw, this.bc)).map(v => clamp(v, -1, 1)));
            
            this.cellState = new Float32Array(vecAdd(vecMul(ft, this.cellState), vecMul(it, c_tilde)).map(v => clamp(v, -100, 100)));
            if (!isFiniteVector(this.cellState)) {
                logger.error('OWM.forward: Cell state became non-finite. Resetting.');
                this.cellState.fill(0);
            }
            activations.push(this.cellState.slice());

            const ot = new Float32Array(sigmoidVec(vecAdd(ot_raw, this.bo)).map(v => clamp(v, 0, 1)));
            this.hiddenState = new Float32Array(vecMul(ot, tanhVec(this.cellState)).map(v => clamp(v, -100, 100)));
            if (!isFiniteVector(this.hiddenState)) {
                logger.error('OWM.forward: Hidden state became non-finite. Resetting.');
                this.hiddenState.fill(0);
            }
            activations.push(this.hiddenState.slice());
        } catch (e) {
            logger.error('OWM.forward: Error during recurrent step. Resetting and returning zeros.', e);
            this.resetRecurrentState();
            return { actionLogits: vecZeros(this.actionDim), stateValue: 0, nextStatePrediction: vecZeros(this.stateDim), activations: [] };
        }

        let actionLogits, stateValueOutput, nextStatePrediction;
        try {
            const rawActionLogits = await runWorkerTask('matVecMul', { matrix: flattenMatrix(this.actorHead.W), vector: this.hiddenState }, 2000);
            const rawStateValueOutput = await runWorkerTask('matVecMul', { matrix: flattenMatrix(this.criticHead.W), vector: this.hiddenState }, 2000);
            const rawNextStatePrediction = await runWorkerTask('matVecMul', { matrix: flattenMatrix(this.statePredHead.W), vector: this.hiddenState }, 2000);

            if (!isFiniteVector(rawActionLogits) || !isFiniteVector(rawStateValueOutput) || !isFiniteVector(rawNextStatePrediction)) {
                logger.error('OWM.forward: A raw output vector from worker is non-finite. Returning zeros for this frame.');
                return { actionLogits: vecZeros(this.actionDim), stateValue: 0, nextStatePrediction: vecZeros(this.stateDim), activations: [] };
            }

            actionLogits = new Float32Array(vecAdd(rawActionLogits, this.actorHead.b).map(v => clamp(v, -10, 10)));
            stateValueOutput = new Float32Array(vecAdd(rawStateValueOutput, this.criticHead.b).map(v => clamp(v, -50, 50)));
            nextStatePrediction = new Float32Array(vecAdd(rawNextStatePrediction, this.statePredHead.b).map(v => clamp(v, -100, 100)));

        } catch (e) {
            logger.error('OWM.forward: Error during output head calculation. Returning zeros for this frame.', e);
            return { actionLogits: vecZeros(this.actionDim), stateValue: 0, nextStatePrediction: vecZeros(this.stateDim), activations: [] };
        }
        
        if (!isFiniteVector(actionLogits) || !isFiniteVector(stateValueOutput) || !isFiniteVector(nextStatePrediction)) {
            logger.warn('OWM.forward: Final output vectors are non-finite after clamping. Returning zeros.');
            return { actionLogits: vecZeros(this.actionDim), stateValue: 0, nextStatePrediction: vecZeros(this.stateDim), activations: [] };
        }

        const scalarStateValue = clamp(stateValueOutput[0] || 0, -50, 50);
        activations.push(actionLogits.slice());

        return { actionLogits, stateValue: scalarStateValue, nextStatePrediction, activations };
    }

    applyAttention(input) {
        if (!isFiniteVector(input) || input.length !== this.inputDim) {
            logger.error('OWM.applyAttention: Invalid input. Returning original input.', {expected: this.inputDim, got: input.length});
            this.lastSoftmaxScores.fill(0);
            return input;
        }

        const scores = this.attentionWeights.map(w => dot(w, input));
        if (!isFiniteVector(scores)) {
            logger.warn('OWM.applyAttention: Scores became non-finite. Returning original input.');
            this.lastSoftmaxScores.fill(0);
            return input;
        }

        const maxScore = scores.length > 0 ? Math.max(...scores) : 0;
        const expScores = new Float32Array(scores.map(s => Math.exp(s - maxScore)));
        
        let sum_exp_scores = expScores.reduce((a, b) => a + b, 0);
        const safe_sum_exp_scores = (Number.isFinite(sum_exp_scores) && sum_exp_scores > 1e-10) ? sum_exp_scores : 1e-10;
        
        const softmaxScores = new Float32Array(expScores.map(s => s / safe_sum_exp_scores));
        
        if (!isFiniteVector(softmaxScores)) {
            logger.warn('OWM.applyAttention: Softmax scores became non-finite. Returning original input.');
            this.lastSoftmaxScores.fill(0);
            return input;
        }

        this.lastSoftmaxScores = softmaxScores;

        const att = vecZeros(input.length);
        for (let i = 0; i < softmaxScores.length; i++) {
            for (let j = 0; j < input.length; j++) {
                att[j] += (softmaxScores[i] || 0) * (input[j] || 0);
            }
        }
        const weightedAttended = vecAdd(input, vecScale(att, this.qualiaSheaf.beta));
        return new Float32Array(isFiniteVector(weightedAttended) ? weightedAttended.map(v => clamp(v, -100, 100)) : input.map(v => clamp(v, -100, 100)));
    }

    async predict(state) {
        if (!this.ready) {
            logger.warn('OWM not ready for prediction.');
            return { actionProbs: vecZeros(this.actionDim), stateValue: 0, activations: [], corrupted: true };
        }
        if (!isFiniteVector(state) || state.length !== this.stateDim) {
            logger.error(`OWM.predict: Invalid state vector.`, { state });
            return { actionProbs: vecZeros(this.actionDim), stateValue: 0, activations: [], corrupted: true };
        }

        let corruptedFlag = false;

        try {
            await this.qualiaSheaf.diffuseQualia(state.slice(0, 8));
        } catch (e) {
            logger.error('OWM.predict: Error during qualia diffusion:', e);
            corruptedFlag = true;
        }
        
        const qualiaStalksMap = this.qualiaSheaf.stalks;
        let finalQualiaArray = vecZeros(this.expectedQualiaInputLength);

        if (qualiaStalksMap instanceof Map && qualiaStalksMap.size > 0) {
            let offset = 0;
            for (const vertexName of this.qualiaSheaf.graph.vertices) {
                let stalk = qualiaStalksMap.get(vertexName);
                if (!stalk || !isFiniteVector(stalk) || stalk.length !== this.qDim) {
                    stalk = vecZeros(this.qDim);
                    corruptedFlag = true;
                }
                finalQualiaArray.set(stalk, offset);
                offset += this.qDim;
            }
        } else {
            corruptedFlag = true;
        }

        const input = new Float32Array([...state, ...finalQualiaArray]);
        
        if (input.length !== this.inputDim || !isFiniteVector(input)) {
             logger.error(`FATAL: Input to OWM is invalid.`, { input_len: input.length, expected: this.inputDim });
             return { actionProbs: vecZeros(this.actionDim), stateValue: 0, activations: [], corrupted: true };
        }

        const attended = this.applyAttention(input);
        
        let forwardResult;
        try {
            forwardResult = await this.forward(attended);
        } catch (e) {
            logger.error('OWM.predict: Error during forward pass:', e);
            corruptedFlag = true;
            forwardResult = { actionLogits: vecZeros(this.actionDim), stateValue: 0, nextStatePrediction: vecZeros(this.stateDim), activations: [] };
        }

        const { actionLogits, stateValue, nextStatePrediction, activations } = forwardResult;
        const actionProbs = softmax(actionLogits);
        
        const error = norm2(vecSub(nextStatePrediction, state));
        this.predictionError = clamp(error, 0, 10);
        if (!Number.isFinite(this.predictionError)) { this.predictionError = 0; corruptedFlag = true; }

        this.freeEnergy = 0.85 * (this.freeEnergy || 0) + 0.15 * (this.predictionError * 0.5 + (this.qualiaSheaf.h1Dimension || 0));
        this.freeEnergy = clamp(this.freeEnergy, 0, 10);
        if (!Number.isFinite(this.freeEnergy)) { this.freeEnergy = 0; corruptedFlag = true; }

        this.lastActivations = activations;
        this.lastActionLogProbs = actionLogits;
        this.lastStateValue = stateValue;

        return { actionProbs, stateValue, activations, corrupted: corruptedFlag };
    }

    async learn(state, actionIndex, reward, nextState, isDone, learningRate = 0.01, gamma = 0.99) {
        if (!isFiniteVector(state) || !isFiniteVector(nextState) || !Number.isFinite(reward) || !Number.isFinite(actionIndex) || actionIndex < 0 || actionIndex >= this.actionDim) {
            logger.warn('OWM.learn: Invalid input. Skipping step.');
            return;
        }

        const { stateValue: currentStateValue } = await this.predict(state);
        const { stateValue: nextStateValue } = await this.predict(nextState);

        const safeCurrentStateValue = Number.isFinite(currentStateValue) ? currentStateValue : 0;
        const safeNextStateValue = isDone ? 0 : (Number.isFinite(nextStateValue) ? nextStateValue : 0);
        const safeReward = Number.isFinite(reward) ? reward : 0;

        const tdTarget = safeReward + gamma * safeNextStateValue;
        const advantage = tdTarget - safeCurrentStateValue;

        this.criticLoss = 0.5 * advantage * advantage;
        const criticGradScale = learningRate * advantage;
        const lastRecurrentState = this.lastActivations[2];

        if (isFiniteVector(lastRecurrentState)) {
            for (let j = 0; j < lastRecurrentState.length; j++) {
                this.criticHead.W[0][j] = clamp((this.criticHead.W[0][j] || 0) + criticGradScale * (lastRecurrentState[j] || 0), -1, 1);
            }
            this.criticHead.b[0] = clamp((this.criticHead.b[0] || 0) + criticGradScale, -1, 1);
        }

        const actionProbsForLog = softmax(this.lastActionLogProbs);
        let chosenActionProb = Math.max(1e-9, actionProbsForLog[actionIndex] || 1e-9);

        this.lastChosenActionLogProb = Math.log(chosenActionProb);
        this.actorLoss = -this.lastChosenActionLogProb * advantage;
        const actorGradScale = learningRate * advantage;

        if (isFiniteVector(lastRecurrentState)) {
            for (let j = 0; j < lastRecurrentState.length; j++) {
                this.actorHead.W[actionIndex][j] = clamp((this.actorHead.W[actionIndex][j] || 0) + actorGradScale * (lastRecurrentState[j] || 0), -1, 1);
            }
            this.actorHead.b[actionIndex] = clamp((this.actorHead.b[actionIndex] || 0) + actorGradScale, -1, 1);
        }

        this.actorHead.W = this.actorHead.W.map(row => new Float32Array(row.map(v => Number.isFinite(v) ? v : 0)));
        this.actorHead.b = new Float32Array(this.actorHead.b.map(v => Number.isFinite(v) ? v : 0));
        this.criticHead.W = this.criticHead.W.map(row => new Float32Array(row.map(v => Number.isFinite(v) ? v : 0)));
        this.criticHead.b = new Float32Array(this.criticHead.b.map(v => Number.isFinite(v) ? v : 0));
    }

    resetRecurrentState() {
        this.hiddenState.fill(0);
        this.cellState.fill(0);
        logger.info(`OWM recurrent state for ${this.isPlayerTwo ? 'AI' : 'Player'} has been reset.`);
    }
}
