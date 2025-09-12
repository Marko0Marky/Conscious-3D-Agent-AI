// --- START OF FILE owm.js ---
import {
    clamp, dot, norm2, vecAdd, vecSub, vecScale, tanhVec, sigmoidVec, vecMul,
    randomMatrix, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix,
    logger, runWorkerTask
} from './utils.js';
import { EnhancedQualiaSheaf } from './qualia-sheaf.js';

/**
 * Represents an Ontological World Model (OWM) for an AI, combining a Qualia Sheaf with a Recurrent Neural Network (LSTM-like).
 * It predicts future states and Q-values for actions.
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
        
        const qValueScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.actionDim));
        this.qValueHead = { W: randomMatrix(this.actionDim, this.recurrentStateSize, qValueScale), b: vecZeros(this.actionDim) };
        
        const statePredScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.stateDim));
        this.statePredHead = { W: randomMatrix(this.stateDim, this.recurrentStateSize, statePredScale), b: vecZeros(this.stateDim) };

        this.attentionWeights = randomMatrix(this.qDim, this.inputDim, 0.1);
        this.lastSoftmaxScores = vecZeros(this.qDim);
        
        this.freeEnergy = 0;
        this.predictionError = 0;
        this.ready = false;
        this.lastActivations = [];

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
            return { qValues: vecZeros(this.actionDim), nextState: vecZeros(this.stateDim), activations: [] };
        }
        
        const activations = [input.slice()];
        
        const combinedInput = new Float32Array([...input, ...this.hiddenState]);
        if (!isFiniteVector(combinedInput)) {
            logger.error('OWM.forward: Combined input for LSTM is non-finite. Resetting hidden/cell states and returning zeros.');
            this.resetRecurrentState();
            return { qValues: vecZeros(this.actionDim), nextState: vecZeros(this.stateDim), activations: [] };
        }

        const Wf_flat = flattenMatrix(this.Wf);
        const Wi_flat = flattenMatrix(this.Wi);
        const Wc_flat = flattenMatrix(this.Wc);
        const Wo_flat = flattenMatrix(this.Wo);
        
        let ft, it, c_tilde, ot;

        try {
            ft = sigmoidVec(vecAdd(await runWorkerTask('matVecMul', {matrix: Wf_flat, vector: combinedInput}, 2000), this.bf));
            it = sigmoidVec(vecAdd(await runWorkerTask('matVecMul', {matrix: Wi_flat, vector: combinedInput}, 2000), this.bi));
            c_tilde = tanhVec(vecAdd(await runWorkerTask('matVecMul', {matrix: Wc_flat, vector: combinedInput}, 2000), this.bc));
            
            this.cellState = vecAdd(vecMul(ft, this.cellState), vecMul(it, c_tilde));
            if (!isFiniteVector(this.cellState)) {
                logger.error('OWM.forward: Cell state became non-finite. Resetting.');
                this.cellState.fill(0);
            }
            activations.push(this.cellState.slice());

            ot = sigmoidVec(vecAdd(await runWorkerTask('matVecMul', {matrix: Wo_flat, vector: combinedInput}, 2000), this.bo));
            this.hiddenState = vecMul(ot, tanhVec(this.cellState));
            if (!isFiniteVector(this.hiddenState)) {
                logger.error('OWM.forward: Hidden state became non-finite. Resetting.');
                this.hiddenState.fill(0);
            }
            activations.push(this.hiddenState.slice());
        } catch (e) {
            logger.error('OWM.forward: Error during recurrent step:', e);
            this.resetRecurrentState();
            return { qValues: vecZeros(this.actionDim), nextState: vecZeros(this.stateDim), activations: [] };
        }

        let qValues, nextState;
        try {
            qValues = vecAdd(await runWorkerTask('matVecMul', { matrix: flattenMatrix(this.qValueHead.W), vector: this.hiddenState }, 2000), this.qValueHead.b);
            nextState = vecAdd(await runWorkerTask('matVecMul', { matrix: flattenMatrix(this.statePredHead.W), vector: this.hiddenState }, 2000), this.statePredHead.b);
        } catch (e) {
            logger.error('OWM.forward: Error during output head calculation:', e);
            return { qValues: vecZeros(this.actionDim), nextState: vecZeros(this.stateDim), activations: [] };
        }
        
        if (!isFiniteVector(qValues)) { logger.warn('OWM.forward: Q-values are non-finite. Returning zeros.'); qValues = vecZeros(this.actionDim); }
        if (!isFiniteVector(nextState)) { logger.warn('OWM.forward: Next state prediction is non-finite. Returning zeros.'); nextState = vecZeros(this.stateDim); }

        activations.push(qValues.slice());
        
        return { qValues, nextState, activations };
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
        const expScores = scores.map(s => Math.exp(s - maxScore));
        const sumExpScores = expScores.reduce((s, x) => s + x, 1e-10);
        const softmaxScores = new Float32Array(expScores.map(s => s / sumExpScores));
        
        if (!isFiniteVector(softmaxScores)) {
            logger.warn('OWM.applyAttention: Softmax scores became non-finite. Returning original input.');
            this.lastSoftmaxScores.fill(0);
            return input;
        }

        this.lastSoftmaxScores = softmaxScores;

        const att = vecZeros(input.length);
        for (let i = 0; i < softmaxScores.length; i++) {
            if (!Number.isFinite(softmaxScores[i])) continue;
            for (let j = 0; j < input.length; j++) {
                att[j] += softmaxScores[i] * (input[j] || 0);
            }
        }
        const weightedAttended = vecAdd(input, vecScale(att, this.qualiaSheaf.beta));
        return isFiniteVector(weightedAttended) ? weightedAttended : input;
    }

    async predict(state) {
        if (!this.ready) {
            logger.warn('OWM not ready for prediction.');
            return { qValues: vecZeros(this.actionDim), activations: [], corrupted: true };
        }
        if (!isFiniteVector(state) || state.length !== this.stateDim) {
            logger.error(`OWM.predict: Invalid state vector received. Expected length ${this.stateDim}, got ${state.length}.`, { state });
            return { qValues: vecZeros(this.actionDim), activations: [], corrupted: true };
        }

        let corruptedFlag = false;

        try {
            await this.qualiaSheaf.diffuseQualia(state.slice(0, 8));
        } catch (e) {
            logger.error('OWM.predict: Error during qualia diffusion:', e);
            corruptedFlag = true;
        }
        
        const qualiaStalksMap = this.qualiaSheaf.stalks;
        let finalQualiaArray = new Float32Array(this.expectedQualiaInputLength);

        if (qualiaStalksMap instanceof Map && qualiaStalksMap.size > 0) {
            let offset = 0;
            for (const vertexName of this.qualiaSheaf.graph.vertices) {
                let stalk = qualiaStalksMap.get(vertexName);
                if (!stalk || !isFiniteVector(stalk) || stalk.length !== this.qDim) {
                    logger.warn(`OWM.predict: Corrupted or missing stalk for vertex "${vertexName}" in qualiaStalksMap. Replacing with zeros.`);
                    stalk = vecZeros(this.qDim); 
                    corruptedFlag = true;
                }
                
                for (let i = 0; i < this.qDim; i++) {
                    finalQualiaArray[offset + i] = Number.isFinite(stalk[i]) ? clamp(stalk[i], -1, 1) : 0;
                }
                offset += this.qDim;
            }
        } else {
            logger.warn('qualiaStalksMap is empty or not a Map, using all-zero qualia vector for input. Flagging as corrupted.');
            corruptedFlag = true;
        }

        if (!isFiniteVector(finalQualiaArray) || finalQualiaArray.length !== this.expectedQualiaInputLength) {
             logger.error(`FATAL: finalQualiaArray became non-finite or wrong length after construction. Resetting to all zeros and flagging corrupted.`);
             finalQualiaArray = vecZeros(this.expectedQualiaInputLength);
             corruptedFlag = true;
        }

        const input = new Float32Array([...state, ...finalQualiaArray]);
        
        if (input.length !== this.inputDim || !isFiniteVector(input)) {
             logger.error(`FATAL: Input size mismatch or non-finite values just before attention. Expected ${this.inputDim}, got ${input.length}. Flagging corrupted.`);
             return { qValues: vecZeros(this.actionDim), activations: [], corrupted: true };
        }

        const attended = this.applyAttention(input);
        
        let qValues, nextState, activations;
        try {
            const forwardResult = await this.forward(attended);
            qValues = forwardResult.qValues;
            nextState = forwardResult.nextState;
            activations = forwardResult.activations;
        } catch (e) {
            logger.error('OWM.predict: Error during forward pass:', e);
            corruptedFlag = true;
            qValues = vecZeros(this.actionDim);
            nextState = vecZeros(this.stateDim);
            activations = [];
        }

        const error = norm2(vecSub(nextState, state));
        this.predictionError = clamp(error, 0, 10);
        if (!Number.isFinite(this.predictionError)) { logger.warn('OWM.predict: Prediction error is non-finite. Resetting to 0.'); this.predictionError = 0; corruptedFlag = true; }

        this.freeEnergy = 0.85 * (this.freeEnergy || 0) + 0.15 * (this.predictionError * 0.5 + (this.qualiaSheaf.h1Dimension || 0));
        this.freeEnergy = clamp(this.freeEnergy, 0, 10);
        if (!Number.isFinite(this.freeEnergy)) { logger.warn('OWM.predict: Free energy is non-finite. Resetting to 0.'); this.freeEnergy = 0; corruptedFlag = true; }

        this.lastActivations = activations;

        return { qValues, activations, corrupted: corruptedFlag };
    }

    async learn(state, actionIndex, reward, nextState, isDone, learningRate = 0.01, gamma = 0.99) {
        if (!isFiniteVector(state) || !isFiniteVector(nextState) || !Number.isFinite(reward) || !Number.isFinite(actionIndex)) {
            logger.warn('OWM.learn: Invalid input (state, nextState, reward, or actionIndex). Skipping learning step.');
            return;
        }
        if (actionIndex < 0 || actionIndex >= this.actionDim) {
            logger.warn(`OWM.learn: Invalid actionIndex ${actionIndex}. Skipping learning step.`);
            return;
        }

        const { qValues: currentQValues } = await this.predict(state);
        const { qValues: nextQValues } = await this.predict(nextState);
        
        const maxNextQ = isDone ? 0 : Math.max(...nextQValues);

        const targetQ = (Number.isFinite(reward) ? reward : 0) + (Number.isFinite(gamma) ? gamma : 0) * (Number.isFinite(maxNextQ) ? maxNextQ : 0);
        const currentQForAction = (currentQValues && Number.isFinite(currentQValues[actionIndex])) ? currentQValues[actionIndex] : 0;
        const tdError = targetQ - currentQForAction;

        const lastRecurrentState = this.lastActivations[2];
        if (lastRecurrentState && isFiniteVector(lastRecurrentState)) {
            for (let j = 0; j < lastRecurrentState.length; j++) {
                const deltaW = (Number.isFinite(learningRate) ? learningRate : 0) * (Number.isFinite(tdError) ? tdError : 0) * (Number.isFinite(lastRecurrentState[j]) ? lastRecurrentState[j] : 0);
                if (Number.isFinite(deltaW)) {
                    this.qValueHead.W[actionIndex][j] = (Number.isFinite(this.qValueHead.W[actionIndex][j]) ? this.qValueHead.W[actionIndex][j] : 0) + deltaW;
                }
            }
            const deltaB = (Number.isFinite(learningRate) ? learningRate : 0) * (Number.isFinite(tdError) ? tdError : 0);
            if (Number.isFinite(deltaB)) {
                this.qValueHead.b[actionIndex] = (Number.isFinite(this.qValueHead.b[actionIndex]) ? this.qValueHead.b[actionIndex] : 0) + deltaB;
            }
        } else {
            logger.warn('OWM.learn: lastRecurrentState is invalid or missing. Skipping learning update.');
        }
    }

    resetRecurrentState() {
        this.hiddenState.fill(0);
        this.cellState.fill(0);
        logger.info(`OWM recurrent state for ${this.isPlayerTwo ? 'AI' : 'Player'} has been reset.`);
    }
}
// --- END OF FILE owm.js ---