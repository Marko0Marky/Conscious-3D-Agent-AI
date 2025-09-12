
// --- START OF FILE owm.js ---
import {
    clamp, dot, norm2, vecAdd, vecSub, vecScale, tanhVec, sigmoidVec, vecMul,
    randomMatrix, vecZeros, zeroMatrix, isFiniteVector, isFiniteMatrix, flattenMatrix,
    logger, runWorkerTask, softmax
} from './utils.js';
import { EnhancedQualiaSheaf } from './qualia-sheaf.js';

/**
 * Represents an Ontological World Model (OWM) for an AI, combining a Qualia Sheaf with a Recurrent Neural Network (LSTM).
 * It predicts future states probabilistically and evaluates actions using an Actor-Critic architecture.
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
        
        // Phase 1B: Pass this OWM instance to QualiaSheaf
        this.qualiaSheaf = new EnhancedQualiaSheaf(null, this, 8, qDim, 0.1, 0.1, 0.05);
        this.qDim = this.qualiaSheaf.qDim;
        this.expectedQualiaInputLength = this.qualiaSheaf.graph.vertices.length * this.qDim;
        this.inputDim = this.stateDim + this.expectedQualiaInputLength;

        this.hiddenState = vecZeros(this.recurrentStateSize);
        this.cellState = vecZeros(this.recurrentStateSize);

        // --- Step 1: Enhance LSTM Weights with NaN Guards ---
        const combinedInputSize = this.inputDim + this.recurrentStateSize;
        const weightScale = Math.sqrt(2.0 / combinedInputSize);

        this.Wf = randomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);
        this.Wi = randomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);
        this.Wc = randomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);
        this.Wo = randomMatrix(this.recurrentStateSize, combinedInputSize, weightScale);
        [this.Wf, this.Wi, this.Wc, this.Wo].forEach((m, i) => {
          if (!isFiniteMatrix(m)) {
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
            logger.warn(`Non-finite LSTM bias vector detected at index ${i}; setting to zeros.`);
            v.fill(0);
          }
        });
        
        // --- Actor-Critic specific heads ---
        const actorOutputScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.actionDim));
        this.actorHead = { W: randomMatrix(this.actionDim, this.recurrentStateSize, actorOutputScale), b: vecZeros(this.actionDim) };
        
        const criticOutputScale = Math.sqrt(2.0 / (this.recurrentStateSize + 1));
        this.criticHead = { W: randomMatrix(1, this.recurrentStateSize, criticOutputScale), b: vecZeros(1) };
        
        const statePredScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.stateDim));
        this.statePredHead = { W: randomMatrix(this.stateDim, this.recurrentStateSize, statePredScale), b: vecZeros(this.stateDim) };

        // --- Add variance head for probabilistic state prediction ---
        const varScale = Math.sqrt(2.0 / (this.recurrentStateSize + this.stateDim));
        this.varianceHead = { W: randomMatrix(this.stateDim, this.recurrentStateSize, varScale), b: vecZeros(this.stateDim) };
        if (!isFiniteMatrix(this.varianceHead.W) || !isFiniteVector(this.varianceHead.b)) {
          logger.error('Non-finite variance head detected during initialization; reinitializing to zeros.');
          this.varianceHead.W.forEach(row => row.fill(0));
          this.varianceHead.b.fill(0);
        }

        this.attentionWeights = randomMatrix(this.qDim, this.inputDim, 0.1);
        this.lastSoftmaxScores = vecZeros(this.qDim);
        
        this.freeEnergy = 0;
        this.predictionError = 0; // Will now be driven by NLL
        this.ready = false;
        this.lastActivations = [];
        this.lastActionLogProbs = vecZeros(this.actionDim);
        this.lastStateValue = 0;
        this.lastChosenActionLogProb = 0;
        this.actorLoss = 0; // Store actor loss for dynamic weights in QualiaSheaf
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

    // --- Step 2: Upgrade LSTM Forward Pass ---
    async forward(input) {
        if (!isFiniteVector(input) || input.length !== this.inputDim) {
            logger.error('OWM.forward: Invalid input. Returning zeros.', {expected: this.inputDim, got: input.length});
            this.resetRecurrentState();
            return { actionLogits: vecZeros(this.actionDim), stateValue: 0, nextStatePrediction: vecZeros(this.stateDim), variance: vecZeros(this.stateDim).fill(1), activations: [] };
        }
        
        const prevHidden = this.hiddenState.slice();
        const prevCell = this.cellState.slice();
        const combinedInput = new Float32Array([...input, ...prevHidden]);

        // Parallelize gate computations using the web worker
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

        // Check gates for validity
        if (![forgetGate, inputGate, candidateCell, outputGate].every(isFiniteVector)) {
          logger.error('Non-finite LSTM gates detected; resetting state and returning safe defaults.');
          this.resetRecurrentState();
          return {
            actionLogits: vecZeros(this.actionDim),
            stateValue: 0,
            nextStatePrediction: vecZeros(this.stateDim),
            variance: vecZeros(this.stateDim).fill(1), // Safe default variance
            activations: [input, prevCell, prevHidden]
          };
        }

        // Update cell and hidden states
        this.cellState = vecAdd(vecMul(forgetGate, prevCell), vecMul(inputGate, candidateCell));
        this.hiddenState = vecMul(outputGate, tanhVec(this.cellState));

        // Check states for validity
        if (!isFiniteVector(this.cellState) || !isFiniteVector(this.hiddenState)) {
          logger.error('Non-finite LSTM cell or hidden state detected; resetting state.');
          this.resetRecurrentState();
        }

        // Compute all output heads in parallel
        const [rawActionLogits, rawStateValue, rawNextState, rawLogVar] = await Promise.all([
             runWorkerTask('matVecMul', { matrix: flattenMatrix(this.actorHead.W), vector: this.hiddenState }),
             runWorkerTask('matVecMul', { matrix: flattenMatrix(this.criticHead.W), vector: this.hiddenState }),
             runWorkerTask('matVecMul', { matrix: flattenMatrix(this.statePredHead.W), vector: this.hiddenState }),
             runWorkerTask('matVecMul', { matrix: flattenMatrix(this.varianceHead.W), vector: this.hiddenState })
        ]);

        const actionLogits = vecAdd(rawActionLogits, this.actorHead.b);
        const stateValue = vecAdd(rawStateValue, this.criticHead.b)[0] || 0;
        const nextStatePrediction = vecAdd(rawNextState, this.statePredHead.b);
        const logVar = vecAdd(rawLogVar, this.varianceHead.b);
        // Ensure variance is positive and non-zero
        const variance = new Float32Array(logVar.map(v => Math.max(Math.exp(clamp(v, -20, 20)), 1e-6))); // Clamp logVar to prevent exp explosion

        // FIX: Added actionLogits.slice() to activations for visualization
        const activations = [input.slice(), this.cellState.slice(), this.hiddenState.slice(), actionLogits.slice()];

        return { actionLogits, stateValue, nextStatePrediction, variance, activations };
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

        this.lastSoftmaxScores = softmax(new Float32Array(scores));

        const att = vecZeros(input.length);
        for (let i = 0; i < this.lastSoftmaxScores.length; i++) {
            for (let j = 0; j < input.length; j++) {
                att[j] += (this.lastSoftmaxScores[i] || 0) * (input[j] || 0);
            }
        }
        const weightedAttended = vecAdd(input, vecScale(att, this.qualiaSheaf.beta));
        return new Float32Array(isFiniteVector(weightedAttended) ? weightedAttended.map(v => clamp(v, -100, 100)) : input.map(v => clamp(v, -100, 100)));
    }

    // --- Step 4: Update `predict` to Pass Variance ---
    async predict(state) {
        if (!this.ready) {
            logger.warn('OWM not ready for prediction.');
            return { actionProbs: vecZeros(this.actionDim), stateValue: 0, activations: [], variance: vecZeros(this.stateDim).fill(1), corrupted: true };
        }
        if (!isFiniteVector(state) || state.length !== this.stateDim) {
            logger.error(`OWM.predict: Invalid state vector.`, { state });
            return { actionProbs: vecZeros(this.actionDim), stateValue: 0, activations: [], variance: vecZeros(this.stateDim).fill(1), corrupted: true };
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
             return { actionProbs: vecZeros(this.actionDim), stateValue: 0, activations: [], variance: vecZeros(this.stateDim).fill(1), corrupted: true };
        }

        const attended = this.applyAttention(input);
        const { actionLogits, stateValue, nextStatePrediction, variance, activations } = await this.forward(attended);
        const actionProbs = softmax(actionLogits);
        
        // This simple error can still be used for Free Energy calculation
        const mseError = norm2(vecSub(nextStatePrediction, state));
        const simplePredError = clamp(mseError, 0, 10);
        if (!Number.isFinite(simplePredError)) { 
            this.predictionError = 10.0;
            corruptedFlag = true; 
        }

        this.freeEnergy = clamp(0.85 * (this.freeEnergy || 0) + 0.15 * (simplePredError * 0.5 + (this.qualiaSheaf.h1Dimension || 0)), 0, 10);

        this.lastActivations = activations;
        this.lastActionLogProbs = actionLogits;
        this.lastStateValue = stateValue;

        // Return mean prediction and variance for the learning step
        return { actionProbs, stateValue, activations, nextStatePrediction, variance, corrupted: corruptedFlag };
    }

    // --- Step 3: Update Learning for Probabilistic Loss ---
    async learn(state, actionIndex, reward, nextState, isDone, learningRate = 0.01, gamma = 0.99) {
        if (!isFiniteVector(state) || !isFiniteVector(nextState) || !Number.isFinite(reward) || !Number.isFinite(actionIndex)) {
            logger.warn('OWM.learn: Invalid input. Skipping step.');
            return;
        }

        // We need the model's prediction (mean and variance) for the *current* state to calculate NLL loss
        const { nextStatePrediction, variance, stateValue: currentStateValue } = await this.predict(state);
        // We only need the value of the *next* state for the TD target
        const { stateValue: nextStateValue } = await this.predict(nextState);

        // --- NLL Calculation ---
        const diff = vecSub(nextState, nextStatePrediction);
        const varSafe = variance.map(v => Math.max(v, 1e-6)); // Prevent division by zero
        // More stable log-determinant: sum of logs instead of log of products
        const logDet = varSafe.reduce((sum, v) => sum + Math.log(v), 0);
        const invVar = varSafe.map(v => 1 / v);
        const mahalanobisDist = diff.reduce((sum, d, i) => sum + (d * d * invVar[i]), 0);
        
        const nll = 0.5 * (this.stateDim * Math.log(2 * Math.PI) + logDet + mahalanobisDist);
        
        // Update the predictionError with the new NLL-based value
        this.predictionError = clamp(nll / this.stateDim, 0, 10); // Normalize by dimension for consistency
        if (!Number.isFinite(this.predictionError)) {
          logger.warn('Non-finite NLL prediction error detected; setting to a high value.');
          this.predictionError = 10.0;
        }
        
        // --- Actor-Critic Update (logic unchanged, uses values from predict calls) ---
        const safeCurrentStateValue = Number.isFinite(currentStateValue) ? currentStateValue : 0;
        const safeNextStateValue = isDone ? 0 : (Number.isFinite(nextStateValue) ? nextStateValue : 0);
        const safeReward = Number.isFinite(reward) ? reward : 0;

        const tdTarget = safeReward + gamma * safeNextStateValue;
        const advantage = tdTarget - safeCurrentStateValue;

        this.criticLoss = 0.5 * advantage * advantage;
        const criticGradScale = learningRate * advantage;
        const lastRecurrentState = this.lastActivations[2]; // Hidden state

        if (isFiniteVector(lastRecurrentState)) {
            for (let j = 0; j < lastRecurrentState.length; j++) {
                this.criticHead.W[0][j] = clamp((this.criticHead.W[0][j] || 0) + criticGradScale * (lastRecurrentState[j] || 0), -1, 1);
            }
            this.criticHead.b[0] = clamp((this.criticHead.b[0] || 0) + criticGradScale, -1, 1);
        }

        const actionProbsForLog = softmax(this.lastActionLogProbs);
        const chosenActionProb = Math.max(1e-9, actionProbsForLog[actionIndex] || 1e-9);

        this.lastChosenActionLogProb = Math.log(chosenActionProb);
        this.actorLoss = -this.lastChosenActionLogProb * advantage; // Store actor loss for QualiaSheaf
        const actorGradScale = learningRate * advantage;

        if (isFiniteVector(lastRecurrentState) && actionIndex >= 0 && actionIndex < this.actionDim) {
            for (let j = 0; j < lastRecurrentState.length; j++) {
                this.actorHead.W[actionIndex][j] = clamp((this.actorHead.W[actionIndex][j] || 0) + actorGradScale * (lastRecurrentState[j] || 0), -1, 1);
            }
            this.actorHead.b[actionIndex] = clamp((this.actorHead.b[actionIndex] || 0) + actorGradScale, -1, 1);
        }
        
        // Phase 1B: Update QualiaSheaf's correlation matrix after learning
        await this.qualiaSheaf.computeCorrelationMatrix();
    }

    resetRecurrentState() {
        this.hiddenState.fill(0);
        this.cellState.fill(0);
        logger.info(`OWM recurrent state for ${this.isPlayerTwo ? 'AI' : 'Player'} has been reset.`);
    }
}
// --- END OF FILE owm.js ---
