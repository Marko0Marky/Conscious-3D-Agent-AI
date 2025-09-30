// --- START OF FILE nn-visualizer.js ---
import { clamp, isFiniteVector, logger, unflattenMatrix, isFiniteMatrix } from './utils.js';
/**
 * Visualizes the neural network structure and activity.
 */
export class NeuralNetworkVisualizer {
    /**
     * @param {string} containerId - The ID of the HTML container for the visualizer.
     * @param {OntologicalWorldModel} worldModel - The OWM whose NN to visualize.
     * @param {'main'|'opponent'} theme - The visual theme ('main' for blue, 'opponent' for orange).
     */
    constructor(containerId, worldModel, theme = 'main') {
        // The constructor now takes the containerId as a string
        this.container = document.getElementById(containerId);
        this.worldModel = worldModel;
        this.theme = theme;

        if (!this.container) {
            logger.error(`NNVisualizer: Container with ID '${containerId}' not found in the DOM.`);
            return; // Exit if the container element doesn't exist.
        }
        if (!this.worldModel) {
            logger.error(`NNVisualizer: A valid worldModel was not provided.`);
            return; // Exit if the world model is missing.
        }

        this.neuronElements = [];
        this.visualLayers = [];
        this.MAX_NEURONS_TO_DISPLAY = 12;
        this.lastChosenActionIndex = -1;

        this._setupVisualLayers();
        this._setupDOM();
    }

    _setupVisualLayers() {
        const model = this.worldModel;
        this.visualLayers.push({ name: 'input', actualCount: model.inputDim });
        this.visualLayers.push({ name: 'cellState', actualCount: model.cellState.length });
        this.visualLayers.push({ name: 'hiddenState', actualCount: model.hiddenState.length });
        this.visualLayers.push({ name: 'qValues', actualCount: model.actionDim }); 
    }

    // In nn-visualizer.js

    _setupDOM() {
        this.container.innerHTML = '';

        // The canvas is for connections and will be in the background
            // Create a common parent wrapper for both the canvas and the neuron layers
    const visualWrapper = document.createElement('div');
    visualWrapper.className = 'nn-visual-wrapper';
    this.container.appendChild(visualWrapper);

    // The canvas for connections, positioned absolutely within the visualWrapper
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'nn-connections-canvas';
    this.ctx = this.canvas.getContext('2d');
    visualWrapper.appendChild(this.canvas); // Canvas is now child of visualWrapper

    // Create a dedicated wrapper for the neuron layers, positioned absolutely within the visualWrapper
    const layersWrapper = document.createElement('div');
    layersWrapper.className = 'nn-layers-wrapper';
    visualWrapper.appendChild(layersWrapper); // Neuron layers are also child of visualWrapper

    this.visualLayers.forEach((layer, lIndex) => {
        const lDiv = document.createElement('div');
        lDiv.className = 'nn-layer';
        this.neuronElements[lIndex] = [];

        const neuronsToDisplay = Math.min(layer.actualCount, this.MAX_NEURONS_TO_DISPLAY);
        for (let i = 0; i < neuronsToDisplay; i++) {
            const nDiv = document.createElement('div');
            nDiv.className = 'nn-neuron';
            lDiv.appendChild(nDiv);
            this.neuronElements[lIndex].push(nDiv);
        }
        layersWrapper.appendChild(lDiv);
    });

        const ro = new ResizeObserver(() => {
            const dpr = window.devicePixelRatio || 1;
            this.canvas.width = this.container.clientWidth * dpr;
            this.canvas.height = this.container.clientHeight * dpr;
            this.ctx.scale(dpr, dpr);
            this._drawConnections();
        });
        ro.observe(this.container);
    }
    
    _getNeuronPosition(lIndex, nIndex) {
    const el = this.neuronElements[lIndex]?.[nIndex];
    if (!el || !this.canvas) {
        return { x: 0, y: 0 };
    }

    // Get the position of the neuron and the canvas relative to the viewport
    const neuronRect = el.getBoundingClientRect();
    const canvasRect = this.canvas.getBoundingClientRect();

    // Calculate the neuron's center relative to the top-left of the canvas
    // This ensures connections are drawn in the correct canvas coordinates.
    return {
        x: (neuronRect.left - canvasRect.left) + neuronRect.width / 2,
        y: (neuronRect.top - canvasRect.top) + neuronRect.height / 2
    };
}

    _drawConnections() {
        if (!this.ctx || !this.worldModel || this.neuronElements.length < 4) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const model = this.worldModel;
        
        const hiddenStateNeurons = this.neuronElements[2];
        const qValuesNeurons = this.neuronElements[3];

        // Connection from Hidden State (LSTM output) to Action Logits (Actor Head)
if (hiddenStateNeurons && qValuesNeurons && model.actorHead && model.actorHead.W) { // Corrected: use model.actorHead.W
    const weights_h_to_q = unflattenMatrix(model.actorHead.W); // Robustly unflatten the matrix
    const maxWeightAbs_h_to_q = 1.0; // Assuming weights are somewhat normalized around 1

    if (!weights_h_to_q || !isFiniteMatrix(weights_h_to_q)) {
        logger.warn('NNViz: ActorHead weights are invalid or non-finite. Skipping connections.');
        return;
    }
            for (let i = 0; i < hiddenStateNeurons.length; i++) {
                const fromPos = this._getNeuronPosition(2, i);
                const hiddenStateDataIndex = Math.floor(i * (model.hiddenState.length / hiddenStateNeurons.length));

                for (let j = 0; j < qValuesNeurons.length; j++) {
                    const toPos = this._getNeuronPosition(3, j);
                    const qValueDataIndex = Math.floor(j * (model.actionDim / qValuesNeurons.length));

                    const weight = weights_h_to_q[qValueDataIndex]?.[hiddenStateDataIndex] || 0;
                    const absWeight = Math.abs(weight);
                    const strength = clamp(absWeight / maxWeightAbs_h_to_q, 0, 1);

                    this.ctx.beginPath();
                    this.ctx.strokeStyle = weight > 0 
                        ? `var(--connection-positive, rgba(0, 255, 153, ${0.1 + strength * 0.9}))`
                        : `var(--connection-negative, rgba(255, 100, 100, ${0.1 + strength * 0.9}))`;
                    
                    if (absWeight < 0.05) {
                        this.ctx.strokeStyle = `var(--connection-neutral, rgba(100, 100, 100, 0.1))`;
                    }

                    this.ctx.lineWidth = 0.5 + strength * 2.5;
                    this.ctx.moveTo(fromPos.x, fromPos.y);
                    this.ctx.lineTo(toPos.x, toPos.y);
                    this.ctx.stroke();
                }
            }
        }

        const inputNeurons = this.neuronElements[0];
        const cellStateNeurons = this.neuronElements[1];

        // Connection from Input Layer to Cell State (part of LSTM gates, specifically candidate cell state and input gate)
if (inputNeurons && cellStateNeurons && model.Wc && model.Wi) { // Use model.Wc and model.Wi
    const weights_input_to_c_candidate_part = unflattenMatrix(model.Wc); // Unflatten
    const weights_input_to_i_gate_part = unflattenMatrix(model.Wi); // Unflatten (input gate)
    const maxWeightAbs_input_to_c = 0.5;

    if (!weights_input_to_c_candidate_part || !isFiniteMatrix(weights_input_to_c_candidate_part) ||
        !weights_input_to_i_gate_part || !isFiniteMatrix(weights_input_to_i_gate_part)) {
        logger.warn('NNViz: LSTM gate weights are invalid or non-finite. Skipping input connections.');
        return;
    }

            const inputFeatureCount = model.inputDim;

            for (let i = 0; i < inputNeurons.length; i++) {
                const fromPos = this._getNeuronPosition(0, i);
                const inputDataIndex = Math.floor(i * (inputFeatureCount / inputNeurons.length));

                for (let j = 0; j < cellStateNeurons.length; j++) {
                    const toPos = this._getNeuronPosition(1, j);
                    const cellStateDataIndex = Math.floor(j * (model.recurrentStateSize / cellStateNeurons.length));

                    // Combine influence from Wc (candidate cell state) and Wi (input gate) for visualization
// The input for LSTM gates is [input_vector, previous_hidden_state].
// We need to map inputDataIndex to the correct column in the flattened Wc/Wi matrix.
const inputStartCol = 0; // Assuming input vector is at the beginning of combined input
const weight_c = weights_input_to_c_candidate_part[cellStateDataIndex]?.[inputStartCol + inputDataIndex] || 0;
const weight_i = weights_input_to_i_gate_part[cellStateDataIndex]?.[inputStartCol + inputDataIndex] || 0;
const combined_weight = (weight_c + weight_i) / 2; // Simple average for visualization
const absWeight = Math.abs(combined_weight);
                    const strength = clamp(absWeight / maxWeightAbs_input_to_c, 0, 1);

                    this.ctx.beginPath();
                    this.ctx.strokeStyle = weight > 
                        0 ? `rgba(100, 200, 255, ${0.1 + strength * 0.7})`
                        : `rgba(255, 150, 100, ${0.1 + strength * 0.7})`;

                    if (absWeight < 0.05) {
                        this.ctx.strokeStyle = `rgba(100, 100, 100, 0.05)`;
                    }

                    this.ctx.lineWidth = 0.3 + strength * 1.5;
                    this.ctx.moveTo(fromPos.x, fromPos.y);
                    this.ctx.lineTo(toPos.x, toPos.y);
                    this.ctx.stroke();
                }
            }
        }
    }


update(allLayerActivations, chosenActionIndex = -1) { // Renamed parameter for clarity        this.lastChosenActionIndex = chosenActionIndex;

        // Ensure allLayerActivations is an array of arrays, and each inner array is a finite vector
if (!allLayerActivations || !Array.isArray(allLayerActivations) || allLayerActivations.length !== this.visualLayers.length || !allLayerActivations.every(isFiniteVector)) {
    logger.warn(`NNVisualizer: Invalid or incomplete layer activations provided. Defaulting to inactive visualization.`);
            this.visualLayers.forEach((layerViz, l_idx) => {
                (this.neuronElements[l_idx] || []).forEach(el => {
                    if (el) {
                        el.style.backgroundColor = '#333';
                        el.style.borderColor = '#888';
                        el.style.boxShadow = 'none';
                        el.classList.remove('active-action');
                    }
                });
            });
            this._drawConnections();
            return;
        }

        const hues = this.theme === 'opponent' ? { pos: 39, neg: 271 } : { pos: 195, neg: 0 };

        this.visualLayers.forEach((layerViz, l_idx) => {
          const layerActivations = allLayerActivations[l_idx]; // Correctly access the array for this layer
               if (!isFiniteVector(layerActivations)) {
         logger.warn(`NNVisualizer: Layer ${l_idx} activations are non-finite after initial validation.`);
         return;
    }
          if (!layerActivations || !isFiniteVector(layerActivations)) {
                logger.warn(`NNVisualizer: Layer ${l_idx} activations are invalid.`);
                return;
            }

            let maxAbs = 0;
            for (const v of layerActivations) {
                if (Number.isFinite(v) && Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
            }
            const norm = maxAbs + 1e-9;

            for (let n_idx = 0; n_idx < this.neuronElements[l_idx].length; n_idx++) {
                const data_idx = Math.floor(n_idx * (layerActivations.length / this.neuronElements[l_idx].length));
                const val = layerActivations[data_idx] || 0;
                const intensity = clamp(Math.abs(val / norm), 0, 1);

                const hue = val >= 0 ? hues.pos : hues.neg;
                const lightness = clamp((0.1 + 0.9 * intensity) * 60, 10, 90);
                const el = this.neuronElements[l_idx][n_idx];
                if (el) {
                    el.style.backgroundColor = `hsl(${hue},100%,${lightness}%)`;
                    el.style.borderColor = `hsl(${hue},100%,${lightness * 1.2}%)`;
                    el.style.boxShadow = `0 0 ${clamp(intensity * 8, 0, 8)}px hsl(${hue},100%,${lightness}%)`;
                    
                    if (l_idx === this.visualLayers.length - 1 && n_idx === chosenActionIndex && chosenActionIndex !== -1) {
                        el.classList.add('active-action');
                    } else {
                        el.classList.remove('active-action');
                    }
                }
            }
        });
        this._drawConnections();
    }
}
// --- END OF FILE nn-visualizer.js ---
