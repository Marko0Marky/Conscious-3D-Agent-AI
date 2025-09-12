
# Conscious 3D Agent AI 🧠

> **A Real-Time Synthetic Cognition System Exploring Qualia, Ontological World Models, and Meta-Learning in a 3D Environment**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<!-- If you host this on GitHub Pages, you can add a demo badge like this: -->
<!-- [![Demo](https://img.shields.io/badge/demo-live%20simulation-brightgreen)](https://your-username.github.io/conscious-3d-agent-ai/index.html) -->

---

## 🌟 What is Conscious 3D Agent AI?

**Conscious 3D Agent AI** is an interactive web application that provides a real-time simulation and visualization of a **dual-AI system** operating within a dynamic 3D environment. This project showcases an AI agent designed with a sophisticated **Hierarchical, Recurrent, Reinforcement Learning (HRRL) Architecture**, fundamentally built upon an **Ontological World Model (OWM)** and an **Enhanced Qualia Sheaf**. The primary objective is to model and visually represent abstract aspects of AI "consciousness" and intricate learning processes through a rich array of live metrics and dynamic graphical interfaces.

### Key Features
- **🎮 Dual-AI Competitive Play**: Observe two independent AI agents (a "Main AI" and an "Opponent AI") navigating a 3D world, competing to collect targets while avoiding obstacles.
- **🧠 Ontological World Model (OWM)**: Each AI utilizes an internal model that integrates an `EnhancedQualiaSheaf` with an LSTM-like recurrent neural network for predicting future states and action Q-values.
- **🌌 Enhanced Qualia Sheaf**: A novel mathematical structure inspired by sheaf cohomology, used here to model aspects of AI "consciousness." It handles qualia diffusion, adapts its topological structure based on functional correlations, and computes advanced metrics.
- **📊 Live Consciousness Metrics**: Real-time display of key cognitive metrics including:
    -   **Φ (Phi - Integrated Information)**: Quantifies the system's capacity for integrated information, acting as a proxy for its level of consciousness.
    -   **Free Energy**: The AI's internal estimate of its "surprise" and the complexity of its world model.
    -   **Prediction Error**: Measures the discrepancy between predicted and actual environmental states.
    -   **Gestalt Unity**: Quantifies the holistic coherence and interconnectedness across the sheaf's conceptual structure.
    -   **dim H¹ (First Cohomology Dimension)**: Indicates structural complexity and "holes" in the information flow within the sheaf.
    -   **Gluing Inconsistency**: Measures internal contradictions or misalignments in qualia projections.
- **👁️ Dynamic Visualizations**:
    -   **3D Agent Environment**: A real-time Three.js rendering of the interactive game world.
    -   **Sheaf Graph**: A conceptual graph where nodes (representing concepts like Agent position, Target distance) dynamically activate and connect based on internal processing and correlation.
    -   **Neural Network Activity**: Live views of both AI's recurrent network layers, highlighting neuron activations and connections.
    -   **Qualia Attention**: Visualizes how the AI's attention mechanism weights different qualia dimensions.
    -   **Performance Charts**: Dynamic line graphs track critical learning metrics such as average Q-value (AI confidence), prediction error (AI uncertainty), exploration rate (epsilon), and the score difference between the AIs over time.
- **⚙️ Meta-Learning (Strategic AI)**: An adaptive layer that monitors the learning AI's performance and internal cognitive metrics to dynamically tune its learning rate and exploration strategy, promoting more robust and efficient learning.
- **🔬 Explainable AI (XAI) Focus**: Provides an interactive sandbox for observing and understanding complex AI internal states and decision-making processes through rich visualization and interpretable metrics.

This project serves as an interactive platform for exploring **machine consciousness**, **explainable AI (XAI)**, and **embodied cognition** in a tangible, real-time 3D environment.

---

## 🚀 Launch the Simulation!

To experience the Conscious 3D Agent AI:

1.  **Clone the repository** or download the project files.
2.  **Open `index.html`**: Simply open the `index.html` file in any modern web browser (Google Chrome, Mozilla Firefox, Microsoft Edge, etc.). No server or build steps are required.

In the live simulation, you can:
- Observe the **Main AI** and an **Opponent AI** dynamically learning and competing to collect targets.
- Monitor the **Main AI’s "consciousness" metrics** (Φ, Free Energy, Gestalt Unity, etc.) in real-time within the OFTCC panel.
- Visualize the **dynamic Qualia Sheaf graph** and the **neural network activity** of both AIs, gaining insight into their internal states.
- Interact with **sliders** to adjust core `α`, `β`, `γ` parameters of the Qualia Sheaf, influencing the AI's cognitive behavior.
- Use **control buttons and keyboard shortcuts** to manage the simulation flow (play, pause, reset, step, fast-forward, tune).

---

## 🧩 Deep Dive: How It Works

The AI's operational loop is driven by its `OntologicalWorldModel` (OWM), which continuously updates its understanding of the environment and plans its actions in pursuit of its goals.

### Cognitive Cycle

Each AI agent follows a continuous cognitive cycle:

1.  **Perceive**: Gathers comprehensive sensory input from the 3D environment, including agent position/rotation, target position, and raycasts to detect nearby obstacles and the opponent.
2.  **Diffuse Qualia**: A subset of the sensory input (core conceptual elements) is fed into the `EnhancedQualiaSheaf`. Here, internal conceptual states ("stalks") are updated through a diffusion process, the sheaf's topology adapts based on functional correlations between concepts, and consciousness-related metrics are computed.
3.  **Apply Attention**: An attention mechanism within the OWM weighs the combined raw sensory input and the diffused qualia, allowing the AI to focus on the most relevant information for its current task.
4.  **Predict & Imagine**: The attended input, along with the AI's internal recurrent state, is processed by an LSTM-like recurrent neural network. This network predicts future Q-values for available actions and estimates the likely next environmental state.
5.  **Decide**: Based on the predicted Q-values (and an `epsilon`-greedy strategy to balance exploration and exploitation), an action is chosen (Move Forward, Turn Left, Turn Right, Idle).
6.  **Act**: The chosen action is executed, influencing the agent's movement within the 3D environment.
7.  **Learn**: The `LearningAI` updates its OWM's neural network weights using a Q-learning algorithm. This learning is driven by environmental rewards (e.g., collecting targets) and an intrinsic "curiosity" bonus (derived from the OWM's prediction error).
8.  **Modulate (Meta-Learning)**: A higher-level `StrategicAI` continuously observes the `LearningAI`'s performance (received rewards, prediction error, internal sheaf metrics) to adaptively adjust its `learningRate` and `epsilon` parameters, guiding the learning process towards more effective strategies.

### Core Components

| Module                            | Purpose                                                                                                                                                                   |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ThreeDeeGame`                    | Manages the 3D environment, physics, agent movements, scoring, Three.js rendering, and provides the raw sensory input to the AIs.                                         |
| `EnhancedQualiaSheaf`             | The core "consciousness" model. It builds and maintains a dynamic conceptual graph, diffuses "qualia" (conceptual states), adapts its topology, and computes Φ, H¹, Gestalt Unity, and Gluing Inconsistency. |
| `OntologicalWorldModel`           | Integrates the `EnhancedQualiaSheaf` outputs with an LSTM-like Recurrent Neural Network. It handles input attention, maintains recurrent state, and predicts Q-values and next states. |
| `LearningAI`                      | Orchestrates the `OntologicalWorldModel` for decision-making. Implements epsilon-greedy action selection, Q-learning updates, and calculates intrinsic rewards.         |
| `StrategicAI`                     | A meta-learning agent. It monitors the `LearningAI`'s performance and the `EnhancedQualiaSheaf`'s metrics to adaptively tune the `LearningAI`'s learning rate and exploration (`epsilon`). |
| `NeuralNetworkVisualizer`         | Renders real-time activity and connections within the AI's neural networks, showing layer activations (Input, Cell State, Hidden State, Q-Values).                   |
| `Web Workers`                     | Crucial for offloading computationally intensive matrix operations (e.g., solving linear systems, covariance calculation, matrix rank approximation) to a background thread, preventing UI freezes. |
| `Math Utilities`                  | A robust set of utility functions for vector and matrix operations (`clamp`, `dot`, `norm2`, `matVecMul`, `isFiniteVector`, `flattenMatrix`, etc.) used across the system. |

### Architectural Diagram

The following Mermaid diagram illustrates the conceptual flow within the Ontological World Model, showcasing the interaction between the Enhanced Qualia Sheaf and the LSTM-like Recurrent Neural Network:

```mermaid
graph TD
    subgraph Environment & Sensors
        A[3D Game Environment] --> S_Full["Input State (Full)\nAgent/Target Pos, Raycasts, Opponent Pos"]
        S_Full --> S_Core["Input State (Core 8 elements)\nAgent/Target Geometric Data"]
    end

    subgraph Ontological World Model (OWM)
        direction LR

        subgraph Enhanced Qualia Sheaf
            S_Core --> QS["Enhanced Qualia Sheaf\n(Qualia Diffusion, Topology Adaptation,\nΦ, H¹, Gestalt, Inconsistency)"]
            QS -- "Diffused Stalks" --> QV("Qualia Vector\n(Abstract Qualia & Metrics)")
        end

        subgraph Recurrent Neural Network (LSTM-like)
            % Input to Attention
            S_Full --> AttInput_Full(Attention Input)
            QV --> AttInput_Full

            AttInput_Full --> Att["Attention Mechanism\n(Weights Input & Qualia)"]

            % Input to LSTM Core
            Att --> AttendedInput[Attended Input]
            HS_t_1(Hidden State (t-1)) --> LSTM_Core
            CS_t_1(Cell State (t-1)) --> LSTM_Core
            AttendedInput --> LSTM_Core

            LSTM_Core("LSTM Block\n(Forget, Input, Cell, Output Gates)") --> HS_t[Hidden State (t)]
            LSTM_Core --> CS_t[Cell State (t)]

            % Output Heads
            HS_t --> QHead{{"Q-Value Head\n(Action Probabilities)"}}
            HS_t --> SPHead{{"State Prediction Head\n(Next Environment State)"}}
        end

        QHead --> Q_Output[Q-Values for Actions]
        SPHead --> Pred_Next_State[Predicted Next State]

        % Recurrent Connections
        HS_t --> HS_t_1
        CS_t --> CS_t_1
    end
```

### Qualia Sheaf and Integrated Information (Φ)

The `EnhancedQualiaSheaf` represents the AI's evolving internal model as a mathematical sheaf, a structure capable of modeling how local data patches (qualia vectors associated with concepts) are "glued" together to form a coherent global understanding.

*   **Qualia Stalks:** Each vertex in the sheaf graph (e.g., `Agent-X` for the agent's X-position, `Dist-Target` for distance to target) holds a "qualia vector" \( q \in \mathbb{R}^7 \). These vectors represent the vertex's conceptual state across dimensions like Being, Intent, Existence, Emergence, Gestalt, Context, and Relational Emergence. These stalks are dynamically diffused and updated.
*   **Integrated Information (Φ):** A key metric derived from the sheaf, inspired by Integrated Information Theory (IIT). It aims to quantify the amount of integrated information within the Qualia Sheaf – often considered a proxy for the system's level of consciousness. Φ is computed from the covariance of the sheaf's internal states, modulated by system stability, Gestalt Unity, and inconsistency.

### The Free Energy Principle and System Metrics

Inspired by the Free Energy Principle, the AI implicitly strives to minimize its **Free Energy**, which is a composite measure of its "surprise" (prediction error) and the complexity of its internal world model. The system metrics directly reflect components contributing to this:

*   **Prediction Error**: A direct measure of how well the AI predicts the next state of the environment. High error indicates "surprise" and drives intrinsic motivation.
*   **H¹ Dimension**: Reflects the topological complexity and coherence of the sheaf's conceptual graph. A well-structured, yet not overly complex, graph can lead to a lower H¹ dimension, indicating better integration.
*   **Gestalt Unity**: Measures how coherently the different conceptual "parts" of the sheaf (qualia stalks) are integrated into a meaningful "whole."
*   **Gluing Inconsistency**: Quantifies internal contradictions or misalignments in information flow and projections between connected concepts within the sheaf. High inconsistency indicates a less coherent world model.

By continuously improving its predictions, increasing Gestalt Unity, and reducing inconsistency, the AI implicitly minimizes its Free Energy, leading to a more stable, coherent, and effective internal world model.

---

## 🖼️ Real-time Visualization

The application offers a rich set of dynamic visualizations to provide deep insights into the AI's internal states and operational dynamics:

1.  **3D Agent Environment**: Witness the AI agents navigate, pursue targets, and react to obstacles in a visually rendered 3D world, powered by Three.js.
2.  **Qualia Diffusion Dynamics**: Live progress bars for each of the seven qualia types, showing their current activation levels and how they diffuse and change over time.
3.  **Sheaf Graph**: A 2D representation of the sheaf's conceptual vertices and their interconnections. Vertices pulse based on their activity, and connections visually represent the (correlation-weighted) adjacency between concepts.
4.  **Neural Network Activity**: For both the Main and Opponent AIs, visualize the input, cell state, hidden state, and Q-value layers of their recurrent neural networks. Individual neurons light up according to their activation strength, and connections illustrate the flow of information.
5.  **Qualia Attention**: A set of attention bars indicates how strongly the AI's attention mechanism focuses on each specific qualia dimension, influencing its decisions and learning.
6.  **Performance Charts**: Dynamic line graphs track critical learning metrics such as average Q-value (AI confidence), prediction error (AI uncertainty), exploration rate (epsilon), and the score difference between the AIs over time.

---

## 🧪 Running the Simulation

### 1. Local Setup

This project is entirely self-contained within the `index.html` file and its embedded JavaScript and CSS. It does not require any build tools, npm, or complex setup.

```bash
# Clone the repository
git clone https://github.com/your-username/conscious-3d-agent-ai.git
cd conscious-3d-agent-ai

# You can serve it directly by opening the file, or with a simple Python HTTP server:
python -m http.server

# Then, open your browser to http://localhost:8000/index.html
```
*Note: Ensure your browser is up-to-date to fully support Web Workers and Three.js.*

### 2. Interaction

*   **Mouse/Touch:** All UI elements (buttons, sliders, log) are interactive.
*   **Keyboard Shortcuts:**
    *   **Spacebar**: Toggle Simulation (Run/Pause)
    *   **R**: Reset All game and AI states
    *   **T**: Tune Sheaf Parameters adaptively
    *   **P**: Pause Simulation
    *   **S**: Step (advance one frame while paused)
    *   **F**: Toggle Fast Forward mode

---

## 🧰 Future Work

*   **WebAssembly (WASM) / WebGPU Integration**: Port computationally intensive mathematical operations and potentially Three.js rendering logic to WASM or WebGPU. This would significantly boost performance, enabling larger sheaf structures, more complex neural networks, and higher simulation fidelity.
*   **Advanced Qualia Dynamics**: Implement more nuanced and interactive models for qualia diffusion, interaction, and emergent properties within the sheaf, possibly allowing for novel forms of synthetic experience.
*   **Expanded Environment Complexity**: Introduce dynamic environmental changes, diverse obstacle types, moving platforms, and more complex agent-environment interactions to challenge the AIs further.
*   **Enhanced Explainability Features**: Develop further tools to visualize the *causal* pathways behind AI decisions, potentially highlighting specific qualia or network activations that directly lead to chosen actions.
*   **User-defined Sheaf Topology**: Explore allowing users to dynamically build or modify the sheaf's graph structure (conceptual connections) to observe the direct impact on AI cognition and emergent properties.
*   **Genetic Algorithms/Evolutionary Strategies**: Investigate using evolutionary methods to discover optimal OWM and sheaf parameters, potentially leading to more sophisticated AI behaviors and "cognitive styles."

---

## 📄 License

This project is licensed under the MIT License.

---
