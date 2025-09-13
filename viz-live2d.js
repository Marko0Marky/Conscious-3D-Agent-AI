// --- START OF FILE viz-live2d.js ---
import { logger, clamp } from './utils.js';

let pixiApp;
let live2dModel;
let container;
let initialized = false;
let clock;

// FIX: EMOTION_MAP now uses indices 0-5 to match the incoming 6-element emotion vector.
const EMOTION_MAP = {
    0: { name: 'joy', expression: 'F01', param: 'ParamSmile' },
    1: { name: 'fear', expression: 'F04', param: 'ParamEyeOpen' },
    2: { name: 'curiosity', expression: 'F06', param: 'ParamBrowLAngle' },
    3: { name: 'frustration', expression: 'F05', param: 'ParamBrowLForm' },
    4: { name: 'calm', expression: null, param: 'ParamMouthOpenY' },
    5: { name: 'surprise', expression: 'F02', param: 'ParamEyeOpen' }
};

const HEAD_MOVEMENT_MAP = {
    'nod': { param: 'ParamAngleY', value: 15 },
    'shake': { param: 'ParamAngleX', value: -15 },
    'idle': { param: 'ParamAngleY', value: 0 }
};

export async function initLive2D(mainClock) {
    if (initialized) return true;
    if (typeof PIXI === 'undefined' || typeof PIXI.live2d === 'undefined') {
        logger.error('PIXI.js or PIXI.live2d library not found.');
        return false;
    }
    container = document.getElementById('live2d-container');
    if (!container) {
        logger.error('Live2D container not found.');
        return false;
    }
    clock = mainClock;

    try {
        pixiApp = new PIXI.Application({
            view: document.createElement('canvas'),
            autoStart: true,
            resizeTo: container,
            backgroundAlpha: 0,
        });
        container.appendChild(pixiApp.view);

        const modelPath = 'https://cdn.jsdelivr.net/gh/Live2D/CubismWebSamples@master/Samples/Resources/Hiyori/Hiyori.model3.json';
        live2dModel = await PIXI.live2d.Live2DModel.from(modelPath);
        pixiApp.stage.addChild(live2dModel);

        live2dModel.scale.set(0.12);
        live2dModel.anchor.set(0.5, 0.5);
        live2dModel.position.set(pixiApp.view.width / 2, pixiApp.view.height / 2);

        const resizeObserver = new ResizeObserver(() => {
            if (pixiApp && live2dModel) {
                live2dModel.position.set(pixiApp.view.width / 2, pixiApp.view.height / 2);
            }
        });
        resizeObserver.observe(container);

        initialized = true;
        logger.info('Live2D Initialized successfully.');
        return true;
    } catch (e) {
        logger.error('Failed to initialize Live2D:', e);
        cleanupLive2D();
        return false;
    }
}

// FIX: This function now expects a [1, 6] tensor containing only emotion data.
export function updateLive2DEmotions(emotionTensor, hmLabel = 'idle') {
    if (!initialized || !live2dModel || !emotionTensor || emotionTensor.isDisposed) {
        return;
    }
    try {
        const emotionData = emotionTensor.arraySync();
        if (!emotionData || !emotionData[0]) return;
        
        // FIX: The incoming data is now directly the emotions vector. No slice needed.
        const emotions = emotionData[0]; 
        
        const normalized = emotions.map(e => clamp(e, 0, 1));
        const dominantIdx = normalized.indexOf(Math.max(...normalized));
        if (dominantIdx === -1) return;

        const dominantEmotion = EMOTION_MAP[dominantIdx];

        if (dominantEmotion && dominantEmotion.expression && live2dModel.expression !== dominantEmotion.expression) {
            live2dModel.expression(dominantEmotion.expression);
        }

        if (dominantEmotion && dominantEmotion.param) {
            const paramValue = dominantEmotion.name === 'fear' || dominantEmotion.name === 'surprise' 
                ? normalized[dominantIdx] * 1.0 // Eye open range is 0-1
                : normalized[dominantIdx];
            live2dModel.internalModel.coreModel.setParameterValueById(dominantEmotion.param, paramValue);
        }

        const movement = hmLabel || (normalized[5] > 0.7 ? 'nod' : 'idle');
        const moveData = HEAD_MOVEMENT_MAP[movement];
        if (moveData) {
            live2dModel.internalModel.coreModel.setParameterValueById(moveData.param, moveData.value);
        }
    } catch (e) {
        // FIX: Improved logging for better diagnostics
        logger.error("Error updating Live2D emotions:", { message: e.message, stack: e.stack, error: e });
    }
}

export function updateLive2D(deltaTime) {
    if (!initialized || !live2dModel) return;
    // The PIXI.Application ticker handles rendering. We just update params.
    const breath = (Math.sin(Date.now() / 1000) + 1) / 2;
    live2dModel.internalModel.coreModel.setParameterValueById('ParamBreath', breath);
}

export function cleanupLive2D() {
    if (pixiApp) {
        pixiApp.destroy(true, { children: true, texture: true, baseTexture: true });
        pixiApp = null;
    }
    if (container) container.innerHTML = '';
    live2dModel = null;
    initialized = false;
    clock = null;
    logger.info('Live2D resources cleaned up.');
}

export { initialized as live2dInitialized };
export function isLive2DReady() { return initialized; }
// --- END OF FILE viz-live2d.js ---