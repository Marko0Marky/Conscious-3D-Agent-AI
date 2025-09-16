import { logger, clamp } from './utils.js';

let pixiApp;
let live2dModel;
let container;
let initialized = false;
let clock;
let sheafInstance = null; // Reference to the sheaf for Phi

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

function getDynamicScale(containerWidth) {
    // Base scale for desktop (container width ~330px)
    const baseScale = 0.12;
    // Scale down proportionally for smaller screens
    const scaleFactor = Math.min(containerWidth / 330, 1); // Normalize to desktop width
    return clamp(baseScale * scaleFactor, 0.03, 0.12); // Min scale 0.03 for mobile, max 0.06
}

export async function initLive2D(mainClock, sheaf) {
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
    sheafInstance = sheaf;

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

        // Set initial scale based on container width
        const containerWidth = container.offsetWidth;
        live2dModel.scale.set(getDynamicScale(containerWidth));
        live2dModel.anchor.set(0.5, 0.5);
        live2dModel.position.set(pixiApp.view.width / 2, pixiApp.view.height / 2);

        const resizeObserver = new ResizeObserver(() => {
            if (pixiApp && live2dModel) {
                const containerWidth = container.offsetWidth;
                live2dModel.scale.set(getDynamicScale(containerWidth));
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

export function updateLive2DEmotions(emotionTensor, hmLabel = 'idle') {
    if (!initialized || !live2dModel || !emotionTensor || emotionTensor.isDisposed) {
        return;
    }
    try {
        const emotionData = emotionTensor.arraySync();
        if (!emotionData || !emotionData[0]) return;
        
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
                ? normalized[dominantIdx] * 1.0
                : normalized[dominantIdx];
            live2dModel.internalModel.coreModel.setParameterValueById(dominantEmotion.param, paramValue);
        }

        const movement = hmLabel || (normalized[5] > 0.7 ? 'nod' : 'idle');
        const moveData = HEAD_MOVEMENT_MAP[movement];
        if (moveData) {
            live2dModel.internalModel.coreModel.setParameterValueById(moveData.param, moveData.value);
        }
    } catch (e) {
        logger.error("Error updating Live2D emotions:", { message: e.message, stack: e.stack, error: e });
    }
}

export function updateLive2D(deltaTime) {
    if (!initialized || !live2dModel || !sheafInstance) return;
    
    const phiNormalized = clamp((sheafInstance.phi || 0.01) / 5.0, 0, 1);
    const breath = (Math.sin(Date.now() / 1000) + 1) / 2;
    const phiDrivenBreath = clamp(0.5 + breath * phiNormalized * 0.5, 0.2, 0.9);

    live2dModel.internalModel.coreModel.setParameterValueById('ParamBreath', phiDrivenBreath);
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
    sheafInstance = null;
    logger.info('Live2D resources cleaned up.');
}

export { initialized as live2dInitialized };
export function isLive2DReady() { return initialized; }
