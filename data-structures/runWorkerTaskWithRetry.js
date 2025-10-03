export async function runWorkerTaskWithRetry(type, data, timeout = 20000, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            return await runWorkerTask(type, data, timeout);
        } catch (err) {
            logger.warn(`Attempt ${i + 1} failed for task ${type}: ${err.message}`);
            if (i === retries - 1) {
                logger.error(`All ${retries} attempts failed for task ${type}.`, { error: err.message, stack: err.stack });
                return type === 'matVecMul' ? vecZeros(data.expectedDim || 7) :
                       type === 'matMul' ? identity(data.rows || 7) :
                       type === 'complexEigenvalues' ? Array(data.dim || 1).fill({ re: 1, im: 0 }) :
                       type === 'ksg_mi' ? 0 :
                       type === 'topologicalScore' ? 0 : null;
            }
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
}